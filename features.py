"""Clickstream -> one feature row per session (or per session prefix).

Only observable data is used: events, plus catalog attributes of the products the
user looked at and the user's region tier. Nothing from labels, and no latent
fields (persona, intent, sensitivities).
"""

import json

import numpy as np
import pandas as pd

# Events that only exist because of the outcome -> would leak the target.
LEAKY = {"page_exit", "order_placed"}

STAGE = {"home": 0, "search": 1, "category": 1, "pdp": 2, "cart": 3, "checkout": 4,
         "payment": 5, "order_confirmation": 6}

PRODUCT_COLS = ["price", "price_vs_market", "image_count", "has_video", "desc_quality", "rating",
                "review_count", "warranty_months", "return_window_days", "has_offer", "cashback_pct",
                "eta_days", "cod_allowed", "emi_allowed", "miscategorized", "keyword_coverage"]

# Event types that carry props we need; parse JSON only for these rows (speed).
PROP_EVENTS = {"search_submitted", "search_refined", "search_result_clicked", "product_clicked",
               "pincode_check", "pincode_changed", "delivery_estimate_shown", "delivery_slot_viewed",
               "coupon_applied", "similar_product_viewed", "fee_breakdown_viewed", "payment_result",
               "payment_method_selected", "review_dwell", "video_pause", "policy_viewed",
               "scroll_depth", "page_view"}


def _props(ev):
    out = pd.Series([{}] * len(ev), index=ev.index, dtype=object)
    m = ev.event_type.isin(PROP_EVENTS)
    out[m] = ev.props[m].map(json.loads)
    return out


def _get(p, key):
    return p.map(lambda d: d.get(key))


def session_features(ev, key="session_id"):
    """ev: events with a grouping column `key` (session_id or a prefix id), sorted by ts."""
    ev = ev[~ev.event_type.isin(LEAKY)].copy()
    p = _props(ev)
    et = ev.event_type
    rc = pd.to_numeric(_get(p, "results_count"), errors="coerce")
    f = pd.DataFrame(index=ev.index)
    f["zero_result_searches"] = et.isin(["search_submitted", "search_refined"]) & (rc == 0)
    f["clicked_rank"] = pd.to_numeric(_get(p, "rank"), errors="coerce")
    f["clicked_position"] = pd.to_numeric(_get(p, "position"), errors="coerce")
    serv = _get(p, "serviceable")
    f["pincode_unserviceable"] = et.isin(["pincode_check", "pincode_changed"]) & (serv == False)  # noqa: E712
    f["eta_seen"] = pd.to_numeric(_get(p, "eta_days").fillna(_get(p, "earliest_days")), errors="coerce")
    succ = _get(p, "success")
    f["coupon_fail"] = (et == "coupon_applied") & (succ == False)  # noqa: E712
    f["coupon_ok"] = (et == "coupon_applied") & (succ == True)  # noqa: E712
    f["price_delta_pct"] = pd.to_numeric(_get(p, "price_delta_pct"), errors="coerce")
    f["fees"] = pd.to_numeric(_get(p, "delivery_fee"), errors="coerce").fillna(0) + \
        pd.to_numeric(_get(p, "platform_fee"), errors="coerce").fillna(0)
    status = _get(p, "status")
    f["pay_fail"] = (et == "payment_result") & (status == "failed")
    f["pay_switch"] = (et == "payment_method_selected") & (_get(p, "switched") == True)  # noqa: E712
    f["review_ms"] = pd.to_numeric(_get(p, "ms"), errors="coerce")
    f["video_pct"] = pd.to_numeric(_get(p, "watched_pct"), errors="coerce")
    f["return_window_seen"] = pd.to_numeric(_get(p, "return_window_days"), errors="coerce")
    f["no_warranty_seen"] = (et == "policy_viewed") & (_get(p, "warranty_months") == 0)
    f["scroll_pct"] = pd.to_numeric(_get(p, "pct"), errors="coerce")
    f["pdp_rating"] = pd.to_numeric(_get(p, "rating"), errors="coerce").where(ev.page_type == "pdp")
    gap = ev.groupby(key).ts.diff().dt.total_seconds()
    f["tab_hidden_secs"] = gap.where(et == "tab_visible")
    f["stage"] = ev.page_type.map(STAGE)
    for pt in ("search", "category", "pdp", "cart", "checkout", "payment"):
        f[f"secs_{pt}"] = gap.where(ev.page_type == pt)
    f[key] = ev[key].values

    g = f.groupby(key)
    X = pd.concat([
        g[["zero_result_searches", "pincode_unserviceable", "coupon_fail", "coupon_ok", "pay_fail",
           "pay_switch", "no_warranty_seen", "fees", "review_ms", "tab_hidden_secs"]
          + [c for c in f if c.startswith("secs_")]].sum(),
        g[["clicked_rank", "clicked_position", "eta_seen", "scroll_pct", "stage"]].max().add_prefix("max_"),
        g[["price_delta_pct", "video_pct", "return_window_seen", "pdp_rating"]].min().add_prefix("min_"),
    ], axis=1)
    g2 = ev.groupby(key)
    X["n_events"] = g2.size()
    X["session_secs"] = (g2.ts.max() - g2.ts.min()).dt.total_seconds()
    X["hour"] = g2.ts.min().dt.hour
    X["logged_in"] = g2.user_id.first().notna().astype(int)
    X["mobile"] = (g2.device.first() == "mobile").astype(int)
    X["category"] = g2.category.first()
    X["anonymous_id"] = g2.anonymous_id.first()
    X["n_products"] = g2.product_id.nunique()
    X["last_product_id"] = ev.dropna(subset=["product_id"]).groupby(key).product_id.last()
    counts = pd.crosstab(ev[key], et).add_prefix("n_")
    X = X.join(counts)
    return X


def add_context(X, users, products):
    """Join observable context: region tier (from pincode/IP) and catalog attributes of the last product."""
    tier = users.set_index("anonymous_id").region_tier.map({"tier1": 1, "tier2": 2, "tier3": 3})
    X["region_tier"] = X.anonymous_id.map(tier)
    prod = products.set_index("product_id")[PRODUCT_COLS].astype(float).add_prefix("prod_")
    X = X.join(prod, on="last_product_id")
    X["category"] = X.category.astype(pd.CategoricalDtype(sorted(products.category.unique())))
    return X.drop(columns=["anonymous_id", "last_product_id"])


def prefix_events(ev, rng, per_session=2, min_len=3):
    """Cut each session at random points (before its final event) -> early-warning training rows."""
    ev = ev[~ev.event_type.isin(LEAKY)].copy()
    ev["seq"] = ev.groupby("session_id").cumcount()
    n = ev.groupby("session_id").size()
    n = n[n > min_len]
    reps = np.repeat(n.index.values, per_session)
    lens = np.repeat(n.values, per_session)
    cuts = (min_len + rng.random(len(reps)) * (lens - min_len)).astype(int)
    cuts = pd.DataFrame({"session_id": reps, "cut": cuts}).drop_duplicates()
    cuts["prefix_id"] = cuts.session_id + "@" + cuts.cut.astype(str)
    ev = ev.merge(cuts, on="session_id")
    return ev[ev.seq < ev.cut].drop(columns=["seq", "cut"])
