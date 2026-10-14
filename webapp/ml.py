"""Bridge between the web app and the trained LightGBM models: live scoring of
clone sessions, plain-language evidence, and supplier-facing solutions."""

import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import PRODUCT_COLS, session_features  # noqa: E402

from .db import ROOT  # noqa: E402

FRICTIONS = ["search", "category_nav", "media", "description", "reviews", "warranty_returns",
             "pricing", "offers", "delivery", "payment"]
CLASSES = FRICTIONS + ["none"]
CATEGORIES = ["beauty", "electronics", "fashion", "furniture", "grocery"]

LABEL = {
    "search": "Search can't find it", "category_nav": "Hard to find in categories",
    "media": "Weak photos / no video", "description": "Unclear title or description",
    "reviews": "Reviews & ratings put buyers off", "warranty_returns": "Warranty / return policy concerns",
    "pricing": "Price higher than alternatives", "offers": "Missing or failing offers",
    "delivery": "Delivery not available or too slow", "payment": "Payment method missing / failing",
    "none": "Low purchase intent",
}

SOLUTION = {
    "search": "Add the words shoppers actually type (synonyms, local terms, common misspellings) to the listing keywords.",
    "category_nav": "Move the product to the category shoppers browse for it and add it to related collections.",
    "media": "Upload at least 6 photos (all angles, in-use, scale) and a short video.",
    "description": "Complete the specs: size/dimensions, material, compatibility, what's in the box.",
    "reviews": "Ask recent buyers for reviews, reply to negative ones, and fix the quality issue they mention.",
    "warranty_returns": "Offer a clear 7–10 day return window and state warranty terms on the product page.",
    "pricing": "Bring price within ~2% of comparable listings, or bundle value (free delivery / gift).",
    "offers": "Attach a working offer (bank / cashback / coupon) so shoppers who look for deals find one.",
    "delivery": "Extend delivery to tier-3 pincodes (or via partner logistics) and cut the delivery estimate.",
    "payment": "Enable Cash-on-Delivery and EMI on this product; these are the methods shoppers fell back from.",
    "none": "No product problem detected — shoppers were browsing. Wishlist reminders work better than discounts.",
}

# Customer-facing live nudges from the recovery assistant
NUDGE = {
    "search": ("Can't find it?", "Try a simpler word — e.g. 'sofa', 'earbuds', 'kurta' — or browse top picks.", "/"),
    "category_nav": ("Looking for something specific?", "Search usually finds it faster than menus.", "/search?q="),
    "media": ("Want a closer look?", "Ask the seller for more photos — answers usually arrive within 2 hours.", None),
    "description": ("Have a question about this item?", "Ask the seller about size, material or compatibility.", None),
    "reviews": ("Unsure about quality?", "Every order is covered by our purchase-protection guarantee.", None),
    "warranty_returns": ("Worried about returns?", "Damaged or not as described? Full refund under purchase protection.", None),
    "pricing": ("A little something for you", "Use code SAVE5 for an extra 5% off at checkout.", None),
    "offers": ("Coupon not working?", "SAVE5 works on every order — 5% off, no minimum.", None),
    "delivery": ("Delivery trouble?", "Try a nearby pincode, or browse items that deliver to you fast.", "/"),
    "payment": ("Payment didn't go through?", "UPI is succeeding fastest right now. Your cart is saved.", None),
}

# One-click fixes a supplier can apply (field -> new value or callable on the product row)
QUICK_FIX = {
    "search": {"keyword_coverage": 0.95},
    "category_nav": {"miscategorized": 0},
    "media": {"image_count": 6, "has_video": 1},
    "description": {"desc_quality": 0.92},
    "warranty_returns": {"return_window_days": lambda p: max(10, p["return_window_days"]),
                         "warranty_months": lambda p: max(12, p["warranty_months"])
                         if p["category"] in ("electronics", "furniture") else p["warranty_months"]},
    "pricing": {"price": lambda p: round(p["price"] / p["price_vs_market"] * 0.98, 2), "price_vs_market": 0.98},
    "offers": {"has_offer": 1, "cashback_pct": lambda p: max(5, p["cashback_pct"] or 0)},
    "delivery": {"serviceable_tiers": "tier1,tier2,tier3", "eta_days": lambda p: round(max(1, p["eta_days"] * 0.6), 1)},
    "payment": {"cod_allowed": 1, "emi_allowed": lambda p: 1 if p["price"] > 3000 else p["emi_allowed"]},
}


def product_diagnostics(p, cause):
    """Concrete, attribute-level facts that back up a cause for this specific product."""
    facts = {
        "search": [f"Listing matches only {p['keyword_coverage']:.0%} of shopper vocabulary"
                   + (" — synonyms like the local term aren't indexed" if p["keyword_coverage"] < 0.7 else "")],
        "category_nav": ["Product is listed under the wrong category" if p["miscategorized"]
                         else "Category placement looks fine; shoppers still struggled to reach it"],
        "media": [f"{p['image_count']} photo(s)" + ("" if p["has_video"] else ", no video")],
        "description": [f"Description completeness {p['desc_quality']:.0%}"],
        "reviews": [f"Rating {p['rating']:.1f}★ from {p['review_count']} reviews"],
        "warranty_returns": [f"Return window: {p['return_window_days'] or 'no returns'}"
                             + (" days" if p["return_window_days"] else "")
                             + (f", warranty: {p['warranty_months']} months"
                                if p["category"] in ("electronics", "furniture") else "")],
        "pricing": [f"Priced {p['price_vs_market'] - 1:+.0%} vs comparable listings"],
        "offers": ["No active offer" if not p["has_offer"] else f"Offer active, cashback {p['cashback_pct']:.0f}%"],
        "delivery": [f"Delivers to: {p['serviceable_tiers'].replace('tier', 'tier-')}",
                     f"Delivery estimate {p['eta_days']:.1f} days"],
        "payment": [f"COD {'available' if p['cod_allowed'] else 'not available'}, "
                    f"EMI {'available' if p['emi_allowed'] else 'not available'}"],
    }
    return facts.get(cause, [])


def apply_quick_fix(p, cause):
    changes = {}
    for field, val in QUICK_FIX.get(cause, {}).items():
        new = val(p) if callable(val) else val
        if new != p[field]:
            changes[field] = {"from": p[field], "to": new}
    return changes


# ---------------- plain-language evidence ----------------
EVIDENCE_TEXT = {
    "zero_result_searches": lambda v: f"got {v:.0f} zero-result search(es)",
    "max_clicked_rank": lambda v: f"had to scroll to result #{v:.0f}",
    "n_search_refined": lambda v: f"rephrased the search {v:.0f} time(s)",
    "n_category_viewed": lambda v: f"hopped through {v:.0f} category pages",
    "n_nav_back": lambda v: f"went back {v:.0f} time(s) while browsing",
    "max_clicked_position": lambda v: f"found it only at position #{v:.0f} in the listing",
    "n_gallery_swipe": lambda v: f"swiped the gallery {v:.0f} times",
    "n_image_zoom": lambda v: f"zoomed into images {v:.0f} times",
    "min_video_pct": lambda v: f"stopped the video at {v:.0f}%",
    "n_section_viewed": lambda v: f"re-read specs/description {v:.0f} times",
    "n_qna_opened": lambda v: "opened Q&A looking for missing details",
    "n_qna_searched": lambda v: "searched Q&A for an answer",
    "n_compare_added": lambda v: "added it to compare with alternatives",
    "n_reviews_sorted": lambda v: "sorted reviews by lowest rating",
    "n_review_filter": lambda v: "filtered to 1-star reviews",
    "review_ms": lambda v: f"spent {v / 1000:.0f}s reading reviews",
    "min_pdp_rating": lambda v: f"saw a {v:.1f}★ rating",
    "n_policy_viewed": lambda v: f"checked the return/warranty policy {v:.0f} time(s)",
    "no_warranty_seen": lambda v: "saw that there is no warranty",
    "min_return_window_seen": lambda v: "saw no return option" if v == 0 else f"saw a {v:.0f}-day return window",
    "n_help_opened": lambda v: "opened help from the product page",
    "n_price_history_opened": lambda v: "checked price history",
    "n_similar_product_viewed": lambda v: f"viewed {v:.0f} similar product(s)",
    "min_price_delta_pct": lambda v: f"found an alternative {abs(v):.0f}% cheaper",
    "fees": lambda v: f"saw ₹{v:.0f} in extra fees",
    "n_fee_breakdown_viewed": lambda v: "opened the fee breakdown in cart",
    "n_remove_from_cart": lambda v: "removed the item from cart",
    "coupon_fail": lambda v: f"had {v:.1g} coupon attempt(s) fail",
    "n_coupon_applied": lambda v: f"tried {v:.0f} coupon(s)",
    "n_offers_viewed": lambda v: "looked for offers",
    "n_bank_offer_toggled": lambda v: "toggled bank offers",
    "pincode_unserviceable": lambda v: "found their pincode unserviceable",
    "n_pincode_changed": lambda v: f"tried {v:.1g} other pincode(s)",
    "n_pincode_check": lambda v: "checked delivery to their pincode",
    "max_eta_seen": lambda v: f"saw a {v:.1f}-day delivery estimate",
    "n_delivery_slot_viewed": lambda v: "looked for an earlier delivery slot",
    "n_address_not_serviceable": lambda v: "had their address rejected at checkout",
    "pay_fail": lambda v: f"had {v:.1g} payment attempt(s) fail",
    "pay_switch": lambda v: f"switched payment method {v:.0f} time(s)",
    "n_payment_method_unavailable": lambda v: "couldn't use their preferred payment method",
    "secs_payment": lambda v: f"spent {v:.0f}s on the payment page",
    "tab_hidden_secs": lambda v: f"left the tab for {v:.0f}s (likely comparing elsewhere)",
    "n_rage_click": lambda v: f"rage-clicked {v:.0f} time(s)",
    "prod_price_vs_market": lambda v: f"product is priced {v - 1:+.0%} vs market",
    "prod_image_count": lambda v: f"listing has only {v:.0f} photo(s)",
    "prod_has_video": lambda v: "listing has no video" if v == 0 else "listing has a video",
    "prod_desc_quality": lambda v: f"description completeness {v:.0%}",
    "prod_rating": lambda v: f"product rated {v:.1f}★",
    "prod_return_window_days": lambda v: "product has no returns" if v == 0 else f"{v:.0f}-day returns",
    "prod_eta_days": lambda v: f"delivery estimate {v:.1f} days",
    "prod_cod_allowed": lambda v: "COD not offered" if v == 0 else "COD offered",
    "prod_keyword_coverage": lambda v: f"listing keywords cover {v:.0%} of search terms",
    "prod_has_offer": lambda v: "no active offer" if v == 0 else "offer active",
    "region_tier": lambda v: f"shopper is in a tier-{v:.0f} city",
}


def describe(feature, value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    fn = EVIDENCE_TEXT.get(feature)
    if fn:
        try:
            return fn(float(value))
        except (TypeError, ValueError):
            return None
    return None


# ---------------- models ----------------
class Models:
    def __init__(self, model_dir=os.path.join(ROOT, "models")):
        self.cause = lgb.Booster(model_file=os.path.join(model_dir, "cause_model.txt"))
        self.risk = lgb.Booster(model_file=os.path.join(model_dir, "risk_model.txt"))
        self.cause_cols = self.cause.feature_name()
        self.risk_cols = self.risk.feature_name()

    def _align(self, X, cols):
        X = X.reindex(columns=cols)
        for c in cols:
            if c.startswith("n_"):
                X[c] = X[c].fillna(0)
        X["category"] = pd.Categorical(X["category"].astype(object), categories=CATEGORIES)
        return X

    def score_session(self, events, product, tier):
        """events: list of DB rows for one session. Returns risk, cause, probabilities and evidence."""
        if not events:
            return None
        ev = pd.DataFrame(events)
        ev["ts"] = pd.to_datetime(ev.ts, format="ISO8601")
        ev = ev.sort_values("ts")
        if ev.category.isna().all():
            ev["category"] = product["category"] if product else "electronics"
        X = session_features(ev, "session_id")
        X["region_tier"] = tier
        if product:
            for c in PRODUCT_COLS:
                X["prod_" + c] = float(product[c]) if product[c] is not None else np.nan
        X = X.drop(columns=["anonymous_id", "last_product_id"], errors="ignore")
        Xr = self._align(X.copy(), self.risk_cols)
        Xc = self._align(X.copy(), self.cause_cols)
        risk = float(self.risk.predict(Xr)[0])
        probs = self.cause.predict(Xc)[0]
        fr = probs[:len(FRICTIONS)]
        k = int(fr.argmax())
        cause = FRICTIONS[k]
        contrib = self.cause.predict(Xc, pred_contrib=True)[0]
        nf = len(self.cause_cols) + 1
        own = contrib[k * nf:(k + 1) * nf - 1]
        evidence = []
        for i in np.argsort(-own):
            if own[i] <= 0 or len(evidence) >= 4:
                break
            col = self.cause_cols[i]
            txt = describe(col, Xc[col].iloc[0] if col != "category" else None)
            if txt:
                evidence.append({"feature": col, "text": txt, "weight": round(float(own[i]), 3)})
        return {"risk": risk, "cause": cause, "cause_conf": float(fr[k] / max(fr.sum(), 1e-9)),
                "p_none": float(probs[-1]), "probs": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
                "evidence": evidence, "n_events": int(len(ev))}


_models = None


def models():
    global _models
    if _models is None:
        _models = Models()
    return _models


def tier_for_pincode(pincode):
    from synth.config import PINCODE_PREFIX
    if pincode:
        for tier, prefixes in PINCODE_PREFIX.items():
            if pincode[:2] in prefixes:
                return int(tier[-1])
    return 2


def evidence_json(ev):
    return json.dumps(ev)
