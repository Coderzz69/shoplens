"""Train the friction models on the synthetic dataset.

    .venv/bin/python train.py --data data --out models

Model 1  cause  : abandoned session -> which friction caused it (10 frictions + none)
Model 2  risk   : session prefix (in progress) -> probability the user will drop
Explanations    : LightGBM TreeSHAP (pred_contrib) grouped by friction
Sanity check    : predicted cause mix per category vs the W matrix used to generate data
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             roc_auc_score, top_k_accuracy_score)

from features import add_context, prefix_events, session_features

FRICTIONS = ["search", "category_nav", "media", "description", "reviews", "warranty_returns",
             "pricing", "offers", "delivery", "payment"]
CLASSES = FRICTIONS + ["none"]

RECOVERY = {
    "search": "Fix zero-result queries: add synonyms/typo tolerance; show 'did you mean' + top sellers",
    "category_nav": "Re-categorize the product; add it to the category the user browsed",
    "media": "Add more images / a video; prompt seller; show customer photos",
    "description": "Auto-fill missing specs/size chart; surface Q&A answers on the page",
    "reviews": "Surface verified-buyer highlights; offer trial / quality guarantee badge",
    "warranty_returns": "Show return & warranty policy upfront; offer easy-return or extended warranty",
    "pricing": "Price-match nudge or targeted coupon; show fee breakdown earlier / free-delivery threshold",
    "offers": "Auto-apply the best valid coupon; explain why a coupon failed and suggest one that works",
    "delivery": "Offer alternate seller / faster shipping; notify when pincode becomes serviceable",
    "payment": "Suggest working method (UPI/COD from another seller), retry via backup gateway, enable EMI",
    "none": "Low intent: save-for-later / wishlist reminder, no discount needed",
}

# feature -> friction group, for grouping SHAP values into business causes
GROUPS = {
    "search": ["zero_result_searches", "max_clicked_rank", "n_search_", "prod_keyword_coverage"],
    "category_nav": ["n_category_viewed", "n_nav_back", "max_clicked_position", "n_filter_", "prod_miscategorized",
                     "secs_category", "n_product_clicked"],
    "media": ["n_gallery_swipe", "n_image_zoom", "n_video_", "min_video_pct", "prod_image_count", "prod_has_video"],
    "description": ["n_section_viewed", "n_qna_", "n_compare_added", "prod_desc_quality"],
    "reviews": ["n_reviews_", "n_review_", "review_ms", "min_pdp_rating", "prod_rating", "prod_review_count"],
    "warranty_returns": ["n_policy_viewed", "no_warranty_seen", "min_return_window_seen", "prod_warranty_months",
                         "prod_return_window_days", "n_help_opened"],
    "pricing": ["n_price_history_opened", "n_similar_product_viewed", "min_price_delta_pct", "fees",
                "n_fee_breakdown_viewed", "prod_price", "n_remove_from_cart"],
    "offers": ["coupon_fail", "coupon_ok", "n_coupon_applied", "n_offers_viewed", "n_bank_offer_toggled",
               "prod_has_offer", "prod_cashback_pct"],
    "delivery": ["pincode_unserviceable", "n_pincode_", "max_eta_seen", "n_delivery_", "n_address_",
                 "prod_eta_days", "region_tier"],
    "payment": ["pay_fail", "pay_switch", "n_payment_", "prod_cod_allowed", "prod_emi_allowed", "secs_payment"],
}


def group_of(col):
    for g, keys in GROUPS.items():
        if any(col == k or (k.endswith("_") and col.startswith(k)) for k in keys):
            return g
    return "journey/context"


def _features_for_part(job):
    path, mode, seed = job
    ev = pd.read_parquet(path).sort_values(["session_id", "ts"])
    if mode == "full":
        return session_features(ev, "session_id")
    pe = prefix_events(ev, np.random.default_rng(seed), per_session=1)
    X = session_features(pe, "prefix_id")
    X["session_id"] = X.index.str.split("@").str[0]
    return X


def build(data, mode, workers):
    parts = sorted(os.path.join(data, "events", f) for f in os.listdir(os.path.join(data, "events")))
    with ProcessPoolExecutor(workers) as ex:
        frames = list(ex.map(_features_for_part, [(p, mode, i) for i, p in enumerate(parts)]))
    X = pd.concat(frames)
    n_cols = [c for c in X if c.startswith("n_")]
    X[n_cols] = X[n_cols].fillna(0)
    return X


def _val(v):
    if pd.isna(v):
        return None
    return float(v) if isinstance(v, (int, float, np.number, bool, np.bool_)) else str(v)


def split_by_day(day):
    return np.where(day < 24, "train", np.where(day < 27, "val", "test"))


def train_lgb(Xtr, ytr, Xva, yva, params, w=None):
    dtr = lgb.Dataset(Xtr, ytr, weight=w)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    return lgb.train(params, dtr, 2000, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--none-cap", type=int, default=60000, help="max 'none' rows in train (class balance)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    labels = pd.read_parquet(f"{args.data}/labels.parquet").set_index("session_id")
    users = pd.read_parquet(f"{args.data}/users.parquet")
    products = pd.read_parquet(f"{args.data}/products.parquet")
    day = (labels.start_ts - labels.start_ts.min()).dt.days
    report = {}

    # ---------------- Model 1: cause of abandonment ----------------
    print("Building session features ...", flush=True)
    X = add_context(build(args.data, "full", args.workers), users, products)
    print(f"  {X.shape[0]:,} sessions x {X.shape[1]} features  ({time.time() - t0:.0f}s)")
    ab = labels[~labels.converted & labels.primary_cause.notna()]
    Xa = X.loc[ab.index]
    y = ab.primary_cause.map({c: i for i, c in enumerate(CLASSES)}).values
    split = split_by_day(day.loc[ab.index].values)
    rng = np.random.default_rng(0)
    tr = split == "train"
    none_idx = np.where(tr & (y == CLASSES.index("none")))[0]
    if len(none_idx) > args.none_cap:
        tr[rng.choice(none_idx, len(none_idx) - args.none_cap, replace=False)] = False
    va, te = split == "val", split == "test"
    freq = np.bincount(y[tr], minlength=len(CLASSES))
    w = (freq.max() / freq[y[tr]]) ** 0.5  # sqrt inverse frequency
    params = dict(objective="multiclass", num_class=len(CLASSES), learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, verbose=-1, num_threads=args.workers)
    print(f"Training cause model on {tr.sum():,} sessions ...", flush=True)
    cause = train_lgb(Xa[tr], y[tr], Xa[va], y[va], params, w)
    cause.save_model(f"{args.out}/cause_model.txt")
    P = cause.predict(Xa[te])
    pred = P.argmax(1)
    yt = y[te]
    fr = yt != CLASSES.index("none")
    report["cause"] = {
        "test_sessions": int(te.sum()), "best_iteration": cause.best_iteration,
        "accuracy": accuracy_score(yt, pred), "macro_f1": f1_score(yt, pred, average="macro"),
        "top2_accuracy": top_k_accuracy_score(yt, P, k=2, labels=range(len(CLASSES))),
        "friction_only_accuracy": accuracy_score(yt[fr], pred[fr]),
        "per_class": classification_report(yt, pred, labels=range(len(CLASSES)), target_names=CLASSES,
                                           output_dict=True, zero_division=0),
    }
    print(classification_report(yt, pred, labels=range(len(CLASSES)), target_names=CLASSES, digits=3,
                                zero_division=0))
    cm = pd.DataFrame(confusion_matrix(yt, pred, labels=range(len(CLASSES))), index=CLASSES, columns=CLASSES)
    cm.to_csv(f"{args.out}/cause_confusion.csv")

    # ---------------- W recovery check ----------------
    # For each friction: does its share of abandonments rank the 5 categories the same way W does?
    # (Comparing across categories cancels out how common each friction's root problem is overall.)
    W = json.load(open(f"{args.data}/friction_weights.json"))["weights"]
    cats = list(W)
    te_cat = Xa[te].category.astype(str).values

    def share_matrix(lab):
        m = np.zeros((len(cats), len(FRICTIONS)))
        for i, cat in enumerate(cats):
            sel = te_cat == cat
            m[i] = np.bincount(lab[sel], minlength=len(CLASSES))[:len(FRICTIONS)] / max(1, sel.sum())
        return m

    Sp, St, Wm = share_matrix(pred), share_matrix(yt), np.array([W[c] for c in cats])
    rec = {}
    for j, fname in enumerate(FRICTIONS):
        rec[fname] = {"spearman_pred_vs_W": spearmanr(Sp[:, j], Wm[:, j]).statistic,
                      "spearman_true_vs_W": spearmanr(St[:, j], Wm[:, j]).statistic,
                      "most_affected_predicted": cats[int(Sp[:, j].argmax())],
                      "most_affected_W": cats[int(Wm[:, j].argmax())]}
    report["w_recovery"] = rec
    report["w_recovery_mean_spearman"] = {
        "pred": float(np.nanmean([v["spearman_pred_vs_W"] for v in rec.values()])),
        "true": float(np.nanmean([v["spearman_true_vs_W"] for v in rec.values()]))}
    pd.DataFrame(Sp, index=cats, columns=FRICTIONS).round(4).to_csv(f"{args.out}/predicted_cause_share.csv")

    # ---------------- Explanations (TreeSHAP) ----------------
    sample = Xa[te].sample(min(3000, int(te.sum())), random_state=0)
    # show examples where a friction (not "none") was diagnosed
    order = np.argsort(cause.predict(sample).argmax(1) == CLASSES.index("none"), kind="stable")
    sample = sample.iloc[order]
    contrib = cause.predict(sample, pred_contrib=True)  # (n, (n_feat+1)*n_class)
    nf = sample.shape[1] + 1
    sp = sample_pred = cause.predict(sample).argmax(1)  # noqa: F841
    own = np.stack([contrib[i, sp[i] * nf:(sp[i] + 1) * nf - 1] for i in range(len(sample))])
    groups = np.array([group_of(c) for c in sample.columns])
    imp = pd.Series(np.abs(own).mean(0), index=sample.columns).sort_values(ascending=False)
    report["top_features"] = imp.head(25).round(4).to_dict()
    report["shap_by_group"] = pd.Series(np.abs(own).mean(0), index=groups).groupby(level=0).sum() \
        .sort_values(ascending=False).round(4).to_dict()
    examples = []
    for i in range(8):
        sid = sample.index[i]
        c = pd.Series(own[i], index=sample.columns)
        top = c.sort_values(ascending=False).head(4)
        examples.append({
            "session_id": sid, "category": str(sample.category.iloc[i]),
            "predicted": CLASSES[sample_pred[i]], "true": ab.primary_cause.loc[sid],
            "confidence": round(float(cause.predict(sample.iloc[[i]]).max()), 3),
            "evidence": {k: {"value": _val(sample[k].iloc[i]),
                             "shap": round(float(v), 3)} for k, v in top.items()},
            "recovery_action": RECOVERY[CLASSES[sample_pred[i]]],
        })
    report["examples"] = examples

    # ---------------- Model 2: early drop risk on prefixes ----------------
    print("Building prefix features ...", flush=True)
    Xp = build(args.data, "prefix", args.workers)
    sid = Xp.pop("session_id")
    Xp = add_context(Xp, users, products)
    yp = (~labels.converted.loc[sid.values]).astype(int).values
    sp_split = split_by_day(day.loc[sid.values].values)
    ptr, pva, pte = (sp_split == s for s in ("train", "val", "test"))
    print(f"Training drop-risk model on {ptr.sum():,} prefixes ...", flush=True)
    risk = train_lgb(Xp[ptr], yp[ptr], Xp[pva], yp[pva],
                     dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1,
                          num_threads=args.workers))
    risk.save_model(f"{args.out}/risk_model.txt")
    pr = risk.predict(Xp[pte])
    by_len = {}
    n_ev = Xp[pte].n_events.values
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 999)):
        m = (n_ev >= lo) & (n_ev < hi)
        if m.sum() > 100 and len(set(yp[pte][m])) == 2:
            by_len[f"{lo}-{hi} events"] = round(roc_auc_score(yp[pte][m], pr[m]), 4)
    report["risk"] = {"test_prefixes": int(pte.sum()), "auc": roc_auc_score(yp[pte], pr),
                      "auc_by_prefix_length": by_len, "base_drop_rate": float(yp[pte].mean())}

    report["feature_columns"] = list(X.columns)
    with open(f"{args.out}/report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    summary(report, time.time() - t0)


def summary(r, secs):
    c, k = r["cause"], r["risk"]
    print(f"\n==== Done in {secs:.0f}s ====")
    print(f"CAUSE  acc {c['accuracy']:.3f} | macro-F1 {c['macro_f1']:.3f} | top-2 {c['top2_accuracy']:.3f} | "
          f"friction-only acc {c['friction_only_accuracy']:.3f}  (test n={c['test_sessions']:,})")
    print(f"RISK   AUC {k['auc']:.3f}  by prefix length {k['auc_by_prefix_length']}")
    m = r["w_recovery_mean_spearman"]
    print(f"\nW recovery, per friction across categories (Spearman pred vs W | true vs W); "
          f"mean {m['pred']:.2f} | {m['true']:.2f}")
    for fname, v in r["w_recovery"].items():
        print(f"  {fname:<17} {v['spearman_pred_vs_W']:+.2f} | {v['spearman_true_vs_W']:+.2f}   "
              f"most affected: predicted {v['most_affected_predicted']:<12} W {v['most_affected_W']}")
    print("\nSHAP importance by friction group:", r["shap_by_group"])
    print("\nExample explanations:")
    for e in r["examples"][:4]:
        print(f"  {e['session_id']} [{e['category']}] predicted={e['predicted']} ({e['confidence']}) "
              f"true={e['true']}")
        for f_, v in e["evidence"].items():
            print(f"      {f_} = {v['value']}  (shap {v['shap']:+})")
        print(f"      -> {e['recovery_action']}")


if __name__ == "__main__":
    main()
