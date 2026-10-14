"""Build webapp/shop.db: accounts, catalog, reviews, tickets, and model-diagnosed
insights per product (from the 500k synthetic sessions).

    .venv/bin/python -m webapp.seed            # full (≈2-4 min)
    .venv/bin/python -m webapp.seed --fast     # skip insights precompute
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from . import db
from .catalog import description_for, keywords_for, name_for
from .ml import CLASSES, FRICTIONS, describe, models

ROOT = db.ROOT
sys.path.insert(0, ROOT)

SUPPLIERS = [("Nimbus Retail", "supplier1@shop.test"), ("Aurora Traders", "supplier2@shop.test"),
             ("Kestrel Electronics", "supplier3@shop.test"), ("Vanta Lifestyle", "supplier4@shop.test"),
             ("Lumen Home", "supplier5@shop.test"), ("Orchid Beauty Co", "supplier6@shop.test"),
             ("Sahyadri Fresh", "supplier7@shop.test"), ("Teakwood Furnishings", "supplier8@shop.test"),
             ("Indigo Fashion House", "supplier9@shop.test"), ("Monsoon Mart", "supplier10@shop.test")]
DEMO_PASSWORD = "demo1234"

# category -> (supplier whose listings are weak on operations, supplier weak on content)
SPECIALISTS = {
    "grocery": ("Monsoon Mart", "Sahyadri Fresh"),
    "electronics": ("Nimbus Retail", "Kestrel Electronics"),
    "fashion": ("Vanta Lifestyle", "Indigo Fashion House"),
    "furniture": ("Teakwood Furnishings", "Lumen Home"),
    "beauty": ("Aurora Traders", "Orchid Beauty Co"),
}


def assign_suppliers(products, store_ids):
    """Each category goes to two specialist stores; within it, listings whose weak spots are
    operational (delivery, price, returns, COD) go to one, content-weak listings (photos,
    description, reviews, keywords) to the other -> each store has a distinct problem profile."""
    df = products.copy()
    ops = (np.clip((df.eta_days - 1) / 8, 0, 1) + np.clip((df.price_vs_market - 0.92) / 0.3, 0, 1)
           + (1 - df.has_offer.astype(float)) * 0.5 + (df.return_window_days == 0) * 0.5
           + (1 - df.cod_allowed.astype(float)) * 0.5 + (df.serviceable_tiers.str.count("tier") < 3) * 0.5)
    content = ((1 - np.minimum(df.image_count / 6, 1)) * 0.7 + (1 - df.has_video.astype(float)) * 0.3
               + (1 - df.desc_quality) + np.clip((4.4 - df.rating) / 1.6, 0, 1) + (1 - df.keyword_coverage))
    df["bias"] = ops - content
    owner = {}
    for cat, grp in df.groupby("category"):
        ops_store, content_store = SPECIALISTS[cat]
        cut = grp.bias.median()
        for pid, bias in zip(grp.product_id, grp.bias):
            owner[pid] = store_ids[ops_store if bias >= cut else content_store]
    return owner


def seed_core(data):
    if os.path.exists(db.DB_PATH):
        os.remove(db.DB_PATH)
    db.init()
    pw = db.hash_password(DEMO_PASSWORD)
    with db.tx() as con:
        con.execute("INSERT INTO accounts (email, password_hash, role, name, pincode) VALUES (?,?,?,?,?)",
                    ("customer@shop.test", pw, "customer", "Demo Customer", "560034"))
        con.execute("INSERT INTO accounts (email, password_hash, role, name, pincode) VALUES (?,?,?,?,?)",
                    ("tier3@shop.test", pw, "customer", "Tier-3 Customer", "845101"))
        store_ids = {}
        for store, email in SUPPLIERS:
            cur = con.execute("INSERT INTO accounts (email, password_hash, role, name, store_name) VALUES (?,?,?,?,?)",
                              (email, pw, "supplier", store.split()[0] + " Admin", store))
            store_ids[store] = cur.lastrowid

        products = pd.read_parquet(f"{data}/products.parquet")
        owner = assign_suppliers(products, store_ids)
        rows = []
        for p in products.to_dict("records"):
            rows.append((
                p["product_id"], owner[p["product_id"]], name_for(p["product_id"], p["category"]), p["category"],
                keywords_for(p["product_id"], p["category"], p["keyword_coverage"]), description_for(p),
                p["price"], p["price_vs_market"], p["image_count"], int(p["has_video"]), p["desc_quality"],
                p["rating"], p["review_count"], p["warranty_months"], p["return_window_days"], int(p["has_offer"]),
                p["cashback_pct"], p["eta_days"], p["serviceable_tiers"], int(p["cod_allowed"]),
                int(p["emi_allowed"]), int(p["miscategorized"]), p["keyword_coverage"]))
        con.executemany(f"INSERT INTO products VALUES ({','.join('?' * 23)})", rows)

        rv = pd.read_parquet(f"{data}/reviews.parquet")
        con.executemany("INSERT INTO reviews VALUES (?,?,?,?)",
                        rv[["review_id", "product_id", "stars", "text"]].itertuples(index=False))
        tk = pd.read_parquet(f"{data}/tickets.parquet").dropna(subset=["product_id"])
        con.executemany("INSERT INTO tickets VALUES (?,?,?,?,?,?)",
                        [(r.ticket_id, r.product_id, r.channel, str(r.ts), r.text, r.label_theme)
                         for r in tk.itertuples()])
    print(f"  accounts: 2 customers + {len(SUPPLIERS)} suppliers (password '{DEMO_PASSWORD}')")
    print(f"  products: {len(rows):,}  reviews: {len(rv):,}  tickets: {len(tk):,}")


def seed_insights(data, workers):
    from train import build  # session features for every synthetic session
    from features import add_context

    t0 = time.time()
    labels = pd.read_parquet(f"{data}/labels.parquet").set_index("session_id")
    users = pd.read_parquet(f"{data}/users.parquet")
    products = pd.read_parquet(f"{data}/products.parquet")
    X = add_context(build(data, "full", workers), users, products)
    # "last product viewed" and "did it convert" are observable; the cause is predicted by the model
    last = labels.last_product_id
    stats = pd.DataFrame({"sessions": last.value_counts()})
    stats["converted"] = labels[labels.converted].last_product_id.value_counts()
    ab = labels[~labels.converted & last.notna()].index
    stats["abandoned"] = last.loc[ab].value_counts()
    stats = stats.fillna(0).astype(int)

    m = models()
    Xa = X.loc[ab, m.cause_cols]
    print(f"  scoring {len(Xa):,} abandoned sessions ... ({time.time() - t0:.0f}s)", flush=True)
    P = m.cause.predict(Xa)
    pred = P.argmax(1)
    keep = pred != CLASSES.index("none")
    stats["friction_abandoned"] = pd.Series(last.loc[ab][keep].values).value_counts()
    stats = stats.fillna(0).astype(int)

    Xf, pf, conf = Xa[keep], pred[keep], P[keep].max(1)
    F, nf = len(m.cause_cols), len(m.cause_cols) + 1

    # 1) TreeSHAP on a sample (~15 ms/session) -> which signals drive each cause globally
    rng = np.random.default_rng(0)
    smp = rng.choice(len(Xf), min(4000, len(Xf)), replace=False)
    contrib = m.cause.predict(Xf.iloc[smp], pred_contrib=True)
    cause_w = np.zeros((len(FRICTIONS), F))
    for i, c in zip(range(len(smp)), pf[smp]):
        cause_w[c] += np.clip(contrib[i, c * nf:(c + 1) * nf - 1], 0, None)
    cause_w /= np.maximum(np.bincount(pf[smp], minlength=len(FRICTIONS))[:, None], 1)
    print(f"  SHAP sample done ({time.time() - t0:.0f}s)", flush=True)

    # 2) per (product, cause): how prevalent those signals were in that product's sessions
    pid = last.loc[Xf.index].values
    key_codes, key_index = pd.factorize(pd.Series(list(zip(pid, pf))))
    K = len(key_index)
    V = Xf.drop(columns=["category"]).reindex(columns=m.cause_cols).to_numpy(dtype=float, na_value=np.nan)
    nzm = ~np.isnan(V) & (V != 0)
    nz, sumV, cnt, sconf = np.zeros((K, F)), np.zeros((K, F)), np.zeros(K), np.zeros(K)
    np.add.at(nz, key_codes, nzm)
    np.add.at(sumV, key_codes, np.where(nzm, V, 0))
    np.add.at(cnt, key_codes, 1)
    np.add.at(sconf, key_codes, conf)

    prod_attrs = products.set_index("product_id")
    rows = []
    for j, (prod_id, c) in enumerate(key_index):
        n = int(cnt[j])
        if n < 2:
            continue
        prev = nz[j] / n
        ev = []
        for i in np.argsort(-(cause_w[c] * np.where(np.char.startswith(np.array(m.cause_cols), "prod_"), 1, prev))):
            if cause_w[c, i] <= 0 or len(ev) >= 4:
                break
            col = m.cause_cols[i]
            if col.startswith("prod_"):
                txt, pre = describe(col, float(prod_attrs.loc[prod_id, col[5:]])), ""
            elif col == "region_tier":
                continue
            else:
                if prev[i] < 0.15:
                    continue
                txt, pre = describe(col, sumV[j, i] / nz[j, i]), f"{prev[i]:.0%} of these shoppers "
            if txt:
                ev.append({"feature": col, "text": pre + txt, "weight": round(float(cause_w[c, i] * max(prev[i], .5)), 3)})
        abandoned = int(stats.loc[prod_id, "abandoned"]) if prod_id in stats.index else n
        rows.append((prod_id, FRICTIONS[c], n, n / max(abandoned, 1), float(sconf[j] / n), json.dumps(ev)))
    with db.tx() as con:
        con.executemany("INSERT INTO product_stats VALUES (?,?,?,?,?)",
                        [(i, int(r.sessions), int(r.converted), int(r.abandoned), int(r.friction_abandoned))
                         for i, r in stats.iterrows()])
        con.executemany("INSERT INTO insights VALUES (?,?,?,?,?,?)", rows)
    print(f"  insights: {len(rows):,} (product × cause) rows  ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    print("Seeding database:", db.DB_PATH, flush=True)
    seed_core(args.data)
    if not args.fast:
        seed_insights(args.data, args.workers)


if __name__ == "__main__":
    main()
