"""Generate a synthetic e-commerce journey dataset.

    python generate.py --sessions 50000 --users 10000 --products 2500 --out data

Outputs (Parquet, optionally CSV) in --out:
    users, products, events, payments, orders, tickets, reviews  -> model inputs
    labels                                                       -> hidden ground truth
    friction_weights.json                                        -> the W matrix used
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from synth.config import CATEGORIES, FRICTIONS, TICKET_RATE, W
from synth.entities import make_products, make_users
from synth.simulator import Session
from synth.text import make_reviews, make_ticket


def build_entities(args):
    rng = np.random.default_rng(args.seed)
    return make_users(args.users, rng), make_products(args.products, rng)


def simulate_chunk(job):
    args, start, stop = job
    users, products = build_entities(args)
    by_cat = {c: [p for p in products if p["category"] == c] for c in CATEGORIES}
    rng = np.random.default_rng([args.seed, start])
    # Zipf-ish user activity: some users come back many times
    weights = 1 / np.arange(1, len(users) + 1) ** 0.6
    weights /= weights.sum()
    events, payments, labels, orders, tickets = [], [], [], [], []
    for i in range(start, stop):
        user = users[rng.choice(len(users), p=weights)]
        s = Session(f"s_{i:07d}", user, rng, by_cat)
        lab = s.run()
        events += s.events
        payments += s.payments
        labels.append(lab)
        if lab["converted"]:
            p = s.product
            delayed = rng.random() < 0.08 + 0.02 * p["eta_days"] + (0.07 if user["region_tier"] == "tier3" else 0)
            returned = rng.random() < {"fashion": 0.18, "electronics": 0.06, "furniture": 0.05}.get(p["category"], 0.02)
            orders.append({"order_id": f"o_{s.sid}", "session_id": s.sid, "user_id": user["user_id"],
                           "product_id": p["product_id"], "amount": p["price"], "payment_method": s.method,
                           "placed_ts": s.t, "promised_eta_days": p["eta_days"],
                           "status": "returned" if returned else "delayed" if delayed else "delivered"})
        elif lab["primary_cause"] not in (None, "none") and rng.random() < TICKET_RATE:
            tickets.append(make_ticket(rng, s.sid, user, s.product, lab["primary_cause"],
                                       s.t + pd.Timedelta(minutes=int(rng.integers(2, 600))), s.events))
    # events are the bulk of the data: write them from the worker to keep memory flat
    pd.DataFrame(events).to_parquet(f"{args.out}/events/part-{start:07d}.parquet", index=False)
    return len(events), payments, labels, orders, tickets


def strip_private(rows):
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=50000)
    ap.add_argument("--users", type=int, default=10000)
    ap.add_argument("--products", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="data")
    ap.add_argument("--csv", action="store_true", help="also write CSV files")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(f"{args.out}/events", exist_ok=True)
    for old in os.listdir(f"{args.out}/events"):
        os.remove(f"{args.out}/events/{old}")
    if os.path.exists(f"{args.out}/events.parquet"):
        os.remove(f"{args.out}/events.parquet")  # superseded by the events/ part directory

    users, products = build_entities(args)
    chunk = max(1000, args.sessions // (args.workers * 4))
    jobs = [(args, a, min(a + chunk, args.sessions)) for a in range(0, args.sessions, chunk)]
    out = {k: [] for k in ("payments", "labels", "orders", "tickets")}
    n_events = 0
    with ProcessPoolExecutor(args.workers) as ex:
        for done, (n, *res) in enumerate(ex.map(simulate_chunk, jobs), 1):
            n_events += n
            for k, rows in zip(out, res):
                out[k] += rows
            print(f"\r  chunks {done}/{len(jobs)}  events {n_events:,}", end="", flush=True)

    rng = np.random.default_rng(args.seed + 1)
    reviews = [r for p in products for r in make_reviews(rng, p)]
    tables = {
        "users": pd.DataFrame(strip_private(users)),
        "products": pd.DataFrame(strip_private(products)),
        "payments": pd.DataFrame(out["payments"]),
        "orders": pd.DataFrame(out["orders"]),
        # _true_theme kept in a separate label column name so it is obvious it's ground truth
        "tickets": pd.DataFrame(out["tickets"]).rename(columns={"_true_theme": "label_theme"}),
        "reviews": pd.DataFrame(reviews).rename(columns={"_true_theme": "label_theme"}),
        "labels": pd.DataFrame(out["labels"]),
    }
    for name, df in tables.items():
        df.to_parquet(f"{args.out}/{name}.parquet", index=False)
        if args.csv:
            df.to_csv(f"{args.out}/{name}.csv", index=False)
    if args.csv:
        pd.read_parquet(f"{args.out}/events").to_csv(f"{args.out}/events.csv", index=False)
    with open(f"{args.out}/friction_weights.json", "w") as f:
        json.dump({"frictions": FRICTIONS, "weights": W}, f, indent=2)

    report(tables, n_events, time.time() - t0)


def report(t, n_events, secs):
    lab = t["labels"]
    print(f"\n\nGenerated in {secs:.1f}s")
    print(f"  {'events':<9} {n_events:>9,} rows  (events/ part files)")
    for k, df in t.items():
        print(f"  {k:<9} {len(df):>9,} rows")
    print(f"\nConversion rate: {lab.converted.mean():.1%}")
    print(f"Cart abandonment: {1 - lab.converted.sum() / max(1, lab.reached_cart.sum()):.1%}")
    print("\nPrimary cause counts:")
    print(lab.primary_cause.value_counts().to_string())
    print("\nDrop stage:")
    print(lab[~lab.converted].drop_stage.value_counts(normalize=True).round(3).to_string())
    print("\nPrimary cause by category (share of abandoned sessions):")
    ab = lab[~lab.converted]
    print(pd.crosstab(ab.primary_cause, ab.category, normalize="columns").round(3).to_string())


if __name__ == "__main__":
    main()
