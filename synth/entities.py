"""Users and products. Product attributes are what create friction; each product
gets a precomputed deficit per friction (0 = no problem, 1 = severe)."""

import numpy as np

from .config import CATALOG, CATEGORIES, FRICTIONS, PERSONAS, PINCODE_PREFIX, REGION_TIERS

FI = {f: i for i, f in enumerate(FRICTIONS)}


def _pick(rng, mapping):
    keys = list(mapping)
    p = np.array([mapping[k] for k in keys], dtype=float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


def make_users(n, rng):
    names = list(PERSONAS)
    shares = np.array([PERSONAS[p][0] for p in names])
    users = []
    for i in range(n):
        persona = names[rng.choice(len(names), p=shares / shares.sum())]
        _, sens, patience, pay = PERSONAS[persona]
        tier = "tier3" if persona == "tier3_cod" else _pick(rng, REGION_TIERS)
        sens = np.clip(np.array(sens) + rng.normal(0, 0.08, len(sens)), 0.05, 1.0)
        affinity = rng.dirichlet(np.ones(len(CATEGORIES)) * 0.8)
        users.append({
            "user_id": f"u_{i:06d}",
            "anonymous_id": f"a_{rng.integers(16**10):010x}",
            "persona": persona,
            "region_tier": tier,
            "pincode": rng.choice(PINCODE_PREFIX[tier]) + f"{rng.integers(0, 10000):04d}",
            "device": "mobile" if rng.random() < (0.85 if tier != "tier1" else 0.7) else "desktop",
            "preferred_payment": _pick(rng, pay),
            "patience": float(np.clip(rng.lognormal(np.log(patience), 0.3), 0.25, 4.0)),
            "is_logged_in": bool(rng.random() < 0.6),
            "_sens": sens,
            "_affinity": affinity,
        })
    return users


def make_products(n, rng):
    products = []
    per_cat = n // len(CATEGORIES)
    for cat in CATEGORIES:
        (lo, hi), eta_mu, cod_p, emi_p, t3_p = CATALOG[cat]
        for j in range(per_cat):
            pid = f"p_{cat[:3]}_{j:05d}"
            price = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            image_count = int(rng.choice([1, 2, 3, 4, 5, 6, 8], p=[.08, .1, .15, .2, .2, .15, .12]))
            has_video = bool(rng.random() < (0.1 if cat == "grocery" else 0.35))
            desc_quality = float(rng.beta(4, 2))
            rating = float(np.clip(rng.normal(4.0, 0.45), 1.5, 5.0))
            review_count = int(rng.lognormal(4, 1.6))
            warranty_months = int(rng.choice([0, 6, 12, 24], p=[.3, .2, .4, .1])) if cat in ("electronics", "furniture") else 0
            return_days = int(rng.choice([0, 7, 10, 30], p=[.2, .4, .2, .2])) if cat != "grocery" else 0
            price_vs_market = float(np.clip(rng.normal(1.0, 0.1), 0.75, 1.45))
            has_offer = bool(rng.random() < 0.45)
            cashback_pct = float(rng.choice([0, 0, 5, 10])) if has_offer else 0.0
            eta_days = max(0.25 if cat == "grocery" else 1, float(rng.gamma(2, eta_mu / 2)))
            serviceable = ["tier1", "tier2"] + (["tier3"] if rng.random() < t3_p else [])
            cod_allowed = bool(rng.random() < cod_p)
            emi_allowed = bool(rng.random() < emi_p) and price > 3000
            miscategorized = bool(rng.random() < 0.12)
            keyword_coverage = float(rng.beta(3, 1.5))  # how well listing matches user vocabulary

            d = np.zeros(len(FRICTIONS))
            d[FI["search"]] = 1 - keyword_coverage
            d[FI["category_nav"]] = 0.9 if miscategorized else 0.08
            d[FI["media"]] = np.clip(1 - 0.7 * min(image_count / 6, 1) - 0.3 * has_video, 0, 1)
            d[FI["description"]] = 1 - desc_quality
            d[FI["reviews"]] = np.clip((4.4 - rating) / 1.6, 0, 1) * 0.75 + (0.25 if review_count < 15 else 0)
            d[FI["warranty_returns"]] = (0.6 if return_days == 0 else 0.25 if return_days <= 7 else 0.05) + \
                (0.4 if (cat in ("electronics", "furniture") and warranty_months == 0) else 0)
            d[FI["pricing"]] = np.clip((price_vs_market - 0.92) / 0.3, 0, 1)
            d[FI["offers"]] = 0.15 if has_offer else 0.65
            d[FI["delivery"]] = np.clip((eta_days - 1) / 8, 0, 1)   # user-dependent part added at runtime
            d[FI["payment"]] = 0.1                                    # user-dependent part added at runtime
            products.append({
                "product_id": pid, "category": cat,
                "title_len": int(rng.integers(20, 140)),
                "price": round(price, 2), "price_vs_market": round(price_vs_market, 3),
                "image_count": image_count, "has_video": has_video,
                "desc_quality": round(desc_quality, 3), "rating": round(rating, 1),
                "review_count": review_count, "warranty_months": warranty_months,
                "return_window_days": return_days, "has_offer": has_offer, "cashback_pct": cashback_pct,
                "eta_days": round(eta_days, 1), "serviceable_tiers": ",".join(serviceable),
                "cod_allowed": cod_allowed, "emi_allowed": emi_allowed,
                "miscategorized": miscategorized, "keyword_coverage": round(keyword_coverage, 3),
                "_deficit": np.clip(d, 0, 1),
            })
    return products
