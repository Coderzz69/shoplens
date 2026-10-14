"""Templated text: support tickets from friction-driven abandoners and product
reviews that reflect product deficits. Swap in an LLM later for more variety."""

import numpy as np

from .config import FRICTIONS
from .entities import FI

TICKETS = {
    "search": ["I searched for {q} but nothing came up, does your site even sell this?",
               "Search is useless, typed {q} three times and got no results.",
               "Couldn't find the {cat} item I wanted, search keeps showing irrelevant stuff."],
    "category_nav": ["Why is this product listed under the wrong section? Took forever to find.",
                     "Your {cat} menu is confusing, I kept going back and forth."],
    "media": ["Photos are blurry, can't tell what the product actually looks like.",
              "Only one image for a {price_band} product? Need more pictures or a video.",
              "Is the colour in the picture accurate? Hard to judge from these images."],
    "description": ["Description doesn't mention the size/dimensions, what exactly am I buying?",
                    "Specs are missing. Is this compatible with my device?",
                    "The title says one thing and the details say another. Please clarify."],
    "reviews": ["Rating is only {rating}, lots of complaints in reviews. Is this a reliable product?",
                "Very few reviews on this, not sure I can trust it.",
                "Saw multiple 1-star reviews about quality, so I didn't buy."],
    "warranty_returns": ["Is there any return option? Policy says {ret} days.",
                         "No warranty mentioned for this {cat} item, that's a deal breaker.",
                         "What happens if it arrives damaged? Return policy isn't clear."],
    "pricing": ["Same thing is cheaper elsewhere. Can you match the price?",
                "Price went up once I reached the cart with all these fees.",
                "Why is the delivery fee so high on a small order?"],
    "offers": ["Coupon {code} is not working, says not applicable.",
               "Bank offer not getting applied at checkout.",
               "Expected a festive discount, there's no offer on this."],
    "delivery": ["Your site says my pincode is not serviceable. When will you deliver to {tier} cities?",
                 "Delivery in {eta} days is too long, I needed it sooner.",
                 "Can't get the delivery slot I need."],
    "payment": ["Payment failed twice but money got debited? Error {err}.",
                "Why is COD not available for this product?",
                "UPI payment stuck on pending, I gave up.",
                "EMI option not showing for my card."],
}

REVIEW_GOOD = ["Great product, exactly as described.", "Value for money, would buy again.",
               "Quality is good and delivery was quick.", "Works as expected, happy with it."]
REVIEW_BAD = {
    "media": ["Looks completely different from the photos.", "Colour doesn't match the images."],
    "description": ["Size was not as described.", "Missing features that the listing implied."],
    "reviews": ["Poor quality, stopped working within weeks.", "Not worth it, cheap material."],
    "warranty_returns": ["Tried to return it, no return option. Disappointed.", "No warranty support."],
    "pricing": ["Overpriced compared to other sites.", "Not worth this price."],
    "delivery": ["Delivery took way too long.", "Arrived late and the box was damaged."],
}


def _price_band(price):
    return "cheap" if price < 500 else "mid-range" if price < 5000 else "expensive"


def make_ticket(rng, sid, user, product, cause, ts, events):
    tmpl = TICKETS[cause][rng.integers(len(TICKETS[cause]))]
    err = next((e for e in reversed(events) if e["event_type"] == "payment_result" and '"failed"' in e["props"]), None)
    text = tmpl.format(
        q=rng.choice(["sofa", "earbuds", "kurti", "atta", "sunscreen"]), cat=product["category"] if product else "",
        price_band=_price_band(product["price"]) if product else "", rating=product["rating"] if product else "",
        ret=product["return_window_days"] if product else 0, code=rng.choice(["FIRST50", "FEST20", "BANK5"]),
        tier=user["region_tier"].replace("tier", "tier-"), eta=product["eta_days"] if product else "",
        err=(err["props"].split('"error_code": "')[1].split('"')[0] if err else "UNKNOWN"),
    )
    return {"ticket_id": f"tk_{sid}", "session_id": sid, "user_id": user["user_id"],
            "product_id": product["product_id"] if product else None,
            "channel": rng.choice(["chat", "email", "call_transcript"], p=[0.6, 0.25, 0.15]),
            "ts": ts, "text": text, "_true_theme": cause}


def make_reviews(rng, product, k=4):
    rows, worst = [], [f for f in REVIEW_BAD if product["_deficit"][FI[f]] > 0.5]
    for i in range(min(k, max(1, product["review_count"]))):
        stars = int(np.clip(round(rng.normal(product["rating"], 0.9)), 1, 5))
        if stars <= 3 and worst:
            theme = worst[rng.integers(len(worst))]
            text = REVIEW_BAD[theme][rng.integers(len(REVIEW_BAD[theme]))]
        else:
            theme, text = "positive", REVIEW_GOOD[rng.integers(len(REVIEW_GOOD))]
        rows.append({"review_id": f"r_{product['product_id']}_{i}", "product_id": product["product_id"],
                     "stars": stars, "text": text, "_true_theme": theme})
    return rows


assert set(TICKETS) == set(FRICTIONS)
