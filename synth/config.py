"""All tunable knobs for the synthetic e-commerce journey generator.

The category x friction weight matrix W encodes domain research: how much each
friction point matters for each product category. The simulator uses it as
hidden ground truth; a good behaviour model should be able to recover it from
the clickstream alone.
"""

FRICTIONS = [
    "search",            # search query friction (zero results, reformulation)
    "category_nav",      # wrong / confusing product category
    "media",             # picture / video quality
    "description",       # title / description clarity
    "reviews",           # product reviews / ratings
    "warranty_returns",  # warranty info / return policy
    "pricing",           # price vs market
    "offers",            # offers / discounts / cashbacks
    "delivery",          # delivery ETA / serviceability
    "payment",           # checkout / preferred payment method
]

CATEGORIES = ["grocery", "electronics", "fashion", "furniture", "beauty"]

#                 search nav  media desc  rev   warr  price offer deliv pay
W = {
    "grocery":     [0.7, 0.5, 0.2, 0.3, 0.1, 0.1, 0.8, 0.6, 0.9, 0.6],
    "electronics": [0.6, 0.4, 0.5, 0.8, 0.9, 0.9, 0.8, 0.7, 0.6, 0.8],
    "fashion":     [0.5, 0.6, 0.9, 0.5, 0.6, 0.8, 0.6, 0.7, 0.5, 0.5],
    "furniture":   [0.4, 0.5, 0.9, 0.6, 0.7, 0.7, 0.7, 0.5, 0.8, 0.7],
    "beauty":      [0.5, 0.4, 0.6, 0.6, 0.8, 0.3, 0.5, 0.6, 0.4, 0.4],
}

# Which frictions can surface at which funnel stage.
STAGE_FRICTIONS = {
    "landing": [],
    "search": ["search"],
    "browse": ["category_nav"],
    "pdp": ["media", "description", "reviews", "warranty_returns", "pricing", "offers", "delivery"],
    "cart": ["pricing", "offers"],
    "checkout": ["delivery"],
    "payment": ["payment"],
}

# Natural (friction-free) exit probability per stage, before intent scaling.
BASE_EXIT = {
    "landing": 0.22, "search": 0.10, "browse": 0.12, "pdp": 0.30,
    "cart": 0.55, "checkout": 0.25, "payment": 0.08,
}
INTENT_MIX = {"high": 0.25, "medium": 0.35, "browse": 0.40}
INTENT_EXIT_MULT = {"high": 0.35, "medium": 1.0, "browse": 2.0}

# Friction firing: p = sigmoid(FIRE_A * s + FIRE_B), s = W * sensitivity * deficit
FIRE_A, FIRE_B = 8.0, -2.6
# Friction-induced exit hazard: 1 - exp(-HAZARD_K * step_score / patience)
HAZARD_K = 4.5
# Probability a non-fired friction still produces a "benign" check event
# (e.g. a happy buyer reading reviews). Keeps signatures from being trivial.
BENIGN_RATE = 0.55
# Fraction of friction labels randomly flipped to simulate imperfect ground truth.
LABEL_NOISE = 0.05
# Fraction of friction-driven abandoners who raise a support ticket / chat.
TICKET_RATE = 0.08

# Persona -> (share, sensitivity vector in FRICTIONS order, patience mean, payment prefs)
PERSONAS = {
    "price_hunter":  (0.22, [0.5, 0.4, 0.3, 0.4, 0.5, 0.4, 0.95, 0.95, 0.5, 0.4], 0.9,
                      {"upi": 0.6, "card": 0.2, "cod": 0.15, "wallet": 0.05}),
    "review_reader": (0.20, [0.5, 0.4, 0.7, 0.8, 0.95, 0.6, 0.5, 0.3, 0.4, 0.3], 1.1,
                      {"upi": 0.4, "card": 0.4, "netbanking": 0.1, "emi": 0.1}),
    "impulse":       (0.15, [0.6, 0.5, 0.8, 0.3, 0.3, 0.2, 0.4, 0.5, 0.6, 0.6], 0.6,
                      {"upi": 0.7, "card": 0.2, "wallet": 0.1}),
    "loyal_repeat":  (0.15, [0.3, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.3], 1.6,
                      {"upi": 0.5, "card": 0.4, "wallet": 0.1}),
    "first_timer":   (0.13, [0.7, 0.7, 0.5, 0.6, 0.6, 0.85, 0.5, 0.4, 0.6, 0.75], 0.8,
                      {"cod": 0.5, "upi": 0.4, "card": 0.1}),
    "tier3_cod":     (0.15, [0.6, 0.5, 0.4, 0.4, 0.4, 0.5, 0.7, 0.6, 0.9, 0.9], 1.0,
                      {"cod": 0.75, "upi": 0.25}),
}

REGION_TIERS = {"tier1": 0.45, "tier2": 0.33, "tier3": 0.22}
PINCODE_PREFIX = {"tier1": ["11", "40", "56", "60", "70"],
                  "tier2": ["30", "38", "45", "50", "52"],
                  "tier3": ["78", "79", "81", "84", "85"]}

GATEWAY_FAIL = {"upi": 0.12, "card": 0.10, "netbanking": 0.12, "wallet": 0.05, "emi": 0.12, "cod": 0.0}
PAYMENT_ERRORS = {
    "upi": ["UPI_PENDING", "UPI_TIMEOUT", "BANK_DECLINE"],
    "card": ["BANK_DECLINE", "OTP_TIMEOUT", "3DS_FAILED"],
    "netbanking": ["BANK_DOWN", "SESSION_EXPIRED"],
    "wallet": ["INSUFFICIENT_BALANCE"],
    "emi": ["EMI_NOT_ELIGIBLE", "BANK_DECLINE"],
}

# Per-category catalog shape: (price range INR, typical eta days, cod allowed prob, emi allowed prob,
# serviceable tiers prob for tier3)
CATALOG = {
    "grocery":     ((20, 1500),     1.5, 0.95, 0.0, 0.35),
    "electronics": ((500, 120000),  3.5, 0.6,  0.8, 0.85),
    "fashion":     ((200, 8000),    4.0, 0.9,  0.1, 0.9),
    "furniture":   ((2000, 90000),  8.0, 0.3,  0.7, 0.4),
    "beauty":      ((100, 5000),    3.5, 0.9,  0.0, 0.9),
}

SEARCH_TERMS = {
    "grocery":     [("atta", "aata"), ("milk", "doodh"), ("basmati rice", "basmti"), ("cooking oil", "refined"),
                    ("eggs", "anda"), ("dal", "toor dal 1kg")],
    "electronics": [("bluetooth earbuds", "tws"), ("smartphone", "5g phone under 15k"), ("laptop", "lapy"),
                    ("smart watch", "smartwach"), ("power bank", "20000 mah")],
    "fashion":     [("kurta", "kurti for women"), ("sneakers", "snekers"), ("jeans", "denim slim"),
                    ("saree", "silk sari"), ("t-shirt", "oversized tee")],
    "furniture":   [("sofa", "couch"), ("study table", "wfh desk"), ("bed", "queen cot"),
                    ("wardrobe", "almirah"), ("office chair", "ergonomic chiar")],
    "beauty":      [("sunscreen", "spf 50"), ("face wash", "facewash"), ("lipstick", "matte lip"),
                    ("shampoo", "anti dandruf"), ("serum", "vit c serum")],
}

CATEGORY_PATHS = {
    "grocery": ["Grocery > Staples", "Grocery > Dairy", "Grocery > Snacks"],
    "electronics": ["Electronics > Audio", "Electronics > Mobiles", "Electronics > Laptops"],
    "fashion": ["Fashion > Women", "Fashion > Men", "Fashion > Footwear"],
    "furniture": ["Home > Furniture > Living", "Home > Furniture > Bedroom", "Home > Furniture > Office"],
    "beauty": ["Beauty > Skincare", "Beauty > Makeup", "Beauty > Haircare"],
}
