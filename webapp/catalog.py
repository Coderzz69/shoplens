"""Human-facing catalog content derived from the synthetic product attributes:
names, search keywords, descriptions, and placeholder product images (SVG)."""

import hashlib

ITEMS = {
    "grocery": [("Whole Wheat Atta", "atta", ["aata", "flour", "gehu"], "🌾"),
                ("Toned Milk", "milk", ["doodh"], "🥛"),
                ("Basmati Rice", "basmati rice", ["basmti", "chawal", "rice"], "🍚"),
                ("Refined Sunflower Oil", "cooking oil", ["refined", "tel", "oil"], "🫗"),
                ("Farm Eggs (12 pcs)", "eggs", ["anda", "egg"], "🥚"),
                ("Toor Dal", "dal", ["toor dal", "arhar", "lentils"], "🫘")],
    "electronics": [("Bluetooth Earbuds", "bluetooth earbuds", ["tws", "earphones", "buds"], "🎧"),
                    ("5G Smartphone", "smartphone", ["5g phone", "mobile", "phone"], "📱"),
                    ("Thin & Light Laptop", "laptop", ["lapy", "notebook"], "💻"),
                    ("Smart Watch", "smart watch", ["smartwach", "fitness band", "watch"], "⌚"),
                    ("Power Bank 20000mAh", "power bank", ["20000 mah", "charger"], "🔋")],
    "fashion": [("Cotton Kurta", "kurta", ["kurti", "ethnic top"], "👘"),
                ("Running Sneakers", "sneakers", ["snekers", "shoes", "trainers"], "👟"),
                ("Slim Fit Jeans", "jeans", ["denim", "pants"], "👖"),
                ("Silk Saree", "saree", ["sari"], "🥻"),
                ("Oversized T-Shirt", "t-shirt", ["tee", "tshirt"], "👕")],
    "furniture": [("3-Seater Sofa", "sofa", ["couch", "settee"], "🛋️"),
                  ("Study Table", "study table", ["wfh desk", "desk"], "🪑"),
                  ("Queen Size Bed", "bed", ["queen cot", "cot", "palang"], "🛏️"),
                  ("2-Door Wardrobe", "wardrobe", ["almirah", "closet", "cupboard"], "🚪"),
                  ("Ergonomic Office Chair", "office chair", ["ergonomic chiar", "chair"], "🪑")],
    "beauty": [("Sunscreen SPF 50", "sunscreen", ["spf 50", "sunblock"], "🧴"),
               ("Gentle Face Wash", "face wash", ["facewash", "cleanser"], "🫧"),
               ("Matte Lipstick", "lipstick", ["matte lip", "lip colour"], "💄"),
               ("Anti-Dandruff Shampoo", "shampoo", ["anti dandruf", "hair wash"], "🧴"),
               ("Vitamin C Serum", "serum", ["vit c serum", "face serum"], "✨")],
}

BRANDS = ["Nimbus", "Aurora", "Kestrel", "Vanta", "Lumen", "Orchid", "Sahyadri", "Teakwood", "Indigo",
          "Monsoon", "Saffron", "Zephyr", "Cobalt", "Banyan", "Mistral", "Pebble"]
COLORS = {"grocery": ("#d9f99d", "#65a30d"), "electronics": ("#bfdbfe", "#1d4ed8"),
          "fashion": ("#fbcfe8", "#be185d"), "furniture": ("#fde68a", "#b45309"),
          "beauty": ("#ddd6fe", "#7c3aed")}
CATEGORY_LABEL = {"grocery": "Grocery", "electronics": "Electronics", "fashion": "Fashion",
                  "furniture": "Furniture", "beauty": "Beauty"}


def _h(s, mod):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod


def item_for(product_id, category):
    return ITEMS[category][_h(product_id + "i", len(ITEMS[category]))]


def name_for(product_id, category):
    title, _, _, _ = item_for(product_id, category)
    brand = BRANDS[_h(product_id + "b", len(BRANDS))]
    suffix = ["", " Pro", " Lite", " Plus", " Classic", " Max", " Neo"][_h(product_id + "s", 7)]
    return f"{brand} {title}{suffix}"


def keywords_for(product_id, category, keyword_coverage):
    """Primary term always matches; synonyms only for listings with good keyword coverage
    (that's exactly the search-friction knob from the simulator)."""
    title, primary, synonyms, _ = item_for(product_id, category)
    kws = [primary, title.lower()]
    if keyword_coverage >= 0.7:
        kws += synonyms
    return " | ".join(kws)


def description_for(p):
    title, primary, _, _ = item_for(p["product_id"], p["category"])
    if p["desc_quality"] < 0.45:
        return f"{title}. Good quality product."
    lines = [f"{title} from {name_for(p['product_id'], p['category']).split()[0]}."]
    if p["desc_quality"] >= 0.6:
        lines.append({"grocery": "Net weight and ingredients listed on pack. Store in a cool, dry place.",
                      "electronics": "Specs: Bluetooth 5.3, USB-C charging, 1-year limited support.",
                      "fashion": "Fabric: 100% cotton. Regular fit. Size chart: S/M/L/XL/XXL.",
                      "furniture": "Dimensions: 180 x 80 x 75 cm. Material: engineered wood. Assembly included.",
                      "beauty": "Dermatologically tested. Suitable for all skin types. 100 ml."}[p["category"]])
    if p["desc_quality"] >= 0.8:
        lines.append("What's in the box, care instructions and compatibility details are listed below.")
    return " ".join(lines)


def svg_image(product_id, category, idx):
    _, _, _, emoji = item_for(product_id, category)
    light, dark = COLORS[category]
    angle = (idx * 47) % 360
    scale = [1.0, 0.8, 1.2, 0.9, 1.1, 0.7, 1.0, 0.85][idx % 8]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">
<defs><linearGradient id="g" gradientTransform="rotate({angle} .5 .5)">
<stop offset="0" stop-color="{light}"/><stop offset="1" stop-color="{dark}" stop-opacity=".35"/></linearGradient></defs>
<rect width="400" height="400" fill="url(#g)"/>
<text x="200" y="{230 + idx * 3}" font-size="{int(150 * scale)}" text-anchor="middle">{emoji}</text>
<text x="388" y="388" font-size="16" text-anchor="end" fill="{dark}" font-family="sans-serif">view {idx + 1}</text>
</svg>"""
