"""ShopLens — e-commerce clone wrapped around the friction-detection models.

    .venv/bin/uvicorn webapp.app:app --reload --port 8000

Customers shop (every interaction is tracked in the same schema as the synthetic data,
and the live model can step in with a recovery nudge). Suppliers see the problems the
model diagnosed on their products, the evidence, and a proposed fix they can apply.
"""

import json
import math
import os
import random
import secrets
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from . import db
from .catalog import CATEGORY_LABEL, COLORS, item_for, svg_image
from .ml import (FRICTIONS, LABEL, NUDGE, QUICK_FIX, SOLUTION, apply_quick_fix, models, product_diagnostics,
                 tier_for_pincode)

HERE = os.path.dirname(os.path.abspath(__file__))
CATEGORIES = list(CATEGORY_LABEL)
GATEWAY_FAIL = {"upi": 0.12, "card": 0.10, "netbanking": 0.12, "wallet": 0.05, "emi": 0.12, "cod": 0.0}
PAYMENT_ERRORS = {"upi": ["UPI_PENDING", "UPI_TIMEOUT", "BANK_DECLINE"], "card": ["BANK_DECLINE", "OTP_TIMEOUT"],
                  "netbanking": ["BANK_DOWN", "SESSION_EXPIRED"], "wallet": ["INSUFFICIENT_BALANCE"],
                  "emi": ["EMI_NOT_ELIGIBLE", "BANK_DECLINE"]}
METHOD_LABEL = {"upi": "UPI", "card": "Credit / Debit card", "netbanking": "Net banking", "wallet": "Wallet",
                "cod": "Cash on Delivery", "emi": "EMI"}
COUPONS = {"SAVE5": (0.05, 0, None), "FIRST50": (0.10, 999, "first_order"), "FEST20": (0.20, 0, "expired")}
RECOVERY_RATE = 0.25  # assumption shown in the UI for "recoverable revenue"

app = FastAPI(title="ShopLens")
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SHOP_SECRET", "dev-" + secrets.token_hex(8)),
                   same_site="lax", max_age=7 * 24 * 3600)
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(HERE, "templates"))


def inr(v):
    return "₹{:,.0f}".format(v or 0)


templates.env.filters["inr"] = inr
templates.env.filters["pct"] = lambda v: f"{(v or 0):.0%}"
templates.env.globals.update(CATEGORY_LABEL=CATEGORY_LABEL, LABEL=LABEL, COLORS=COLORS, METHOD_LABEL=METHOD_LABEL)


@app.on_event("startup")
def startup():
    db.init()
    if not db.q("SELECT 1 FROM products LIMIT 1"):
        raise RuntimeError("Database is empty. Run: .venv/bin/python -m webapp.seed")
    models()  # load once


# ---------------- helpers ----------------
def now():
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="milliseconds")


@app.middleware("http")
async def ensure_ids(request: Request, call_next):
    """Session id (30-min sliding window) + anonymous id cookies shared with tracker.js."""
    sid = request.cookies.get("sid") or "s_live_" + uuid.uuid4().hex[:12]
    aid = request.cookies.get("aid") or "a_live_" + uuid.uuid4().hex[:10]
    request.state.sid, request.state.aid = sid, aid
    resp = await call_next(request)
    if not request.url.path.startswith(("/static", "/img")):
        resp.set_cookie("sid", sid, max_age=1800, samesite="lax")
        if "aid" not in request.cookies:
            resp.set_cookie("aid", aid, max_age=365 * 24 * 3600, samesite="lax")
    return resp


def me(request):
    aid = request.session.get("account_id")
    return db.q("SELECT * FROM accounts WHERE id=?", (aid,), one=True) if aid else None


def require(request, role):
    user = me(request)
    if not user:
        raise HTTPException(303, headers={"Location": f"/login?next={request.url.path}"})
    if user["role"] != role:
        raise HTTPException(403, "This page is for " + role + "s")
    return user


def render(request, name, **ctx):
    user = me(request)
    cart_count = 0
    if user and user["role"] == "customer":
        cart_count = db.q("SELECT COALESCE(SUM(qty),0) n FROM cart_items WHERE account_id=?", (user["id"],), one=True)["n"]
    return templates.TemplateResponse(request, name, {"user": user, "cart_count": cart_count, **ctx})


def device(request):
    return "mobile" if "Mobi" in request.headers.get("user-agent", "") else "desktop"


def insert_events(rows):
    with db.tx() as con:
        con.executemany("INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
            (r["event_id"], r["session_id"], r.get("user_id"), r.get("anonymous_id"), r["ts"], r["event_type"],
             r.get("page_type"), r.get("product_id"), r.get("category"), r.get("device"),
             json.dumps(r.get("props") or {}), r.get("source", "client")) for r in rows])


def emit(request, event_type, page_type, product=None, **props):
    """Server-side event (search results, coupons, pincode, payment) — the source of truth."""
    user = me(request)
    insert_events([{
        "event_id": uuid.uuid4().hex, "session_id": request.state.sid,
        "user_id": f"acc_{user['id']}" if user else None, "anonymous_id": request.state.aid, "ts": now(),
        "event_type": event_type, "page_type": page_type,
        "product_id": product["product_id"] if product else None,
        "category": product["category"] if product else None, "device": device(request),
        "props": props, "source": "server"}])


def product(pid):
    p = db.q("SELECT * FROM products WHERE product_id=?", (pid,), one=True)
    if not p:
        raise HTTPException(404, "Product not found")
    p["shown_category"] = shown_category(p)
    p["emoji"] = item_for(p["product_id"], p["category"])[3]
    return p


def shown_category(p):
    """Mis-categorized listings appear under the wrong department (category friction)."""
    if p["miscategorized"]:
        return CATEGORIES[(CATEGORIES.index(p["category"]) + 1) % len(CATEGORIES)]
    return p["category"]


def decorate(rows):
    for p in rows:
        p["emoji"] = item_for(p["product_id"], p["category"])[3]
        p["shown_category"] = shown_category(p)
    return rows


def current_pincode(request):
    user = me(request)
    return request.session.get("pincode") or (user["pincode"] if user else None)


def serviceable(p, pincode):
    tier = tier_for_pincode(pincode)
    return f"tier{tier}" in p["serviceable_tiers"].split(","), tier


def eta_for(p, tier):
    return round(p["eta_days"] * (1.5 if tier == 3 else 1.0), 1)


def cart_lines(account_id):
    rows = db.q("""SELECT c.qty, p.* FROM cart_items c JOIN products p USING(product_id)
                   WHERE c.account_id=?""", (account_id,))
    return decorate(rows)


def cart_totals(lines, coupon=None):
    subtotal = sum(line["price"] * line["qty"] for line in lines)
    delivery_fee = 0 if subtotal >= 499 or not lines else 40
    platform_fee = 5 if lines else 0
    discount = 0.0
    if coupon and coupon in COUPONS:
        discount = round(subtotal * COUPONS[coupon][0], 2)
    return {"subtotal": subtotal, "delivery_fee": delivery_fee, "platform_fee": platform_fee,
            "discount": discount, "total": max(0, subtotal + delivery_fee + platform_fee - discount)}


# ---------------- auth ----------------
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/"):
    return render(request, "login.html", next=next, error=None)


@app.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...), next: str = Form("/")):
    acc = db.q("SELECT * FROM accounts WHERE email=?", (email.strip().lower(),), one=True)
    if not acc or not db.check_password(password, acc["password_hash"]):
        return render(request, "login.html", next=next, error="Wrong email or password")
    request.session.clear()
    request.session["account_id"] = acc["id"]
    if acc["role"] == "supplier":
        return RedirectResponse("/supplier", 303)
    return RedirectResponse(next if next.startswith("/") and not next.startswith("/supplier") else "/", 303)


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request, role: str = "customer"):
    return render(request, "register.html", role=role, error=None)


@app.post("/register")
def register(request: Request, role: str = Form(...), name: str = Form(...), email: str = Form(...),
             password: str = Form(...), store_name: str = Form(""), pincode: str = Form("")):
    email = email.strip().lower()
    if role not in ("customer", "supplier"):
        role = "customer"
    err = None
    if len(password) < 8:
        err = "Password must be at least 8 characters"
    elif "@" not in email:
        err = "Enter a valid email"
    elif db.q("SELECT 1 FROM accounts WHERE email=?", (email,), one=True):
        err = "An account with this email already exists"
    elif role == "supplier" and not store_name.strip():
        err = "Store name is required for suppliers"
    if err:
        return render(request, "register.html", role=role, error=err)
    acc_id = db.ex("INSERT INTO accounts (email, password_hash, role, name, store_name, pincode) VALUES (?,?,?,?,?,?)",
                   (email, db.hash_password(password), role, name.strip(), store_name.strip() or None,
                    pincode.strip() or None))
    request.session.clear()
    request.session["account_id"] = acc_id
    return RedirectResponse("/supplier" if role == "supplier" else "/", 303)


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", 303)


# ---------------- storefront ----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    user = me(request)
    if user and user["role"] == "supplier":
        return RedirectResponse("/supplier", 303)
    rails = {}
    for c in CATEGORIES:
        rails[c] = decorate(db.q("""SELECT * FROM products WHERE category=? AND miscategorized=0
                                    ORDER BY rating * (1 + review_count / 200.0) DESC LIMIT 8""", (c,)))
    deals = decorate(db.q("SELECT * FROM products WHERE has_offer=1 ORDER BY cashback_pct DESC, rating DESC LIMIT 8"))
    return render(request, "home.html", rails=rails, deals=deals, page="home")


@app.get("/search", response_class=HTMLResponse)
def search(request: Request, q: str = ""):
    q = q.strip()
    results = []
    if q:
        terms = [t for t in q.lower().split() if len(t) > 1]
        where = " AND ".join(["(lower(keywords) LIKE ? OR lower(name) LIKE ?)"] * len(terms)) or "1=0"
        args = [a for t in terms for a in (f"%{t}%", f"%{t}%")]
        results = decorate(db.q(f"""SELECT * FROM products WHERE {where}
                                    ORDER BY rating * (1 + review_count / 200.0) DESC LIMIT 40""", args))
        prior = db.q("""SELECT 1 FROM events WHERE session_id=? AND event_type IN ('search_submitted','search_refined')
                        LIMIT 1""", (request.state.sid,), one=True)
        emit(request, "search_refined" if prior else "search_submitted", "search", query=q,
             results_count=len(results))
    suggestions = ["sofa", "earbuds", "kurta", "atta", "sunscreen", "laptop", "sneakers"] if q and not results else []
    return render(request, "search.html", q=q, results=results, suggestions=suggestions, page="search")


@app.get("/c/{category}", response_class=HTMLResponse)
def category_page(request: Request, category: str, sort: str = "popular", deliverable: int = 0):
    if category not in CATEGORIES:
        raise HTTPException(404)
    order = {"popular": "rating * (1 + review_count / 200.0) DESC", "price_low": "price ASC",
             "price_high": "price DESC", "rating": "rating DESC"}.get(sort, "rating DESC")
    rows = db.q(f"SELECT * FROM products ORDER BY {order}")
    rows = [p for p in decorate(rows) if p["shown_category"] == category]
    pincode = current_pincode(request)
    if deliverable and pincode:
        rows = [p for p in rows if serviceable(p, pincode)[0]]
    return render(request, "category.html", category=category, products=rows[:60], sort=sort,
                  deliverable=deliverable, pincode=pincode, page="category")


@app.get("/p/{pid}", response_class=HTMLResponse)
def product_page(request: Request, pid: str):
    p = product(pid)
    reviews = db.q("SELECT * FROM reviews WHERE product_id=? ORDER BY stars DESC", (pid,))
    similar = decorate(db.q("""SELECT * FROM products WHERE category=? AND product_id<>?
                               ORDER BY abs(price - ?) LIMIT 4""", (p["category"], pid, p["price"])))
    for s in similar:
        s["delta_pct"] = round((s["price"] / p["price"] - 1) * 100)
    pincode = current_pincode(request)
    delivery = None
    if pincode:
        ok, tier = serviceable(p, pincode)
        delivery = {"pincode": pincode, "ok": ok, "eta": eta_for(p, tier)}
    market = p["price"] / p["price_vs_market"]
    history = [round(market * f) for f in (1.08, 1.04, 1.02, 0.99, 1.01, p["price_vs_market"])]
    return render(request, "product.html", p=p, reviews=reviews, similar=similar, delivery=delivery,
                  history=history, page="pdp")


@app.get("/img/{pid}/{idx}.svg")
def image(pid: str, idx: int):
    p = db.q("SELECT category FROM products WHERE product_id=?", (pid,), one=True)
    if not p:
        raise HTTPException(404)
    return Response(svg_image(pid, p["category"], idx), media_type="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.post("/api/pincode")
async def check_pincode(request: Request):
    body = await request.json()
    pincode = str(body.get("pincode", "")).strip()[:6]
    p = product(body["product_id"])
    if not (pincode.isdigit() and len(pincode) == 6):
        return JSONResponse({"error": "Enter a 6-digit pincode"}, 400)
    prior = db.q("SELECT 1 FROM events WHERE session_id=? AND event_type='pincode_check' AND product_id=? LIMIT 1",
                 (request.state.sid, p["product_id"]), one=True)
    ok, tier = serviceable(p, pincode)
    eta = eta_for(p, tier) if ok else None
    emit(request, "pincode_changed" if prior else "pincode_check", "pdp", p,
         pincode=pincode[:3] + "***", serviceable=ok, eta_days=eta)
    request.session["pincode"] = pincode
    return {"serviceable": ok, "eta_days": eta, "cod": bool(p["cod_allowed"])}


# ---------------- cart / checkout / payment ----------------
@app.post("/cart/add")
def cart_add(request: Request, product_id: str = Form(...), buy_now: int = Form(0)):
    user = me(request)
    p = product(product_id)
    if not user or user["role"] != "customer":
        return RedirectResponse(f"/login?next=/p/{product_id}", 303)
    db.ex("""INSERT INTO cart_items VALUES (?,?,1)
             ON CONFLICT(account_id, product_id) DO UPDATE SET qty = qty + 1""", (user["id"], product_id))
    emit(request, "add_to_cart", "pdp", p, price=p["price"])
    return RedirectResponse("/checkout" if buy_now else "/cart", 303)


@app.post("/cart/remove")
def cart_remove(request: Request, product_id: str = Form(...)):
    user = require(request, "customer")
    p = product(product_id)
    db.ex("DELETE FROM cart_items WHERE account_id=? AND product_id=?", (user["id"], product_id))
    emit(request, "remove_from_cart", "cart", p)
    return RedirectResponse("/cart", 303)


@app.get("/cart", response_class=HTMLResponse)
def cart(request: Request):
    user = require(request, "customer")
    lines = cart_lines(user["id"])
    coupon = request.session.get("coupon")
    return render(request, "cart.html", lines=lines, totals=cart_totals(lines, coupon), coupon=coupon,
                  coupon_msg=request.session.pop("coupon_msg", None), page="cart",
                  last=lines[-1] if lines else None)


@app.post("/cart/coupon")
def apply_coupon(request: Request, code: str = Form(...)):
    user = require(request, "customer")
    code = code.strip().upper()
    lines = cart_lines(user["id"])
    subtotal = cart_totals(lines)["subtotal"]
    ok, reason = False, "not_applicable"
    if code in COUPONS:
        _, min_order, rule = COUPONS[code]
        has_orders = db.q("SELECT 1 FROM orders WHERE account_id=? LIMIT 1", (user["id"],), one=True)
        if rule == "expired":
            reason = "expired"
        elif subtotal < min_order:
            reason = "min_order_not_met"
        elif rule == "first_order" and has_orders:
            reason = "not_applicable"
        else:
            ok, reason = True, None
    emit(request, "coupon_applied", "cart", lines[-1] if lines else None, code=code, success=ok, reason=reason)
    if ok:
        request.session["coupon"] = code
        request.session["coupon_msg"] = f"✓ {code} applied"
    else:
        request.session["coupon_msg"] = {"expired": f"{code} has expired",
                                         "min_order_not_met": f"{code} needs a minimum order of ₹999",
                                         "not_applicable": f"{code} can't be applied to this order"}[reason]
    return RedirectResponse("/cart", 303)


@app.get("/checkout", response_class=HTMLResponse)
def checkout(request: Request):
    user = require(request, "customer")
    lines = cart_lines(user["id"])
    if not lines:
        return RedirectResponse("/cart", 303)
    emit(request, "checkout_step", "checkout", lines[-1], step="address")
    return render(request, "checkout.html", lines=lines, totals=cart_totals(lines, request.session.get("coupon")),
                  pincode=current_pincode(request) or "", error=None, blocked=[], page="checkout", last=lines[-1])


@app.post("/checkout", response_class=HTMLResponse)
def checkout_submit(request: Request, address: str = Form(...), pincode: str = Form(...)):
    user = require(request, "customer")
    lines = cart_lines(user["id"])
    pincode = pincode.strip()
    blocked = [line for line in lines if not serviceable(line, pincode)[0]]
    totals = cart_totals(lines, request.session.get("coupon"))
    if not (pincode.isdigit() and len(pincode) == 6):
        return render(request, "checkout.html", lines=lines, totals=totals, pincode=pincode,
                      error="Enter a valid 6-digit pincode", blocked=[], page="checkout", last=lines[-1])
    if blocked:
        emit(request, "address_not_serviceable", "checkout", blocked[0], pincode=pincode[:3] + "***")
        return render(request, "checkout.html", lines=lines, totals=totals, pincode=pincode, blocked=blocked,
                      error="Some items can't be delivered to this pincode", page="checkout", last=lines[-1])
    tier = tier_for_pincode(pincode)
    eta = max(eta_for(line, tier) for line in lines)
    emit(request, "delivery_estimate_shown", "checkout", lines[-1], eta_days=eta)
    emit(request, "checkout_step", "checkout", lines[-1], step="summary")
    request.session.update(pincode=pincode, address=address[:200], eta=eta)
    return RedirectResponse("/pay", 303)


def available_methods(lines):
    m = ["upi", "card", "netbanking", "wallet"]
    if all(line["cod_allowed"] for line in lines):
        m.append("cod")
    if all(line["emi_allowed"] for line in lines):
        m.append("emi")
    return m


@app.get("/pay", response_class=HTMLResponse)
def pay_page(request: Request):
    user = require(request, "customer")
    lines = cart_lines(user["id"])
    if not lines or "pincode" not in request.session:
        return RedirectResponse("/checkout", 303)
    emit(request, "checkout_step", "payment", lines[-1], step="payment")
    return render(request, "pay.html", lines=lines, totals=cart_totals(lines, request.session.get("coupon")),
                  methods=available_methods(lines), error=None, page="payment", last=lines[-1],
                  eta=request.session.get("eta"))


@app.post("/pay", response_class=HTMLResponse)
def pay(request: Request, method: str = Form(...)):
    user = require(request, "customer")
    lines = cart_lines(user["id"])
    if not lines:
        return RedirectResponse("/cart", 303)
    methods = available_methods(lines)
    totals = cart_totals(lines, request.session.get("coupon"))
    last = lines[-1]
    if method not in methods:
        emit(request, "payment_method_unavailable", "payment", last, method=method)
        return render(request, "pay.html", lines=lines, totals=totals, methods=methods, last=last,
                      error=f"{METHOD_LABEL.get(method, method)} isn't available for this order", page="payment",
                      eta=request.session.get("eta"))
    prev = request.session.get("pay_method")
    emit(request, "payment_method_selected", "payment", last, method=method, switched=bool(prev and prev != method))
    request.session["pay_method"] = method
    attempt = request.session.get("pay_attempt", 0) + 1
    request.session["pay_attempt"] = attempt
    emit(request, "payment_attempt", "payment", last, method=method, attempt=attempt)
    tier = tier_for_pincode(request.session.get("pincode"))
    fail_p = GATEWAY_FAIL[method] * (1.8 if tier == 3 else 1.0)
    if os.environ.get("SHOP_FORCE_PAYMENT_FAIL") == "1" and method != "cod":
        fail_p = 1.0
    if random.random() < fail_p:
        err = random.choice(PAYMENT_ERRORS[method])
        emit(request, "payment_result", "payment", last, method=method, status="failed", error_code=err)
        return render(request, "pay.html", lines=lines, totals=totals, methods=methods, last=last,
                      error=f"Payment failed ({err}). No money was deducted — try again or pick another method.",
                      page="payment", eta=request.session.get("eta"))
    emit(request, "payment_result", "payment", last, method=method, status="success", error_code=None)
    order_id = "o_" + uuid.uuid4().hex[:10]
    with db.tx() as con:
        con.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?)",
                    (order_id, user["id"], request.state.sid, totals["total"], totals["discount"], method,
                     request.session.get("pincode"), "placed", now()))
        con.executemany("INSERT INTO order_items VALUES (?,?,?,?)",
                        [(order_id, line["product_id"], line["qty"], line["price"]) for line in lines])
        con.execute("DELETE FROM cart_items WHERE account_id=?", (user["id"],))
        con.execute("UPDATE live_sessions SET converted=1 WHERE session_id=?", (request.state.sid,))
    emit(request, "order_placed", "order_confirmation", last, amount=totals["total"], method=method)
    for k in ("coupon", "pay_method", "pay_attempt"):
        request.session.pop(k, None)
    return RedirectResponse(f"/orders/{order_id}", 303)


@app.get("/orders", response_class=HTMLResponse)
def orders(request: Request):
    user = require(request, "customer")
    rows = db.q("SELECT * FROM orders WHERE account_id=? ORDER BY ts DESC", (user["id"],))
    return render(request, "orders.html", orders=rows, page="orders")


@app.get("/orders/{order_id}", response_class=HTMLResponse)
def order_detail(request: Request, order_id: str):
    user = require(request, "customer")
    o = db.q("SELECT * FROM orders WHERE order_id=? AND account_id=?", (order_id, user["id"]), one=True)
    if not o:
        raise HTTPException(404)
    items = decorate(db.q("""SELECT oi.qty, oi.price AS paid, p.* FROM order_items oi JOIN products p USING(product_id)
                             WHERE order_id=?""", (order_id,)))
    return render(request, "order.html", o=o, items=items, page="order_confirmation")


# ---------------- clickstream collector + live recovery assistant ----------------
@app.post("/collect")
async def collect(request: Request):
    try:
        batch = json.loads(await request.body())
    except json.JSONDecodeError:
        return JSONResponse({"error": "bad json"}, 400)
    if not isinstance(batch, list) or not batch:
        return {}
    user = me(request)
    sid = request.state.sid
    rows = []
    for e in batch[:200]:
        if not isinstance(e, dict) or not isinstance(e.get("event_type"), str):
            continue
        ts = str(e.get("ts", now())).replace("Z", "")[:23]
        rows.append({
            "event_id": str(e.get("event_id") or uuid.uuid4().hex)[:64], "session_id": sid,
            "user_id": f"acc_{user['id']}" if user else None, "anonymous_id": request.state.aid,
            "ts": ts, "event_type": e["event_type"][:64], "page_type": str(e.get("page_type") or "")[:32] or None,
            "product_id": e.get("product_id") or None, "category": e.get("category") or None,
            "device": device(request), "props": e.get("props") if isinstance(e.get("props"), dict) else {},
            "source": "client"})
    insert_events(rows)
    if user and user["role"] == "supplier":
        return {}
    return {"intervention": score_and_intervene(request, user)}


def score_and_intervene(request, user):
    sid = request.state.sid
    events = db.q("SELECT * FROM events WHERE session_id=? ORDER BY ts", (sid,))
    last_pid = next((e["product_id"] for e in reversed(events) if e["product_id"]), None)
    if not last_pid or len(events) < 4:
        return None
    p = db.q("SELECT * FROM products WHERE product_id=?", (last_pid,), one=True)
    pincode = current_pincode(request)
    s = models().score_session(events, p, tier_for_pincode(pincode))
    if not s:
        return None
    with db.tx() as con:
        con.execute("""INSERT INTO live_sessions (session_id, account_id, pincode, last_product_id, risk, cause,
                       cause_conf, evidence, updated_at) VALUES (?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(session_id) DO UPDATE SET account_id=excluded.account_id, pincode=excluded.pincode,
                       last_product_id=excluded.last_product_id, risk=excluded.risk, cause=excluded.cause,
                       cause_conf=excluded.cause_conf, evidence=excluded.evidence, updated_at=excluded.updated_at""",
                    (sid, user["id"] if user else None, pincode, last_pid, s["risk"], s["cause"], s["cause_conf"],
                     json.dumps({"evidence": s["evidence"], "probs": s["probs"], "p_none": s["p_none"]}), now()))
    shown = db.q("SELECT cause FROM interventions WHERE session_id=?", (sid,))
    if len(shown) >= 3 or s["cause"] in {r["cause"] for r in shown}:
        return None
    if s["risk"] < 0.4 or s["cause_conf"] < 0.5:
        return None
    title, body, link = NUDGE[s["cause"]]
    if s["cause"] == "delivery":
        link = f"/c/{p['category']}?deliverable=1"
    if s["cause"] == "reviews":
        best = db.q("SELECT text FROM reviews WHERE product_id=? AND stars>=4 LIMIT 1", (last_pid,), one=True)
        if best:
            body = f"Verified buyer: “{best['text']}” " + body
    db.ex("INSERT INTO interventions (session_id, cause, product_id, message) VALUES (?,?,?,?)",
          (sid, s["cause"], last_pid, title))
    emit(request, "intervention_shown", None, p, cause=s["cause"])
    return {"title": title, "body": body, "link": link, "cause": s["cause"], "risk": round(s["risk"], 2),
            "evidence": [e["text"] for e in s["evidence"][:2]]}


# ---------------- supplier portal ----------------
def supplier_products(sup_id):
    return db.q("SELECT product_id FROM products WHERE supplier_id=?", (sup_id,))


@app.get("/supplier", response_class=HTMLResponse)
def supplier_home(request: Request):
    sup = require(request, "supplier")
    sid = sup["id"]
    kpi = db.q("""SELECT COUNT(*) n_products, COALESCE(SUM(s.sessions),0) sessions, COALESCE(SUM(s.converted),0) converted,
                  COALESCE(SUM(s.abandoned),0) abandoned, COALESCE(SUM(s.friction_abandoned),0) friction
                  FROM products p LEFT JOIN product_stats s USING(product_id) WHERE p.supplier_id=?""", (sid,), one=True)
    by_cause = db.q("""SELECT i.cause, SUM(i.sessions) sessions, SUM(i.sessions * p.price) value
                       FROM insights i JOIN products p USING(product_id) WHERE p.supplier_id=?
                       GROUP BY i.cause ORDER BY sessions DESC""", (sid,))
    problems = db.q("""SELECT i.*, p.name, p.category, p.price, p.product_id,
                       (SELECT ts FROM fixes f WHERE f.product_id=i.product_id AND f.cause=i.cause
                        ORDER BY ts DESC LIMIT 1) AS fixed_at
                       FROM insights i JOIN products p USING(product_id)
                       WHERE p.supplier_id=? ORDER BY i.sessions * p.price DESC LIMIT 25""", (sid,))
    for r in problems:
        r["evidence"] = json.loads(r["evidence"])
        r["recoverable"] = r["sessions"] * r["price"] * RECOVERY_RATE
        r["solution"] = SOLUTION[r["cause"]]
        r["emoji"] = item_for(r["product_id"], r["category"])[3]
    live = live_sessions_for(sid, 8)
    at_risk = sum(c["value"] for c in by_cause) * RECOVERY_RATE
    fixes = db.q("""SELECT f.*, p.name FROM fixes f JOIN products p USING(product_id) WHERE f.supplier_id=?
                    ORDER BY f.ts DESC LIMIT 8""", (sid,))
    return render(request, "supplier_home.html", kpi=kpi, by_cause=by_cause, problems=problems, live=live,
                  at_risk=at_risk, fixes=fixes, recovery_rate=RECOVERY_RATE, page="supplier")


def live_sessions_for(sup_id, limit=50, product_id=None):
    sql = """SELECT l.*, p.name, p.category, a.name AS customer,
             (SELECT COUNT(*) FROM events e WHERE e.session_id=l.session_id) n_events,
             (SELECT group_concat(cause) FROM interventions i WHERE i.session_id=l.session_id) nudges
             FROM live_sessions l JOIN products p ON p.product_id=l.last_product_id
             LEFT JOIN accounts a ON a.id=l.account_id WHERE p.supplier_id=?"""
    args = [sup_id]
    if product_id:
        sql += " AND p.product_id=?"
        args.append(product_id)
    rows = db.q(sql + " ORDER BY l.updated_at DESC LIMIT ?", (*args, limit))
    for r in rows:
        d = json.loads(r["evidence"] or "{}")
        r["evidence"] = d.get("evidence", [])
        r["p_none"] = d.get("p_none", 0)
    return rows


@app.get("/supplier/products", response_class=HTMLResponse)
def supplier_products_page(request: Request, sort: str = "lost"):
    sup = require(request, "supplier")
    rows = db.q("""SELECT p.*, COALESCE(s.sessions,0) sessions, COALESCE(s.converted,0) converted,
                   COALESCE(s.friction_abandoned,0) friction,
                   (SELECT cause FROM insights i WHERE i.product_id=p.product_id ORDER BY sessions DESC LIMIT 1) top_cause
                   FROM products p LEFT JOIN product_stats s USING(product_id) WHERE p.supplier_id=?""", (sup["id"],))
    key = {"lost": lambda r: -r["friction"] * r["price"], "name": lambda r: r["name"],
           "conversion": lambda r: r["converted"] / max(r["sessions"], 1)}.get(sort, lambda r: -r["friction"])
    rows.sort(key=key)
    return render(request, "supplier_products.html", products=decorate(rows), sort=sort, page="supplier")


@app.get("/supplier/product/{pid}", response_class=HTMLResponse)
def supplier_product(request: Request, pid: str):
    sup = require(request, "supplier")
    p = product(pid)
    if p["supplier_id"] != sup["id"]:
        raise HTTPException(403, "Not your product")
    stats = db.q("SELECT * FROM product_stats WHERE product_id=?", (pid,), one=True) or \
        {"sessions": 0, "converted": 0, "abandoned": 0, "friction_abandoned": 0}
    insights = db.q("SELECT * FROM insights WHERE product_id=? ORDER BY sessions DESC", (pid,))
    fixes = db.q("SELECT * FROM fixes WHERE product_id=? ORDER BY ts DESC", (pid,))
    fixed = {f["cause"]: f for f in reversed(fixes)}
    for r in insights:
        r["evidence"] = json.loads(r["evidence"])
        r["facts"] = product_diagnostics(p, r["cause"])
        r["solution"] = SOLUTION[r["cause"]]
        r["quick_fix"] = apply_quick_fix(p, r["cause"])
        r["fixable"] = r["cause"] in QUICK_FIX
        r["fixed"] = fixed.get(r["cause"])
        r["recoverable"] = r["sessions"] * p["price"] * RECOVERY_RATE
    for f in fixes:
        f["changes"] = json.loads(f["changes"])
    reviews = db.q("SELECT * FROM reviews WHERE product_id=? ORDER BY stars", (pid,))
    tickets = db.q("SELECT * FROM tickets WHERE product_id=? ORDER BY ts DESC LIMIT 10", (pid,))
    live = live_sessions_for(sup["id"], 20, pid)
    return render(request, "supplier_product.html", p=p, stats=stats, insights=insights, fixes=fixes,
                  reviews=reviews, tickets=tickets, live=live, flash=request.session.pop("flash", None),
                  page="supplier")


@app.post("/supplier/product/{pid}/fix/{cause}")
def supplier_fix(request: Request, pid: str, cause: str):
    sup = require(request, "supplier")
    p = product(pid)
    if p["supplier_id"] != sup["id"] or cause not in QUICK_FIX:
        raise HTTPException(403)
    changes = apply_quick_fix(p, cause)
    if changes:
        update_product(pid, {k: v["to"] for k, v in changes.items()})
        db.ex("INSERT INTO fixes (product_id, supplier_id, cause, changes) VALUES (?,?,?,?)",
              (pid, sup["id"], cause, json.dumps(changes)))
        request.session["flash"] = f"Applied fix for “{LABEL[cause]}” — live on the store now."
    else:
        request.session["flash"] = "Nothing to change — this listing already meets the recommendation."
    return RedirectResponse(f"/supplier/product/{pid}", 303)


EDITABLE = {"name": str, "price": float, "description": str, "image_count": int, "has_video": int,
            "warranty_months": int, "return_window_days": int, "has_offer": int, "cashback_pct": float,
            "eta_days": float, "cod_allowed": int, "emi_allowed": int}


def update_product(pid, fields):
    fields = {k: v for k, v in fields.items() if k in EDITABLE or k in
              ("keyword_coverage", "miscategorized", "desc_quality", "price_vs_market", "serviceable_tiers")}
    if not fields:
        return
    p = db.q("SELECT * FROM products WHERE product_id=?", (pid,), one=True)
    if "price" in fields and "price_vs_market" not in fields:
        fields["price_vs_market"] = round(p["price_vs_market"] * fields["price"] / p["price"], 3)
    if "keyword_coverage" in fields:
        from .catalog import keywords_for
        fields["keywords"] = keywords_for(pid, p["category"], fields["keyword_coverage"])
    if "desc_quality" in fields:
        from .catalog import description_for
        fields["description"] = description_for({**p, **fields})
    sets = ", ".join(f"{k}=?" for k in fields)
    db.ex(f"UPDATE products SET {sets} WHERE product_id=?", (*fields.values(), pid))


@app.get("/supplier/product/{pid}/edit", response_class=HTMLResponse)
def supplier_edit_page(request: Request, pid: str):
    sup = require(request, "supplier")
    p = product(pid)
    if p["supplier_id"] != sup["id"]:
        raise HTTPException(403)
    return render(request, "supplier_edit.html", p=p, page="supplier")


@app.post("/supplier/product/{pid}/edit")
async def supplier_edit(request: Request, pid: str):
    sup = require(request, "supplier")
    p = product(pid)
    if p["supplier_id"] != sup["id"]:
        raise HTTPException(403)
    form = await request.form()
    fields, changes = {}, {}
    for k, typ in EDITABLE.items():
        if typ is int and k in ("has_video", "has_offer", "cod_allowed", "emi_allowed"):
            v = 1 if form.get(k) else 0
        elif k in form and str(form[k]).strip() != "":
            try:
                v = typ(form[k])
            except ValueError:
                continue
        else:
            continue
        if v != p[k]:
            fields[k] = v
            changes[k] = {"from": p[k], "to": v}
    tiers = ",".join(t for t in ("tier1", "tier2", "tier3") if form.get(t))
    if tiers and tiers != p["serviceable_tiers"]:
        fields["serviceable_tiers"] = tiers
        changes["serviceable_tiers"] = {"from": p["serviceable_tiers"], "to": tiers}
    if fields:
        update_product(pid, fields)
        db.ex("INSERT INTO fixes (product_id, supplier_id, cause, changes) VALUES (?,?,?,?)",
              (pid, sup["id"], "manual_edit", json.dumps(changes)))
        request.session["flash"] = f"Saved {len(fields)} change(s)."
    return RedirectResponse(f"/supplier/product/{pid}", 303)


@app.get("/supplier/live", response_class=HTMLResponse)
def supplier_live(request: Request):
    sup = require(request, "supplier")
    return render(request, "supplier_live.html", live=live_sessions_for(sup["id"], 100), page="supplier")


@app.get("/supplier/session/{session_id}", response_class=HTMLResponse)
def supplier_session(request: Request, session_id: str):
    sup = require(request, "supplier")
    ls = db.q("""SELECT l.* FROM live_sessions l JOIN products p ON p.product_id=l.last_product_id
                 WHERE l.session_id=? AND p.supplier_id=?""", (session_id, sup["id"]), one=True)
    if not ls:
        raise HTTPException(404, "Session not found for your products")
    p = product(ls["last_product_id"])
    events = db.q("SELECT * FROM events WHERE session_id=? ORDER BY ts", (session_id,))
    s = models().score_session(events, p, tier_for_pincode(ls["pincode"]))
    for e in events:
        e["props"] = json.loads(e["props"] or "{}")
    nudges = db.q("SELECT * FROM interventions WHERE session_id=? ORDER BY ts", (session_id,))
    top = sorted(((c, v) for c, v in s["probs"].items()), key=lambda x: -x[1])[:5] if s else []
    return render(request, "supplier_session.html", ls=ls, p=p, events=events, s=s, top=top, nudges=nudges,
                  solution=SOLUTION.get(s["cause"]) if s else None, page="supplier")


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    if exc.status_code == 303:
        return RedirectResponse(exc.headers["Location"], 303)
    resp = render(request, "error.html", code=exc.status_code, detail=exc.detail)
    resp.status_code = exc.status_code
    return resp


templates.env.globals["math"] = math
templates.env.globals["FRICTIONS"] = FRICTIONS
