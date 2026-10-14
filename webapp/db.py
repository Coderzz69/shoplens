"""SQLite storage: accounts, catalog, cart/orders, live clickstream, model insights."""

import hashlib
import os
import secrets
import sqlite3
from contextlib import contextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("SHOP_DB", os.path.join(ROOT, "webapp", "shop.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('customer', 'supplier')),
    name TEXT NOT NULL,
    store_name TEXT,
    pincode TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    supplier_id INTEGER REFERENCES accounts(id),
    name TEXT, category TEXT, keywords TEXT, description TEXT,
    price REAL, price_vs_market REAL, image_count INTEGER, has_video INTEGER, desc_quality REAL,
    rating REAL, review_count INTEGER, warranty_months INTEGER, return_window_days INTEGER,
    has_offer INTEGER, cashback_pct REAL, eta_days REAL, serviceable_tiers TEXT,
    cod_allowed INTEGER, emi_allowed INTEGER, miscategorized INTEGER, keyword_coverage REAL
);
CREATE INDEX IF NOT EXISTS ix_products_cat ON products(category);
CREATE INDEX IF NOT EXISTS ix_products_supplier ON products(supplier_id);
CREATE TABLE IF NOT EXISTS reviews (review_id TEXT PRIMARY KEY, product_id TEXT, stars INTEGER, text TEXT);
CREATE INDEX IF NOT EXISTS ix_reviews_product ON reviews(product_id);
CREATE TABLE IF NOT EXISTS tickets (ticket_id TEXT PRIMARY KEY, product_id TEXT, channel TEXT, ts TEXT,
                                    text TEXT, theme TEXT);
CREATE INDEX IF NOT EXISTS ix_tickets_product ON tickets(product_id);
CREATE TABLE IF NOT EXISTS cart_items (account_id INTEGER, product_id TEXT, qty INTEGER,
                                       PRIMARY KEY (account_id, product_id));
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY, account_id INTEGER, session_id TEXT, amount REAL, discount REAL,
    method TEXT, pincode TEXT, status TEXT, ts TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS order_items (order_id TEXT, product_id TEXT, qty INTEGER, price REAL);
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY, session_id TEXT, user_id TEXT, anonymous_id TEXT, ts TEXT,
    event_type TEXT, page_type TEXT, product_id TEXT, category TEXT, device TEXT, props TEXT, source TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_session ON events(session_id, ts);
CREATE INDEX IF NOT EXISTS ix_events_product ON events(product_id);
CREATE TABLE IF NOT EXISTS live_sessions (
    session_id TEXT PRIMARY KEY, account_id INTEGER, pincode TEXT, last_product_id TEXT,
    risk REAL, cause TEXT, cause_conf REAL, evidence TEXT, converted INTEGER DEFAULT 0,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS interventions (session_id TEXT, cause TEXT, product_id TEXT, message TEXT,
                                          ts TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS product_stats (product_id TEXT PRIMARY KEY, sessions INTEGER, converted INTEGER,
                                          abandoned INTEGER, friction_abandoned INTEGER);
CREATE TABLE IF NOT EXISTS insights (
    product_id TEXT, cause TEXT, sessions INTEGER, share REAL, avg_conf REAL, evidence TEXT,
    PRIMARY KEY (product_id, cause)
);
CREATE INDEX IF NOT EXISTS ix_insights_product ON insights(product_id);
CREATE TABLE IF NOT EXISTS fixes (product_id TEXT, supplier_id INTEGER, cause TEXT, changes TEXT,
                                  ts TEXT DEFAULT CURRENT_TIMESTAMP);
"""


def connect():
    con = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


@contextmanager
def tx():
    con = connect()
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init():
    with tx() as con:
        con.executescript(SCHEMA)


def hash_password(pw, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex()
    return f"pbkdf2${salt}${digest}"


def check_password(pw, stored):
    _, salt, _ = stored.split("$")
    return secrets.compare_digest(hash_password(pw, salt), stored)


def q(sql, args=(), one=False):
    with tx() as con:
        rows = con.execute(sql, args).fetchall()
    rows = [dict(r) for r in rows]
    return (rows[0] if rows else None) if one else rows


def ex(sql, args=()):
    with tx() as con:
        cur = con.execute(sql, args)
        return cur.lastrowid
