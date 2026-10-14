"""Session simulator: walks a user through the funnel, fires frictions from hidden
(category x user x product) scores, and emits ONLY observable events. The cause
of each drop-off is written to a separate labels record."""

import json
import math
from datetime import datetime, timedelta

import numpy as np

from .config import (BASE_EXIT, BENIGN_RATE, CATEGORY_PATHS, FIRE_A, FIRE_B, FRICTIONS, GATEWAY_FAIL,
                     HAZARD_K, INTENT_EXIT_MULT, INTENT_MIX, LABEL_NOISE, PAYMENT_ERRORS, SEARCH_TERMS,
                     W)
from .entities import FI

EPOCH = datetime(2026, 9, 1)
HOUR_WEIGHTS = np.array([1, .6, .4, .3, .3, .4, .8, 1.5, 2, 2.3, 2.5, 2.7,
                         3, 3.2, 2.8, 2.5, 2.4, 2.6, 3, 3.6, 4.2, 4.5, 3.5, 2])
HOUR_P = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()


def _sig(x):
    return 1 / (1 + math.exp(-x))


class Session:
    def __init__(self, sid, user, rng, by_cat):
        self.sid, self.u, self.rng, self.by_cat = sid, user, rng, by_cat
        self.events, self.payments = [], []
        day = int(rng.integers(0, 30))
        hour = int(rng.choice(24, p=HOUR_P))
        self.t = EPOCH + timedelta(days=day, hours=hour, seconds=int(rng.integers(0, 3600)))
        self.start = self.t
        self.intent = list(INTENT_MIX)[rng.choice(3, p=list(INTENT_MIX.values()))]
        self.cat = list(W)[rng.choice(len(W), p=user["_affinity"])]
        self.contrib = np.zeros(len(FRICTIONS))
        self.page = "home"
        self.product = None
        self.viewed = []

    # ---------- primitives ----------
    def emit(self, etype, dwell=2.0, **props):
        self.t += timedelta(seconds=float(dwell * self.rng.lognormal(0, 0.35)))
        self.events.append({
            "event_id": f"{self.sid}_{len(self.events):03d}",
            "session_id": self.sid,
            "user_id": self.u["user_id"] if self.u["is_logged_in"] else None,
            "anonymous_id": self.u["anonymous_id"],
            "ts": self.t,
            "event_type": etype,
            "page_type": self.page,
            "product_id": self.product["product_id"] if self.product else None,
            "category": self.cat,
            "device": self.u["device"],
            "props": json.dumps(props) if props else "{}",
            "source": "simulator",
        })

    def deficit(self, f):
        p, u = self.product, self.u
        d = p["_deficit"][FI[f]]
        if f == "delivery" and u["region_tier"] not in p["serviceable_tiers"]:
            return 1.0
        if f == "payment":
            pref = u["preferred_payment"]
            if (pref == "cod" and not p["cod_allowed"]) or (pref == "emi" and not p["emi_allowed"]):
                return 0.9
        return d

    def score(self, f):
        return W[self.cat][FI[f]] * self.u["_sens"][FI[f]] * self.deficit(f)

    def fires(self, f):
        s = self.score(f)
        p = 0.95 if (f == "delivery" and self.deficit(f) == 1.0) else _sig(FIRE_A * s + FIRE_B)
        return self.rng.random() < p, s

    def benign(self, f):
        return self.rng.random() < BENIGN_RATE * W[self.cat][FI[f]] * self.u["_sens"][FI[f]]

    def noise(self, fired_any):
        r = self.rng
        if r.random() < (0.25 if fired_any else 0.07):
            self.emit("tab_hidden", 1)
            self.emit("tab_visible", r.choice([5, 20, 60, 180]))
        if self.u["device"] == "mobile" and r.random() < (0.08 if fired_any else 0.015):
            self.emit("rage_click", 0.5, target=r.choice(["image", "button", "price", "link"]))
        if r.random() < 0.03:
            self.emit("slow_load", 0.2, ms=int(r.integers(2500, 9000)))

    def exit_check(self, stage, step, force=False):
        """Returns True if the user leaves now. Decides if friction or base intent caused it."""
        h_base = min(0.95, BASE_EXIT[stage] * INTENT_EXIT_MULT[self.intent])
        h_fric = 1 - math.exp(-HAZARD_K * step.sum() / self.u["patience"])
        p_exit = 1 - (1 - h_base) * (1 - h_fric)
        self.contrib += step
        if force:
            # blocked journey: friction (if any) is the cause, not intent
            h_base = 0.0 if step.sum() > 0 else 1.0
        elif self.rng.random() >= p_exit:
            return False
        self.drop_stage = stage
        if step.sum() > 0 and self.rng.random() < h_fric / (h_base + h_fric + 1e-9):
            self.cause = FRICTIONS[int(np.argmax(self.contrib + step))]
        else:
            self.cause = "none"
        self.emit("page_exit", 1, dwell_ms=int((self.t - self.start).total_seconds() * 1000))
        return True

    # ---------- stages ----------
    def run(self):
        self.cause, self.drop_stage, self.converted = None, None, False
        self.page = "home"
        self.emit("page_view", 0, referrer=self.rng.choice(["direct", "google", "instagram", "push", "email"]))
        if self.exit_check("landing", np.zeros(len(FRICTIONS))):
            return self.finish()

        target = self.by_cat[self.cat][self.rng.integers(len(self.by_cat[self.cat]))]
        for view in range(4):
            stage = "search" if self.rng.random() < 0.6 else "browse"
            self.product = None
            step = self.search(target) if stage == "search" else self.browse(target)
            if self.exit_check(stage, step):
                return self.finish()
            self.product = target
            step = self.pdp()
            if self.exit_check("pdp", step):
                return self.finish()
            if self.rng.random() < 0.65 or view == 3:
                break
            # keep comparing: pick another product in the same category
            target = self.by_cat[self.cat][self.rng.integers(len(self.by_cat[self.cat]))]
        for stage, fn in (("cart", self.cart), ("checkout", self.checkout), ("payment", self.pay)):
            self.blocked = False
            step = fn()
            if stage == "payment" and self.paid:
                break
            # undeliverable address or failed/unavailable payment: the journey cannot continue
            force = self.blocked or stage == "payment"
            if self.exit_check(stage, step, force=force):
                return self.finish()
        self.converted = True
        self.page = "order_confirmation"
        self.emit("order_placed", 2, amount=self.product["price"], method=self.method)
        return self.finish()

    def search(self, target):
        r, step = self.rng, np.zeros(len(FRICTIONS))
        self.page = "search"
        good, bad = SEARCH_TERMS[self.cat][r.integers(len(SEARCH_TERMS[self.cat]))]
        self.product = target
        fired, s = self.fires("search")
        self.product = None
        if fired:
            step[FI["search"]] = s
            self.emit("search_submitted", 4, query=bad, results_count=0)
            for _ in range(int(r.integers(1, 4))):
                q = bad if r.random() < 0.4 else bad + " " + r.choice(["cheap", "best", "new", "online"])
                self.emit("search_refined", 5, query=q, results_count=int(r.choice([0, 0, 2, 5])))
            if r.random() < 0.5:
                self.emit("filter_applied", 3, filter=r.choice(["price", "brand", "rating"]))
                self.emit("filter_removed", 3)
        else:
            self.emit("search_submitted", 4, query=good, results_count=int(r.integers(20, 400)))
        self.emit("search_result_clicked", 6, rank=int(r.integers(5, 25) if fired else r.integers(1, 6)))
        self.noise(fired)
        return step

    def browse(self, target):
        r, step = self.rng, np.zeros(len(FRICTIONS))
        self.page = "category"
        self.product = target
        fired, s = self.fires("category_nav")
        self.product = None
        paths = CATEGORY_PATHS[self.cat]
        self.emit("category_viewed", 3, path=paths[r.integers(len(paths))])
        if fired:
            step[FI["category_nav"]] = s
            for _ in range(int(r.integers(2, 5))):
                self.emit("category_viewed", 4, path=paths[r.integers(len(paths))])
                if r.random() < 0.5:
                    self.emit("nav_back", 2)
            self.emit("filter_applied", 3, filter=r.choice(["price", "brand", "size", "rating"]))
            self.emit("filter_removed", 3)
        elif r.random() < 0.4:
            self.emit("filter_applied", 3, filter=r.choice(["price", "brand", "rating"]))
        self.emit("product_clicked", 5, position=int(r.integers(8, 40) if fired else r.integers(1, 10)))
        self.noise(fired)
        return step

    def pdp(self):
        r, p, step = self.rng, self.product, np.zeros(len(FRICTIONS))
        self.page = "pdp"
        self.viewed.append(p["product_id"])
        self.emit("page_view", 2, price=p["price"], rating=p["rating"], has_offer=p["has_offer"])
        fired_any = False
        order = ["media", "description", "reviews", "warranty_returns", "pricing", "offers", "delivery"]
        r.shuffle(order)
        for f in order:
            fired, s = self.fires(f)
            if fired:
                step[FI[f]] = s
                fired_any = True
                getattr(self, "sig_" + f)(True)
            elif self.benign(f):
                getattr(self, "sig_" + f)(False)
        self.emit("scroll_depth", 3, pct=int(r.choice([50, 75, 100] if fired_any else [25, 50, 75, 100])))
        self.noise(fired_any)
        return step

    # ---------- PDP behavioural signatures ----------
    def sig_media(self, bad):
        r = self.rng
        n = int(r.integers(4, 10)) if bad else int(r.integers(1, 4))
        for i in range(n):
            self.emit("gallery_swipe", 1.5, index=i % max(1, self.product["image_count"]))
        if bad or r.random() < 0.3:
            for _ in range(int(r.integers(2, 5)) if bad else 1):
                self.emit("image_zoom", 2.5)
        if self.product["has_video"] and (bad or r.random() < 0.3):
            self.emit("video_play", 1)
            self.emit("video_pause", 8, watched_pct=int(r.integers(10, 40)) if bad else int(r.integers(60, 100)))

    def sig_description(self, bad):
        r = self.rng
        self.emit("section_viewed", 6 if bad else 3, section="specs")
        if bad:
            self.emit("section_viewed", 8, section="description")
            self.emit("qna_opened", 4)
            if r.random() < 0.6:
                self.emit("qna_searched", 6, query=r.choice(["size", "material", "compatible", "dimensions", "ingredients"]))
            if r.random() < 0.5:
                self.emit("compare_added", 2)

    def sig_reviews(self, bad):
        r = self.rng
        self.emit("reviews_opened", 2)
        if bad:
            self.emit("reviews_sorted", 2, by="lowest")
            if r.random() < 0.6:
                self.emit("review_filter", 2, stars=1)
            self.emit("review_dwell", 20, ms=int(r.integers(15000, 60000)))
        else:
            self.emit("review_dwell", 8, ms=int(r.integers(3000, 15000)))

    def sig_warranty_returns(self, bad):
        r = self.rng
        p = self.product
        self.emit("policy_viewed", 4 if bad else 2, policy="returns", return_window_days=p["return_window_days"])
        if bad:
            if p["category"] in ("electronics", "furniture"):
                self.emit("policy_viewed", 4, policy="warranty", warranty_months=p["warranty_months"])
            if r.random() < 0.4:
                self.emit("help_opened", 5, origin="pdp")

    def sig_pricing(self, bad):
        r = self.rng
        if bad or r.random() < 0.3:
            self.emit("price_history_opened", 3)
        if bad:
            for _ in range(int(r.integers(1, 3))):
                self.emit("similar_product_viewed", 6, price_delta_pct=int(r.integers(-25, -3)))
            self.emit("tab_hidden", 1)
            self.emit("tab_visible", r.choice([30, 90, 240]))

    def sig_offers(self, bad):
        r = self.rng
        self.emit("offers_viewed", 3, offers_count=int(self.product["has_offer"]) * int(r.integers(1, 4)))
        if bad and r.random() < 0.5:
            self.emit("bank_offer_toggled", 2)

    def sig_delivery(self, bad):
        r, p, u = self.rng, self.product, self.u
        serviceable = u["region_tier"] in p["serviceable_tiers"]
        self.emit("pincode_check", 4, pincode=u["pincode"][:3] + "***", serviceable=serviceable,
                  eta_days=p["eta_days"] if serviceable else None)
        if bad:
            for _ in range(int(r.integers(1, 3))):
                self.emit("pincode_changed", 5, serviceable=serviceable and r.random() < 0.7)
            if serviceable:
                self.emit("delivery_slot_viewed", 3, earliest_days=p["eta_days"])

    # ---------- cart / checkout / payment ----------
    def cart(self):
        r, p, step = self.rng, self.product, np.zeros(len(FRICTIONS))
        self.emit("add_to_cart", 2, price=p["price"])
        self.page = "cart"
        self.emit("page_view", 3, cart_value=p["price"])
        fired_any = False
        fired, s = self.fires("pricing")
        if fired:
            # price shock at cart: fees / shipping revealed
            step[FI["pricing"]] = s * 0.8
            fired_any = True
            self.emit("fee_breakdown_viewed", 4, delivery_fee=int(r.choice([0, 40, 79, 99])),
                      platform_fee=int(r.choice([0, 5, 10])))
            if r.random() < 0.5:
                self.emit("remove_from_cart", 3)
                self.emit("add_to_cart", 4, price=p["price"])
        fired, s = self.fires("offers")
        if fired:
            step[FI["offers"]] = s
            fired_any = True
            for _ in range(int(r.integers(1, 4))):
                self.emit("coupon_applied", 6, code=r.choice(["FIRST50", "SAVE10", "FEST20", "BANK5"]),
                          success=False, reason=r.choice(["min_order_not_met", "expired", "not_applicable"]))
        elif self.benign("offers"):
            self.emit("coupon_applied", 5, code="SAVE10", success=True, reason=None)
        self.noise(fired_any)
        return step

    def checkout(self):
        r, p, u, step = self.rng, self.product, self.u, np.zeros(len(FRICTIONS))
        self.page = "checkout"
        self.emit("checkout_step", 3, step="address")
        if not u["is_logged_in"]:
            self.emit("login_prompt", 4)
        serviceable = u["region_tier"] in p["serviceable_tiers"]
        if not serviceable:
            # undeliverable discovered at address step; nearly always fatal
            step[FI["delivery"]] = max(0.6, self.score("delivery"))
            self.emit("address_not_serviceable", 4, pincode=u["pincode"][:3] + "***")
            self.emit("address_changed", 8)
            self.emit("address_not_serviceable", 3, pincode=u["pincode"][:3] + "***")
            self.noise(True)
            self.blocked = True
            return step
        fired, s = self.fires("delivery")
        self.emit("delivery_estimate_shown", 2, eta_days=p["eta_days"])
        if fired:
            step[FI["delivery"]] = s * 0.7
            self.emit("delivery_slot_viewed", 5, earliest_days=p["eta_days"])
        self.emit("checkout_step", 4, step="summary")
        self.noise(fired)
        return step

    def pay(self):
        r, p, u, step = self.rng, self.product, self.u, np.zeros(len(FRICTIONS))
        self.page = "payment"
        self.paid = False
        self.emit("checkout_step", 2, step="payment")
        pref = u["preferred_payment"]
        available = {"upi", "card", "netbanking", "wallet"} | ({"cod"} if p["cod_allowed"] else set()) \
            | ({"emi"} if p["emi_allowed"] else set())
        w_s = W[self.cat][FI["payment"]] * u["_sens"][FI["payment"]]
        method = pref
        if pref not in available:
            self.emit("payment_method_unavailable", 3, method=pref)
            step[FI["payment"]] += w_s * 1.5
            if r.random() < 0.5 + 0.4 * w_s * 0.9:
                self.emit("help_opened", 4, origin="payment")
            method = r.choice(sorted(available - {"cod", "emi"}))
            self.emit("payment_method_selected", 4, method=method, switched=True)
        else:
            self.emit("payment_method_selected", 3, method=method, switched=False)
        self.method = method
        if method == "cod":
            self.paid = step.sum() == 0 or r.random() > step.sum()
            if self.paid:
                self.emit("payment_result", 2, method="cod", status="success", error_code=None)
                self.payments.append(self._pay_row(method, "success", None, 1))
            return step
        for attempt in range(1, 4):
            self.emit("payment_attempt", 2, method=method, attempt=attempt)
            fail_p = GATEWAY_FAIL[method] * (1.8 if u["region_tier"] == "tier3" else 1)
            if r.random() < fail_p:
                err = r.choice(PAYMENT_ERRORS[method])
                self.emit("payment_result", 6, method=method, status="failed", error_code=err)
                self.payments.append(self._pay_row(method, "failed", err, attempt))
                step[FI["payment"]] += w_s * 0.9
                if r.random() < 1 - math.exp(-HAZARD_K * step.sum() / u["patience"]):
                    return step  # abandon after failure -> exit_check will likely fire
                if r.random() < 0.4:
                    alt = r.choice(sorted(available - {method, "cod", "emi"}))
                    self.emit("payment_method_selected", 3, method=alt, switched=True)
                    method = self.method = alt
                continue
            self.emit("payment_result", 4, method=method, status="success", error_code=None)
            self.payments.append(self._pay_row(method, "success", None, attempt))
            self.paid = True
            return step
        return step

    def _pay_row(self, method, status, err, attempt):
        return {"txn_id": f"t_{self.sid}_{len(self.payments)}", "session_id": self.sid,
                "user_id": self.u["user_id"], "ts": self.t, "method": method,
                "gateway": {"upi": "upi_psp", "card": "card_pg", "netbanking": "nb_pg",
                            "wallet": "wallet_pg", "emi": "card_pg", "cod": "none"}[method],
                "status": status, "error_code": err, "attempt": attempt,
                "latency_ms": int(self.rng.lognormal(7, 0.5)) if method != "cod" else 0,
                "amount": self.product["price"]}

    def finish(self):
        cause = self.cause
        if cause not in (None, "none") and self.rng.random() < LABEL_NOISE:
            cause = FRICTIONS[self.rng.integers(len(FRICTIONS))]
        return {
            "session_id": self.sid, "user_id": self.u["user_id"], "category": self.cat,
            "start_ts": self.start, "end_ts": self.t, "n_events": len(self.events),
            "products_viewed": len(self.viewed),
            "reached_cart": any(e["event_type"] == "add_to_cart" for e in self.events),
            "converted": self.converted, "drop_stage": self.drop_stage,
            "primary_cause": None if self.converted else cause,
            # hidden latent info, NOT model input
            "persona": self.u["persona"], "intent": self.intent,
            "cause_vector": json.dumps({f: round(float(v), 4) for f, v in zip(FRICTIONS, self.contrib) if v > 0}),
            "last_product_id": self.product["product_id"] if self.product else None,
        }
