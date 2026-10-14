# Synthetic Customer Journey Data — Friction Detection

Generates e-commerce clickstreams with **hidden ground-truth friction causes** for
training/evaluating a journey-friction detection model.

```bash
pip install numpy pandas pyarrow
python generate.py --sessions 100000 --users 15000 --products 2500 --out data   # ~30s on 12 cores
python generate.py --sessions 5000 --out data_dev                               # quick iteration
```

## How it works

1. `synth/config.py` — the **category × friction weight matrix `W`** (domain research),
   personas with friction sensitivities, funnel exit rates, payment failure rates.
2. `synth/entities.py` — users (persona, region tier, pincode, device, preferred payment, patience)
   and products (images, rating, warranty, returns, price vs market, ETA, serviceability, COD/EMI).
   Each product gets a per-friction *deficit* score.
3. `synth/simulator.py` — each session walks
   `landing → search|browse → PDP (compare up to 4) → cart → checkout → payment → order`.
   At each stage a friction fires with `p = sigmoid(8·W·sensitivity·deficit − 2.6)`;
   a fired friction emits its **behavioural signature** (zero-result searches, image zooms,
   1-star review sorting, pincode failures, failed coupons, payment errors…) and raises the exit
   hazard. Happy users emit "benign" versions of the same events so signatures overlap.
4. `synth/text.py` — templated support tickets (for friction-driven abandoners) and reviews.

## Output tables (`data/`)

| Table | Use | Key columns |
|---|---|---|
| `events/` (part files; `pd.read_parquet("data/events")`) | **model input** | session_id, ts, event_type, page_type, product_id, category, device, props (JSON) |
| `payments` | model input | method, gateway, status, error_code, attempt, latency_ms |
| `orders` | model input / post-purchase | status (delivered / delayed / returned) |
| `tickets` | model input (NLP) | channel, text, `label_theme` (truth) |
| `reviews` | model input (NLP) | stars, text, `label_theme` (truth) |
| `users`, `products` | context features | persona is latent — don't use it as a feature if you want to be honest |
| `labels` | **ground truth — never a feature** | converted, drop_stage, primary_cause, cause_vector, intent |
| `friction_weights.json` | the `W` used | for "did the model recover W?" evaluation |

`primary_cause = "none"` means the user left for non-friction reasons (low intent / browsing).
5% of friction labels are deliberately flipped to simulate noisy ground truth.

## Calibration (100k sessions, default seed)

Conversion ≈ 4.9%, cart abandonment ≈ 65%, ~11.5 events/session.
Category patterns follow `W`: furniture loses most to delivery, fashion to media,
electronics to warranty/reviews/pricing, grocery barely to reviews.

Tune in `config.py`: `BASE_EXIT`, `HAZARD_K`, `FIRE_A/B`, `BENIGN_RATE`, `GATEWAY_FAIL`.

## Same schema as the live clone

`events` mirrors the tracker envelope (`source = simulator`). Real clone events
(`source = client | server | bot`) go into the same table, so models trained here
can be evaluated on real sessions directly.

## Training

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python train.py --data data --out models      # ~2 min on 500k sessions
```

- `features.py` — clickstream → one row per session (≈97 features: signature counts, props-derived
  signals like zero-result searches / unserviceable pincode / coupon & payment failures, dwell
  times, furthest stage, plus catalog attributes of the viewed product). `page_exit` and
  `order_placed` are dropped to avoid leakage.
- **Cause model** (`models/cause_model.txt`): abandoned sessions → 10 frictions + `none`,
  LightGBM multiclass, sqrt-inverse-frequency weights, `none` capped at 60k in train.
- **Drop-risk model** (`models/risk_model.txt`): random session *prefixes* → will this user drop?
  Used for live intervention while the user is still on site.
- Split by day (train days 0–23, val 24–26, test 27–29).
- Explanations via LightGBM TreeSHAP (`pred_contrib=True`), grouped into friction causes,
  each mapped to a recovery action. Full metrics, examples and W-recovery in `models/report.json`.

## Web app (ShopLens)

```bash
uv pip install --python .venv/bin/python fastapi uvicorn jinja2 itsdangerous python-multipart
.venv/bin/python -m webapp.seed                     # builds webapp/shop.db (~1.5 min)
.venv/bin/uvicorn webapp.app:app --port 8000        # http://localhost:8000
```

Demo logins (password `demo1234`): `customer@shop.test` (tier-1), `tier3@shop.test` (tier-3),
`supplier1@shop.test` … `supplier10@shop.test`.

- **Customer store**: search, categories, product page, cart, coupons, checkout, mock payment gateway.
  `static/tracker.js` + server-side events write the same schema as the synthetic data to `events`.
- **Live recovery assistant**: every `/collect` batch re-scores the session (risk + cause models); when
  risk ≥ 0.4 and the friction-cause confidence ≥ 0.5 a nudge pops up with its evidence.
- **Seller Hub** (`/supplier`): diagnosed problems per product (cause, evidence, share of drop-offs,
  recoverable revenue), proposed solution, one-click fix that updates the live listing, live sessions
  with session replay.
