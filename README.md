# Autonomous Supplier Optimization Agent

An MVP backend that demonstrates **agentic supplier orchestration**: ingest
messy supplier data (invoices / POs / performance logs), compute an auditable
**ROI / Supplier Value Score**, and run a stateful **LangGraph** decision +
remediation loop that drafts concrete corrective actions.

> Backend / agentic logic only — no UI (by design).

## Tech stack

| Layer            | Choice                                  |
| ---------------- | --------------------------------------- |
| Language         | Python 3.11+ (3.12+ recommended)        |
| API              | FastAPI                                 |
| AI orchestration | LangGraph (stateful `StateGraph`)       |
| Schemas          | Pydantic v2                             |
| Data handling    | Pandas (normalization)                  |

## Architecture

```
app/                      deterministic business core
  models.py               Pydantic schemas (SupplierData, ROIReport, ...)
  roi_engine.py           ROI / Supplier Value Score engine
  ingestion.py            flexible CSV/JSON ingestion + normalization (Pandas)
  main.py                 FastAPI application
  logging_config.py       shared logging (agents log their reasoning)

agents/                   LangGraph agentic workflow
  state.py                OrchestrationState (shared graph state)
  decision_agent.py       alerts when ROI < threshold
  remediation_agent.py    drafts a mock supplier email (price / performance)
  graph.py                builds + runs the StateGraph

data/                     mock + synthetic "dirty" data
tests/                    pytest suite (engine, ingestion, graph, API)
run_demo.py               end-to-end CLI showing the agents' thought process
```

### The agentic loop

```
START -> roi_node -> decision_node --[ROI < threshold]--> remediation_node -> END
                                    \--[healthy]---------------------------> END
```

Each node appends its reasoning to an append-only `agent_log`, so the entire
"thought process" is observable.

## The ROI model (explicit & auditable)

```
Cost of Service (TCO) = cost + late_delivery_penalty + quality_defect_cost
Net Value             = target_price (benchmark worth) - Cost of Service
ROI                   = Net Value / Cost of Service
```

Hidden costs that procurement teams routinely miss are made first-class:

* **late_delivery_penalty** — scales with lead time over a target (capped).
* **quality_defect_cost** — scales with `(1 - quality_score/100)`.

A normalized 0–100 **Supplier Value Score** blends ROI (50%), quality (30%)
and lead-time reliability (20%) for ranking. All assumptions live in
`ROIConfig` and are fully tunable.

**Status thresholds:** `ROI >= 15%` → HEALTHY · `0 <= ROI < 15%` → AT_RISK ·
`ROI <= 0%` → CRITICAL.

## Quickstart

```bash
pip install -r requirements.txt

# 1) End-to-end demo (watch the agents reason)
python run_demo.py                             # clean CSV
python run_demo.py data/suppliers_messy.json   # the "dirty data" sandbox

# 2) Run the API
uvicorn app.main:app --reload
# -> http://127.0.0.1:8000/docs

# 3) Tests
python -m pytest -q
```

### API endpoints

| Method | Path                  | Purpose                                       |
| ------ | --------------------- | --------------------------------------------- |
| GET    | `/health`             | Liveness check                                |
| POST   | `/suppliers/score`    | ROI report for one supplier                   |
| POST   | `/suppliers/ingest`   | Normalize messy JSON records → clean data     |
| POST   | `/agents/run`         | Full LangGraph loop for one supplier          |
| POST   | `/agents/run-batch`   | Full loop across a batch of suppliers         |

Example:

```bash
curl -s localhost:8000/agents/run -H 'content-type: application/json' -d '{
  "supplier": {"supplier_name": "Umbrella", "cost": 2100, "lead_time": 35,
               "quality_score": 70, "target_price": 1900}
}' | python -m json.tool
```

## The "dirty data" sandbox

`data/suppliers_messy.json` deliberately contains the mess enterprise exports
arrive with — aliased columns (`Vendor Name`, `Unit Cost`, `Lead Time (days)`),
currency symbols and thousands separators (`"$1,200.00"`), quality as a
fraction *or* a percentage (`0.82`, `"98%"`), per-row varying aliases, and an
unsalvageable record. The ingestion layer normalizes what it can and reports
the rest as structured `IngestionIssue`s rather than failing silently.

## Configuration

* `SOA_LOG_LEVEL` — log verbosity (`DEBUG`, `INFO` default, `WARNING`, ...).
* `ROIConfig` — pass a custom config to any scoring/workflow call (thresholds,
  penalty rates, value-score weights).

---

# Betfair + Keras sporting-event outcome predictor

A second, independent module in this repo (`betting_predictor/`): connects
to the Betfair Exchange API for live market prices and uses a Keras deep
learning model to predict event outcomes and flag "value" selections where
the model's probability beats the market's own de-vigged price.

```
betting_predictor/
  config.py            env-driven config (Betfair creds, model paths)
  betfair_client.py    Betfair API-NG JSON-RPC client (interactive + cert login)
  features.py          market data -> normalized feature vectors
  synthetic_data.py     synthetic market/outcome generator (no live account needed)
  model.py              Keras model: build / train / save / load / predict
  train.py               CLI to train on synthetic data and save the model
  app.py                 Flask API
run_betting_server.py    entry point (python run_betting_server.py)
```

### How the model works

Each selection (e.g. home / draw / away) becomes a feature vector: best back
price, best lay price, implied probability, overround-adjusted ("de-vigged")
probability, matched volume, and 5-minute price drift. A feedforward Keras
network (Dense → BatchNorm → Dropout, twice, softmax output) is trained with
categorical cross-entropy to output calibrated outcome probabilities. Since
there's no live Betfair account or historical data feed available in this
environment, `synthetic_data.py` generates structured synthetic markets
(latent true win probabilities distorted by realistic overround + noise) so
the whole pipeline runs end-to-end — swap it for a real historical loader
once you have Betfair credentials / historical data access.

`train.py` reports the model's validation log loss against a market
baseline (just using the de-vigged market price as the prediction), so you
can see whether the network is actually adding edge over the market.

### Quickstart

```bash
pip install -r requirements.txt

# 1) Train the model (synthetic data — no Betfair account required)
python -m betting_predictor.train --events 20000 --epochs 30

# 2) Run the API
python run_betting_server.py
# -> http://127.0.0.1:5000

# 3) Tests
python -m pytest tests/test_betting_predictor.py -q
```

Set Betfair credentials (see `.env.example`) to use the live endpoints:

```bash
export BETFAIR_APP_KEY=...
export BETFAIR_USERNAME=...
export BETFAIR_PASSWORD=...
```

### API endpoints

| Method | Path                              | Purpose                                            |
| ------ | ---------------------------------- | --------------------------------------------------- |
| GET    | `/health`                          | Liveness + config status check                       |
| GET    | `/betfair/event-types`             | List Betfair sport event types (requires creds)      |
| GET    | `/betfair/markets`                 | List markets (`event_type_id`, `market_type` query)  |
| GET    | `/betfair/market/<id>/prices`      | Live prices for one market                           |
| POST   | `/predict`                         | Predict outcome probabilities + value-bet flags      |
| POST   | `/train`                           | Retrain on fresh synthetic data (demo/dev)            |

`/predict` accepts either a live `market_id` (fetched from Betfair) or a raw
`selections` list for manual testing:

```bash
curl -s localhost:5000/predict -H 'content-type: application/json' -d '{
  "selections": [
    {"back_price": 2.0, "lay_price": 2.02, "matched_volume": 15000},
    {"back_price": 3.5, "lay_price": 3.6,  "matched_volume": 8000},
    {"back_price": 4.2, "lay_price": 4.3,  "matched_volume": 5000}
  ]
}' | python -m json.tool
```

Each returned selection includes `market_implied_prob`, `model_prob`,
`edge` (model minus market), and `value_bet` (true when the edge exceeds
`VALUE_EDGE_THRESHOLD`, default 0.02).

### Notes

* Betfair credentials are read from environment variables only — never
  hardcoded or logged. Certificate login is used automatically when
  `BETFAIR_CERT_FILE`/`BETFAIR_CERT_KEY` are set (Betfair's recommended flow
  for unattended clients); otherwise the client falls back to interactive
  login.
* This predicts probabilities for informational/research purposes — it is
  not betting advice, and past synthetic/backtested performance is not
  indicative of real-market results.
