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
