"""End-to-end tests for the LangGraph agentic workflow and the API."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.models import SupplierData
from agents.graph import run_workflow

client = TestClient(app)


def test_healthy_supplier_skips_remediation():
    supplier = SupplierData(
        supplier_name="Soylent", cost=800, lead_time=9,
        quality_score=99, target_price=1000,
    )
    state = run_workflow(supplier)
    assert state["report"].status.value == "HEALTHY"
    assert state["decision"].needs_remediation is False
    assert state.get("remediation") is None
    # Both upstream nodes logged their reasoning.
    assert any("ROIEngine" in line for line in state["agent_log"])
    assert any("DecisionAgent" in line for line in state["agent_log"])


def test_overpriced_supplier_triggers_price_review():
    supplier = SupplierData(
        supplier_name="Umbrella", cost=2100, lead_time=35,
        quality_score=70, target_price=1900,
    )
    state = run_workflow(supplier)
    assert state["decision"].needs_remediation is True
    draft = state["remediation"]
    assert draft is not None
    assert draft.action_type in {"PRICE_REVIEW", "PERFORMANCE_CORRECTION"}
    assert draft.supplier_name == "Umbrella"
    assert any("RemediationAgent" in line for line in state["agent_log"])


def test_api_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_api_run_agent_endpoint():
    payload = {
        "supplier": {
            "supplier_name": "Globex", "cost": 1800, "lead_time": 22,
            "quality_score": 82, "target_price": 1750,
        }
    }
    resp = client.post("/agents/run", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "report" in body and "decision" in body
    assert body["report"]["supplier_name"] == "Globex"


def test_api_ingest_endpoint_normalizes():
    payload = {"records": [
        {"Vendor Name": "Acme", "Unit Cost": "$1,200", "Lead Time (days)": "10",
         "Quality": "98%", "Target Price": "$1,500"},
    ]}
    resp = client.post("/suppliers/ingest", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"][0]["cost"] == 1200.0
    assert body["valid"][0]["quality_score"] == 98.0
