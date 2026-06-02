"""FastAPI application layer.

Exposes the ROI engine, the ingestion service, and the agentic workflow over
HTTP. Run locally with:

    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for interactive API docs.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.ingestion import ingest_json
from app.logging_config import get_logger
from app.models import (
    AgentDecision,
    ROIConfig,
    ROIReport,
    RemediationDraft,
    SupplierData,
)
from app.roi_engine import calculate_roi

from agents.graph import run_workflow

logger = get_logger("app.api")

app = FastAPI(
    title="Autonomous Supplier Optimization Agent",
    description=(
        "MVP backend for agentic supplier orchestration: ingest messy supplier "
        "data, compute ROI / Supplier Value Score, and run a LangGraph decision "
        "+ remediation loop."
    ),
    version="0.1.0",
)


# --------------------------------------------------------------------------- #
# Request / response envelopes
# --------------------------------------------------------------------------- #
class ScoreRequest(BaseModel):
    supplier: SupplierData
    config: ROIConfig | None = None


class IngestRequest(BaseModel):
    records: list[dict[str, Any]] = Field(
        ..., description="Raw (possibly messy) supplier records as JSON objects."
    )


class IngestResponse(BaseModel):
    valid: list[SupplierData]
    issues: list[dict[str, Any]]
    summary: str


class WorkflowResponse(BaseModel):
    report: ROIReport
    decision: AgentDecision
    remediation: RemediationDraft | None
    agent_log: list[str]


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "supplier-optimization-agent"}


@app.post("/suppliers/score", response_model=ROIReport)
def score_supplier(req: ScoreRequest) -> ROIReport:
    """Compute the ROI report for a single supplier (no agentic loop)."""
    return calculate_roi(req.supplier, req.config)


@app.post("/suppliers/ingest", response_model=IngestResponse)
def ingest_suppliers(req: IngestRequest) -> IngestResponse:
    """Normalize a batch of messy JSON records into clean supplier data."""
    result = ingest_json(req.records)
    return IngestResponse(
        valid=result.records,
        issues=[issue.__dict__ for issue in result.issues],
        summary=result.summary(),
    )


@app.post("/agents/run", response_model=WorkflowResponse)
def run_agent(req: ScoreRequest) -> WorkflowResponse:
    """Run the full LangGraph decision + remediation workflow for one supplier."""
    state = run_workflow(req.supplier, req.config)
    if "report" not in state or "decision" not in state:
        raise HTTPException(status_code=500, detail="Workflow produced no result.")
    return WorkflowResponse(
        report=state["report"],
        decision=state["decision"],
        remediation=state.get("remediation"),
        agent_log=state.get("agent_log", []),
    )


@app.post("/agents/run-batch", response_model=list[WorkflowResponse])
def run_agent_batch(suppliers: list[SupplierData]) -> list[WorkflowResponse]:
    """Run the agentic workflow across a batch of suppliers."""
    responses: list[WorkflowResponse] = []
    for supplier in suppliers:
        state = run_workflow(supplier)
        responses.append(
            WorkflowResponse(
                report=state["report"],
                decision=state["decision"],
                remediation=state.get("remediation"),
                agent_log=state.get("agent_log", []),
            )
        )
    return responses
