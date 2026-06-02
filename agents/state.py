"""Shared LangGraph state for the supplier orchestration workflow.

LangGraph threads a single typed state dict through every node. Each node
returns a partial update which LangGraph merges into the running state. We
keep the rich Pydantic objects (``SupplierData``, ``ROIReport`` ...) on the
state directly, plus an append-only ``agent_log`` that captures the agents'
reasoning so the loop is fully observable.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from app.models import (
    AgentDecision,
    ROIConfig,
    ROIReport,
    RemediationDraft,
    SupplierData,
)


class OrchestrationState(TypedDict, total=False):
    """State passed between LangGraph nodes.

    ``agent_log`` uses ``operator.add`` so each node can append its reasoning
    lines without clobbering earlier entries.
    """

    supplier: SupplierData
    config: ROIConfig
    report: ROIReport
    decision: AgentDecision
    remediation: RemediationDraft | None
    agent_log: Annotated[list[str], operator.add]
