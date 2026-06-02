"""LangGraph workflow definition and runner.

Builds the stateful agentic graph:

    START -> roi_node -> decision_node --[needs_remediation]--> remediation_node -> END
                                        \\--[healthy]-------------------------------> END

``build_graph`` compiles the graph once; ``run_workflow`` executes it for a
single supplier and returns the final :class:`OrchestrationState`.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.logging_config import get_logger
from app.models import ROIConfig, SupplierData
from app.roi_engine import calculate_roi

from agents.decision_agent import decision_node, route_after_decision
from agents.remediation_agent import remediation_node
from agents.state import OrchestrationState

logger = get_logger("agents.graph")


def roi_node(state: OrchestrationState) -> OrchestrationState:
    """Entry node: run the deterministic ROI engine and record the report."""
    supplier = state["supplier"]
    config = state.get("config") or ROIConfig()
    report = calculate_roi(supplier, config)
    log_line = (
        f"[ROIEngine] {supplier.supplier_name}: ROI {report.roi:.1%}, "
        f"value score {report.value_score:.1f}, status {report.status.value}."
    )
    return {"report": report, "config": config, "agent_log": [log_line]}


def build_graph():
    """Construct and compile the LangGraph ``StateGraph``."""
    graph = StateGraph(OrchestrationState)

    graph.add_node("roi", roi_node)
    graph.add_node("decision", decision_node)
    graph.add_node("remediation", remediation_node)

    graph.add_edge(START, "roi")
    graph.add_edge("roi", "decision")
    graph.add_conditional_edges(
        "decision",
        route_after_decision,
        {"remediate": "remediation", "done": END},
    )
    graph.add_edge("remediation", END)

    return graph.compile()


@lru_cache(maxsize=1)
def _compiled_graph():
    logger.info("Compiling supplier orchestration graph.")
    return build_graph()


def run_workflow(
    supplier: SupplierData, config: ROIConfig | None = None
) -> OrchestrationState:
    """Run the full agentic workflow for one supplier."""
    logger.info("=== Orchestrating supplier: %s ===", supplier.supplier_name)
    initial: OrchestrationState = {
        "supplier": supplier,
        "config": config or ROIConfig(),
        "agent_log": [],
    }
    final_state = _compiled_graph().invoke(initial)
    logger.info("=== Done: %s ===", supplier.supplier_name)
    return final_state
