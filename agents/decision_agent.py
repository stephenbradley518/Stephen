"""Decision Agent.

Consumes the ROI report and decides whether the supplier relationship needs
intervention. The trigger condition is the one specified in the PRD:

    raise an alert (needs_remediation=True) when ROI < healthy threshold.

The agent logs its reasoning to both the application log and the shared
``agent_log`` so the decision is transparent.
"""

from __future__ import annotations

from app.logging_config import get_logger
from app.models import AgentDecision, ROIConfig, SupplierStatus

from agents.state import OrchestrationState

logger = get_logger("agents.decision")


def decision_node(state: OrchestrationState) -> OrchestrationState:
    report = state["report"]
    config: ROIConfig = state.get("config") or ROIConfig()

    roi_gap = config.healthy_roi_threshold - report.roi
    needs_remediation = report.roi < config.healthy_roi_threshold

    if not needs_remediation:
        reason = (
            f"ROI {report.roi:.1%} >= target {config.healthy_roi_threshold:.0%}; "
            f"supplier is {report.status.value}. No action required."
        )
    elif report.status is SupplierStatus.CRITICAL:
        reason = (
            f"ROI {report.roi:.1%} is value-destroying (<= {config.critical_roi_threshold:.0%}). "
            f"Escalate: dominant driver is {report.dominant_cost_driver}."
        )
    else:
        reason = (
            f"ROI {report.roi:.1%} is below target {config.healthy_roi_threshold:.0%} "
            f"(gap {roi_gap:.1%}). Flag for remediation; "
            f"dominant driver is {report.dominant_cost_driver}."
        )

    decision = AgentDecision(
        needs_remediation=needs_remediation,
        reason=reason,
        roi_gap=round(roi_gap, 4),
    )

    log_line = f"[DecisionAgent] {reason}"
    logger.info("%s", reason)

    return {"decision": decision, "agent_log": [log_line]}


def route_after_decision(state: OrchestrationState) -> str:
    """Conditional edge: branch to remediation only when needed."""
    return "remediate" if state["decision"].needs_remediation else "done"
