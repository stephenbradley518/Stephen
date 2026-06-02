"""Remediation Agent.

When the Decision Agent flags a supplier, this agent drafts a concrete
(mock) outreach action — an email requesting a price review or a
performance correction — tailored to the *dominant cost driver* identified
by the ROI engine.

The draft is generated deterministically from templates (no LLM/API key
required) so the MVP runs anywhere, while leaving a clean seam to swap in an
LLM-backed writer later.
"""

from __future__ import annotations

from app.logging_config import get_logger
from app.models import ROIReport, RemediationDraft

from agents.state import OrchestrationState

logger = get_logger("agents.remediation")


def _draft_price_review(report: ROIReport, roi_gap: float) -> RemediationDraft:
    # Ask for a price that restores at least the target benchmark.
    requested = round(min(report.value_delivered, report.cost) * 0.95, 2)
    overpayment = max(0.0, report.cost - report.value_delivered)
    subject = f"Commercial review request — {report.supplier_name}"
    body = (
        f"Hello {report.supplier_name} team,\n\n"
        f"As part of our quarterly supplier value review, our current ROI on this "
        f"engagement is {report.roi:.1%}, which is {roi_gap:.1%} below our target. "
        f"The principal driver is pricing: at a unit cost of {report.cost:,.2f} against "
        f"a benchmark value of {report.value_delivered:,.2f}, we are carrying roughly "
        f"{overpayment:,.2f} of overpayment per unit.\n\n"
        f"We would like to open a price review and propose a revised target of "
        f"{requested:,.2f}. Could we schedule a call this week to discuss?\n\n"
        f"Best regards,\nProcurement Optimization"
    )
    return RemediationDraft(
        supplier_name=report.supplier_name,
        action_type="PRICE_REVIEW",
        subject=subject,
        body=body,
        requested_target_price=requested,
    )


def _draft_performance_correction(report: ROIReport, roi_gap: float) -> RemediationDraft:
    if report.dominant_cost_driver == "DELIVERY":
        problem = (
            f"on-time delivery: late-delivery costs are running at "
            f"{report.late_delivery_penalty:,.2f}, eroding the value of the contract"
        )
        ask = "a corrective delivery plan and a committed lead-time SLA"
    else:  # QUALITY
        problem = (
            f"quality: defect-related costs are running at "
            f"{report.quality_defect_cost:,.2f}, well above acceptable levels"
        )
        ask = "a quality corrective-action plan (root cause + containment)"

    subject = f"Performance correction request — {report.supplier_name}"
    body = (
        f"Hello {report.supplier_name} team,\n\n"
        f"Our latest supplier scorecard puts the ROI on this engagement at "
        f"{report.roi:.1%}, {roi_gap:.1%} below target. The primary issue is {problem}.\n\n"
        f"We are requesting {ask} within 10 business days so we can keep this "
        f"relationship in good standing. Please let us know a suitable time to review.\n\n"
        f"Best regards,\nProcurement Optimization"
    )
    return RemediationDraft(
        supplier_name=report.supplier_name,
        action_type="PERFORMANCE_CORRECTION",
        subject=subject,
        body=body,
        requested_target_price=None,
    )


def remediation_node(state: OrchestrationState) -> OrchestrationState:
    report = state["report"]
    decision = state["decision"]

    if report.dominant_cost_driver == "PRICE":
        draft = _draft_price_review(report, decision.roi_gap)
    else:
        draft = _draft_performance_correction(report, decision.roi_gap)

    log_line = (
        f"[RemediationAgent] Drafted {draft.action_type} email to "
        f"{draft.supplier_name} (driver={report.dominant_cost_driver})."
    )
    logger.info("Drafted %s for %s (driver=%s)",
                draft.action_type, draft.supplier_name, report.dominant_cost_driver)
    logger.debug("Email subject: %s", draft.subject)

    return {"remediation": draft, "agent_log": [log_line]}
