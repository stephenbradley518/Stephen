"""Pydantic schemas — the contract enforced across every layer.

These models are intentionally the first thing built: ingestion produces
``SupplierData``, the ROI engine consumes it and produces an ``ROIReport``,
and the agents pass both through the LangGraph state. Schema enforcement
here means the rest of the system can trust its inputs.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class SupplierStatus(str, Enum):
    """Health classification derived from a supplier's ROI."""

    HEALTHY = "HEALTHY"      # ROI comfortably above target
    AT_RISK = "AT_RISK"      # positive but below the target threshold
    CRITICAL = "CRITICAL"    # ROI at or below zero — value-destroying


class SupplierData(BaseModel):
    """A single, normalized supplier record.

    All monetary values are assumed to be in the same currency and refer to
    the same unit of analysis (e.g. per-contract or per-order). ``lead_time``
    is in days and ``quality_score`` is a 0–100 percentage.
    """

    supplier_name: str = Field(..., description="Supplier / vendor name.")
    cost: float = Field(..., gt=0, description="Actual price paid (cost of service).")
    lead_time: float = Field(..., ge=0, description="Average delivery lead time, days.")
    quality_score: float = Field(
        ..., ge=0, le=100, description="Quality / acceptance rate, 0–100."
    )
    target_price: float = Field(
        ..., gt=0, description="Benchmark value the goods are worth to the business."
    )

    @field_validator("supplier_name")
    @classmethod
    def _non_empty_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("supplier_name must not be empty")
        return v


class ROIConfig(BaseModel):
    """Tunable assumptions for the ROI / Supplier-Value-Score engine.

    Defaults are deliberately conservative and fully documented so the model
    is auditable rather than a black box.
    """

    target_lead_time_days: float = Field(
        14.0, gt=0, description="Lead time at/under which no late penalty applies."
    )
    daily_late_penalty_rate: float = Field(
        0.015, ge=0,
        description="Late penalty per excess day, as a fraction of cost (1.5%/day).",
    )
    max_late_penalty_fraction: float = Field(
        0.50, ge=0,
        description="Cap on the late penalty, as a fraction of cost.",
    )
    defect_cost_multiplier: float = Field(
        1.0, ge=0,
        description="Scales (1 - quality/100) * cost into a defect/rework cost.",
    )
    healthy_roi_threshold: float = Field(
        0.15, description="ROI at/above which a supplier is HEALTHY."
    )
    critical_roi_threshold: float = Field(
        0.0, description="ROI at/below which a supplier is CRITICAL."
    )

    # Supplier Value Score weights (must sum to 1.0).
    weight_roi: float = Field(0.5, ge=0, le=1)
    weight_quality: float = Field(0.3, ge=0, le=1)
    weight_lead_time: float = Field(0.2, ge=0, le=1)


class ROIReport(BaseModel):
    """The output of the ROI engine for one supplier — fully itemised."""

    supplier_name: str

    # Cost build-up (total cost of ownership).
    cost: float = Field(..., description="Base cost of service paid.")
    late_delivery_penalty: float = Field(..., description="Hidden cost of late delivery.")
    quality_defect_cost: float = Field(..., description="Hidden cost of quality defects.")
    total_cost_of_service: float = Field(..., description="cost + hidden costs (TCO).")

    # Value & return.
    value_delivered: float = Field(..., description="Benchmark value to the business.")
    net_value: float = Field(..., description="value_delivered - total_cost_of_service.")
    roi: float = Field(..., description="net_value / total_cost_of_service.")
    value_score: float = Field(..., ge=0, le=100, description="Composite 0–100 score.")

    status: SupplierStatus
    dominant_cost_driver: str = Field(
        ..., description="Largest erosion factor: PRICE | DELIVERY | QUALITY."
    )
    notes: list[str] = Field(default_factory=list, description="Human-readable rationale.")


class AgentDecision(BaseModel):
    """Output of the Decision Agent."""

    needs_remediation: bool
    reason: str
    roi_gap: float = Field(
        ..., description="healthy_threshold - roi (positive => below target)."
    )


class RemediationDraft(BaseModel):
    """A drafted (mock) outreach action from the Remediation Agent."""

    supplier_name: str
    action_type: str = Field(..., description="e.g. PRICE_REVIEW | PERFORMANCE_CORRECTION.")
    subject: str
    body: str
    requested_target_price: float | None = None
