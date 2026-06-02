"""ROI Calculation Engine.

Implements the Supplier Value Score and the underlying ROI formula:

    ROI = Net Value / Cost of Service

where the *Cost of Service* is the true total cost of ownership (TCO) — the
sticker price plus hidden costs that procurement teams routinely overlook:

    * late-delivery penalties (the cost of delays / expediting / stockouts)
    * quality-defect costs (rework, returns, scrap)

and the *Net Value* is the benchmark worth of the goods to the business
(``target_price``) minus that total cost of service.

Every intermediate term is surfaced on the :class:`~app.models.ROIReport`
so the result is auditable rather than a single opaque number.
"""

from __future__ import annotations

from app.logging_config import get_logger
from app.models import ROIConfig, ROIReport, SupplierData, SupplierStatus

logger = get_logger("app.roi_engine")


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _late_delivery_penalty(supplier: SupplierData, config: ROIConfig) -> float:
    """Penalty for lead time beyond the acceptable target, capped."""
    excess_days = max(0.0, supplier.lead_time - config.target_lead_time_days)
    fraction = min(
        excess_days * config.daily_late_penalty_rate,
        config.max_late_penalty_fraction,
    )
    return fraction * supplier.cost


def _quality_defect_cost(supplier: SupplierData, config: ROIConfig) -> float:
    """Cost attributable to defects — grows as quality_score falls below 100."""
    defect_fraction = (100.0 - supplier.quality_score) / 100.0
    return defect_fraction * config.defect_cost_multiplier * supplier.cost


def _value_score(
    roi: float, supplier: SupplierData, config: ROIConfig
) -> float:
    """Composite 0–100 score blending ROI, quality and lead-time reliability."""
    # ROI of 0 -> 50; +0.5 -> 100; -0.5 -> 0.
    roi_component = _clamp(50.0 + roi * 100.0)
    quality_component = _clamp(supplier.quality_score)
    # On-or-under target lead time -> 100; degrades as lead time grows.
    lead_time_component = _clamp(
        100.0 * config.target_lead_time_days / max(supplier.lead_time, 1.0)
    )
    return (
        config.weight_roi * roi_component
        + config.weight_quality * quality_component
        + config.weight_lead_time * lead_time_component
    )


def _classify(roi: float, config: ROIConfig) -> SupplierStatus:
    if roi <= config.critical_roi_threshold:
        return SupplierStatus.CRITICAL
    if roi < config.healthy_roi_threshold:
        return SupplierStatus.AT_RISK
    return SupplierStatus.HEALTHY


def _dominant_cost_driver(
    supplier: SupplierData, late_penalty: float, defect_cost: float
) -> str:
    """Identify the largest source of value erosion for targeted remediation."""
    # Overpayment relative to benchmark is a "PRICE" problem.
    overpayment = max(0.0, supplier.cost - supplier.target_price)
    candidates = {
        "PRICE": overpayment,
        "DELIVERY": late_penalty,
        "QUALITY": defect_cost,
    }
    return max(candidates, key=candidates.get)


def calculate_roi(
    supplier: SupplierData, config: ROIConfig | None = None
) -> ROIReport:
    """Compute the full ROI report for a single supplier.

    Logs each step so the calculation is observable as part of the agentic
    "thought process".
    """
    config = config or ROIConfig()
    logger.info("Scoring supplier '%s' (cost=%.2f, target=%.2f, lead=%.1fd, q=%.1f)",
                supplier.supplier_name, supplier.cost, supplier.target_price,
                supplier.lead_time, supplier.quality_score)

    late_penalty = _late_delivery_penalty(supplier, config)
    defect_cost = _quality_defect_cost(supplier, config)
    total_cost = supplier.cost + late_penalty + defect_cost

    value_delivered = supplier.target_price
    net_value = value_delivered - total_cost
    roi = net_value / total_cost if total_cost > 0 else 0.0

    value_score = _value_score(roi, supplier, config)
    status = _classify(roi, config)
    driver = _dominant_cost_driver(supplier, late_penalty, defect_cost)

    notes = [
        f"Late-delivery penalty: {late_penalty:,.2f} "
        f"(lead {supplier.lead_time:g}d vs target {config.target_lead_time_days:g}d).",
        f"Quality-defect cost: {defect_cost:,.2f} "
        f"(quality {supplier.quality_score:g}/100).",
        f"Total cost of service (TCO): {total_cost:,.2f}.",
        f"Net value: {net_value:,.2f} -> ROI {roi:.1%}.",
        f"Dominant cost driver: {driver}.",
    ]

    logger.info("  -> TCO=%.2f, net_value=%.2f, ROI=%.1f%%, score=%.1f, status=%s",
                total_cost, net_value, roi * 100, value_score, status.value)

    return ROIReport(
        supplier_name=supplier.supplier_name,
        cost=supplier.cost,
        late_delivery_penalty=round(late_penalty, 2),
        quality_defect_cost=round(defect_cost, 2),
        total_cost_of_service=round(total_cost, 2),
        value_delivered=round(value_delivered, 2),
        net_value=round(net_value, 2),
        roi=round(roi, 4),
        value_score=round(value_score, 1),
        status=status,
        dominant_cost_driver=driver,
        notes=notes,
    )
