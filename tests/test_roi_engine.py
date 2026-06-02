"""Tests for the ROI calculation engine."""

from __future__ import annotations

from app.models import ROIConfig, SupplierData, SupplierStatus
from app.roi_engine import calculate_roi


def _supplier(**overrides) -> SupplierData:
    base = dict(
        supplier_name="Test Co",
        cost=1000.0,
        lead_time=10.0,
        quality_score=100.0,
        target_price=1300.0,
    )
    base.update(overrides)
    return SupplierData(**base)


def test_no_hidden_costs_when_fast_and_perfect():
    # Lead time under target, perfect quality -> TCO == cost.
    report = calculate_roi(_supplier())
    assert report.late_delivery_penalty == 0.0
    assert report.quality_defect_cost == 0.0
    assert report.total_cost_of_service == 1000.0
    # ROI = (1300 - 1000) / 1000 = 0.30
    assert report.roi == 0.30
    assert report.status is SupplierStatus.HEALTHY


def test_late_delivery_penalty_applied_and_capped():
    config = ROIConfig()
    # 200 days late => uncapped would exceed the 50% cap.
    report = calculate_roi(_supplier(lead_time=200.0), config)
    assert report.late_delivery_penalty == config.max_late_penalty_fraction * 1000.0
    assert report.dominant_cost_driver == "DELIVERY"


def test_quality_defect_cost_scales_with_quality():
    report = calculate_roi(_supplier(quality_score=80.0))
    # (1 - 0.8) * 1.0 * 1000 = 200
    assert report.quality_defect_cost == 200.0


def test_critical_status_for_negative_roi():
    # Overpaying (cost > target) with poor quality -> value-destroying.
    report = calculate_roi(_supplier(cost=2000.0, target_price=1000.0, quality_score=50.0))
    assert report.net_value < 0
    assert report.roi < 0
    assert report.status is SupplierStatus.CRITICAL
    assert report.dominant_cost_driver == "PRICE"


def test_value_score_within_bounds():
    report = calculate_roi(_supplier(lead_time=300.0, quality_score=0.0, cost=5000.0, target_price=100.0))
    assert 0.0 <= report.value_score <= 100.0


def test_threshold_is_configurable():
    strict = ROIConfig(healthy_roi_threshold=0.5)
    report = calculate_roi(_supplier(), strict)  # ROI 0.30 < 0.50
    assert report.status is SupplierStatus.AT_RISK
