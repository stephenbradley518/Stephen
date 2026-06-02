"""Tests for the ingestion / normalization layer (the 'dirty data' sandbox)."""

from __future__ import annotations

import io

from app.ingestion import ingest_csv, ingest_json


MESSY_RECORDS = [
    # Aliased columns, currency symbols, percent quality, whitespace.
    {"Vendor Name": "  Acme  ", "Unit Cost": "$1,200.00",
     "Lead Time (days)": "10", "Quality": "98%", "Target Price": "$1,500"},
    # Quality as a 0–1 fraction; thousands separator on benchmark.
    {"vendor": "Globex", "price": "1800", "leadtime": 22,
     "quality_rating": 0.82, "benchmark_price": "1,750"},
    # Unsalvageable row: missing cost and lead time.
    {"vendor_name": "Broken Co", "unit_cost": "n/a", "lead_time_days": "",
     "quality": "", "target_cost": "1000"},
]


def test_json_normalizes_aliases_and_currency():
    result = ingest_json(MESSY_RECORDS)
    by_name = {r.supplier_name: r for r in result.records}

    assert "Acme" in by_name              # whitespace stripped
    acme = by_name["Acme"]
    assert acme.cost == 1200.0            # "$1,200.00" -> 1200.0
    assert acme.target_price == 1500.0
    assert acme.quality_score == 98.0     # "98%" -> 98.0


def test_quality_fraction_scaled_to_percentage():
    result = ingest_json(MESSY_RECORDS)
    globex = next(r for r in result.records if r.supplier_name == "Globex")
    assert globex.quality_score == 82.0   # 0.82 -> 82.0


def test_unsalvageable_row_is_reported_not_crashed():
    result = ingest_json(MESSY_RECORDS)
    names = {r.supplier_name for r in result.records}
    assert "Broken Co" not in names
    assert any(issue.row_index == 2 for issue in result.issues)
    assert len(result.records) == 2


def test_csv_ingestion_from_buffer():
    csv_text = (
        "Supplier,Unit Cost,Lead Time,Quality,Target\n"
        "Soylent,$800,9,99%,1000\n"
    )
    result = ingest_csv(io.StringIO(csv_text))
    assert result.ok
    rec = result.records[0]
    assert rec.supplier_name == "Soylent"
    assert rec.cost == 800.0
    assert rec.quality_score == 99.0
