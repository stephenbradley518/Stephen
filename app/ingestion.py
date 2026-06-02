"""Data Ingestion Layer.

A flexible service that parses messy supplier data (CSV or JSON) and
normalizes it into clean :class:`~app.models.SupplierData` records using
Pandas. Enterprise supplier exports are rarely tidy, so this layer tolerates:

* inconsistent / aliased column names ("Vendor", "Unit Cost", "Lead Time (days)")
* currency symbols, thousands separators and stray whitespace ("$1,250.00 ")
* quality scores expressed as fractions (0.92) or percentages (92) or "92%"
* missing fields (records that can't be salvaged are reported, not silently
  dropped without trace)

It returns an :class:`IngestionResult` carrying the valid records *and* a
structured list of issues encountered, so the messiness is observable.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from app.logging_config import get_logger
from app.models import SupplierData

logger = get_logger("app.ingestion")

# Map of canonical field -> set of accepted (normalized) source aliases.
_COLUMN_ALIASES: dict[str, set[str]] = {
    "supplier_name": {"supplier_name", "supplier", "vendor", "vendor_name", "name", "supplier_id"},
    "cost": {"cost", "unit_cost", "price", "actual_cost", "cost_of_service", "spend"},
    "lead_time": {"lead_time", "leadtime", "lead_time_days", "lead_time_in_days", "delivery_days", "lead"},
    "quality_score": {"quality_score", "quality", "quality_rating", "acceptance_rate", "qa_score"},
    "target_price": {"target_price", "target", "benchmark_price", "target_cost", "budget_price"},
}


@dataclass
class IngestionIssue:
    """A single normalization or validation problem, tied to a source row."""

    row_index: int
    field: str | None
    message: str


@dataclass
class IngestionResult:
    """Outcome of an ingestion run."""

    records: list[SupplierData] = field(default_factory=list)
    issues: list[IngestionIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.records) > 0

    def summary(self) -> str:
        return (
            f"{len(self.records)} valid record(s), "
            f"{len(self.issues)} issue(s) encountered"
        )


def _normalize_key(key: str) -> str:
    """Lower-case, strip, and collapse separators to underscores."""
    key = key.strip().lower()
    key = re.sub(r"[^\w]+", "_", key)        # non-word -> underscore
    key = re.sub(r"_+", "_", key).strip("_")  # collapse / trim underscores
    return key


def _build_column_map(columns: list[str]) -> dict[str, list[str]]:
    """Map each canonical field to *all* matching source columns.

    Returning every alias hit (not just the first) lets us coalesce across
    columns per-row — essential when different rows use different aliases for
    the same field (e.g. one row has "Vendor Name", another has "vendor").
    """
    normalized = {col: _normalize_key(col) for col in columns}
    mapping: dict[str, list[str]] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        matches = [src_col for src_col, norm in normalized.items() if norm in aliases]
        if matches:
            mapping[canonical] = matches
    return mapping


def _coalesce(row: pd.Series, src_cols: list[str]) -> Any:
    """First non-empty value across the candidate source columns for a row."""
    for col in src_cols:
        value = row[col]
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


_NUMERIC_CLEAN_RE = re.compile(r"[^\d.\-]")


def _to_float(value: Any) -> float | None:
    """Best-effort coercion of a messy value to float (strips $ , % whitespace)."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if pd.isna(value):
            return None
        return float(value)
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "n/a", "na", "null", "-"}:
        return None
    cleaned = _NUMERIC_CLEAN_RE.sub("", text)
    if cleaned in {"", "-", ".", "-."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalize_quality(value: Any) -> float | None:
    """Normalize quality to a 0–100 scale (accepts 0.92, 92, '92%')."""
    raw = _to_float(value)
    if raw is None:
        return None
    # A fraction such as 0.92 -> 92. Values in (1, 100] assumed already %.
    if 0 <= raw <= 1:
        return round(raw * 100, 2)
    return round(raw, 2)


def _records_from_dataframe(df: pd.DataFrame) -> IngestionResult:
    result = IngestionResult()
    if df.empty:
        result.issues.append(IngestionIssue(-1, None, "Source contained no rows."))
        return result

    column_map = _build_column_map(list(df.columns))
    missing_columns = [c for c in _COLUMN_ALIASES if c not in column_map]
    if missing_columns:
        result.issues.append(
            IngestionIssue(
                -1, None,
                f"No source column matched required field(s): {', '.join(missing_columns)}.",
            )
        )

    logger.info("Resolved column map: %s", column_map)

    for idx, row in df.iterrows():
        raw: dict[str, Any] = {}
        row_issues: list[IngestionIssue] = []

        for canonical in _COLUMN_ALIASES:
            src_cols = column_map.get(canonical)
            value = _coalesce(row, src_cols) if src_cols else None

            if canonical == "supplier_name":
                name = None if value is None or pd.isna(value) else str(value).strip()
                if not name:
                    row_issues.append(IngestionIssue(int(idx), canonical, "missing supplier name"))
                raw[canonical] = name
            elif canonical == "quality_score":
                q = _normalize_quality(value)
                if q is None:
                    row_issues.append(IngestionIssue(int(idx), canonical, "missing/unparseable quality"))
                raw[canonical] = q
            else:
                num = _to_float(value)
                if num is None:
                    row_issues.append(IngestionIssue(int(idx), canonical, f"missing/unparseable {canonical}"))
                raw[canonical] = num

        if row_issues:
            for issue in row_issues:
                logger.warning("Row %s: %s", issue.row_index, issue.message)
            result.issues.extend(row_issues)
            continue

        try:
            result.records.append(SupplierData(**raw))
        except ValidationError as exc:
            for err in exc.errors():
                fld = ".".join(str(p) for p in err["loc"]) or None
                msg = f"validation: {err['msg']}"
                logger.warning("Row %s field '%s': %s", idx, fld, msg)
                result.issues.append(IngestionIssue(int(idx), fld, msg))

    logger.info("Ingestion complete: %s", result.summary())
    return result


def ingest_csv(source: str | Path | io.StringIO) -> IngestionResult:
    """Ingest supplier data from a CSV file path or in-memory buffer."""
    logger.info("Ingesting CSV from %s", getattr(source, "name", source))
    df = pd.read_csv(source, dtype=str, keep_default_na=False)
    return _records_from_dataframe(df)


def ingest_json(source: str | Path | list[dict[str, Any]]) -> IngestionResult:
    """Ingest supplier data from a JSON file path, JSON string, or list of dicts."""
    if isinstance(source, list):
        data = source
    else:
        path = Path(source)
        if path.exists():
            logger.info("Ingesting JSON from %s", path)
            data = json.loads(path.read_text())
        else:
            logger.info("Ingesting JSON from string literal")
            data = json.loads(str(source))

    if isinstance(data, dict):
        data = [data]
    df = pd.DataFrame(data, dtype=object)
    return _records_from_dataframe(df)
