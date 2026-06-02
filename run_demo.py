"""End-to-end demo / CLI.

Ingests supplier data (clean CSV or the messy JSON sandbox), then runs the
LangGraph agentic workflow over every supplier, printing each agent's
reasoning so you can watch the "thought process".

Usage:
    python run_demo.py                      # uses data/suppliers.csv
    python run_demo.py data/suppliers.csv   # explicit CSV
    python run_demo.py data/suppliers_messy.json
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.ingestion import ingest_csv, ingest_json
from app.logging_config import get_logger
from agents.graph import run_workflow

logger = get_logger("demo")

DEFAULT_SOURCE = Path("data/suppliers.csv")


def load(source: Path):
    if source.suffix.lower() == ".json":
        return ingest_json(source)
    return ingest_csv(source)


def main(argv: list[str]) -> int:
    source = Path(argv[1]) if len(argv) > 1 else DEFAULT_SOURCE
    if not source.exists():
        logger.error("Source not found: %s", source)
        return 1

    print("=" * 78)
    print(f" INGESTING: {source}")
    print("=" * 78)
    result = load(source)
    print(f"\nIngestion summary: {result.summary()}")
    if result.issues:
        print("Issues detected during normalization:")
        for issue in result.issues:
            print(f"  - row {issue.row_index} [{issue.field}]: {issue.message}")

    print("\n" + "=" * 78)
    print(" RUNNING AGENTIC WORKFLOW (ROI -> Decision -> Remediation)")
    print("=" * 78)

    flagged = 0
    for supplier in result.records:
        print("\n" + "-" * 78)
        state = run_workflow(supplier)
        report = state["report"]
        decision = state["decision"]

        print(f"SUPPLIER : {report.supplier_name}")
        print(f"ROI      : {report.roi:.1%}   "
              f"Value Score: {report.value_score:.1f}   "
              f"Status: {report.status.value}")
        print(f"TCO      : {report.total_cost_of_service:,.2f}  "
              f"(base {report.cost:,.2f} + late {report.late_delivery_penalty:,.2f} "
              f"+ defects {report.quality_defect_cost:,.2f})")
        print(f"DECISION : {decision.reason}")

        remediation = state.get("remediation")
        if remediation:
            flagged += 1
            print(f"ACTION   : {remediation.action_type}")
            print("  --- Drafted email ---")
            print(f"  Subject: {remediation.subject}")
            for line in remediation.body.splitlines():
                print(f"  | {line}")

    print("\n" + "=" * 78)
    print(f" SUMMARY: scored {len(result.records)} supplier(s), "
          f"{flagged} flagged for remediation.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
