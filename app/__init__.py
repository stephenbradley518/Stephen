"""Autonomous Supplier Optimization Agent — application package.

The :mod:`app` package holds the deterministic, business-logic core:

* :mod:`app.models`      – Pydantic schemas (``SupplierData``, ``ROIReport`` ...).
* :mod:`app.roi_engine`  – the ROI / Supplier-Value-Score calculation engine.
* :mod:`app.ingestion`   – flexible CSV/JSON ingestion + normalization (Pandas).
* :mod:`app.main`        – the FastAPI application layer.

The agentic orchestration lives in the sibling :mod:`agents` package.
"""

__version__ = "0.1.0"
