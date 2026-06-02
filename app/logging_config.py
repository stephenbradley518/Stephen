"""Centralised logging configuration.

A single helper so every layer (ingestion, ROI engine, agents, API) emits
logs in a consistent, human-readable format. The agents deliberately log
their *reasoning* at ``INFO`` level so you can watch the "thought process"
of the orchestration loop on the console.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(level: str | int | None = None) -> None:
    """Configure root logging once, idempotently.

    The level can be overridden via the ``SOA_LOG_LEVEL`` environment
    variable (e.g. ``DEBUG``); defaults to ``INFO``.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = level or os.getenv("SOA_LOG_LEVEL", "INFO")
    logging.basicConfig(level=resolved, format=_DEFAULT_FORMAT, datefmt=_DATEFMT)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for *name* (configures logging on first use)."""
    configure_logging()
    return logging.getLogger(name)
