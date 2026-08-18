"""Environment-driven configuration. No secrets are hardcoded."""

from __future__ import annotations

import os
from pathlib import Path

# --- Betfair credentials (set these in your environment, never commit them) ---
BETFAIR_APP_KEY = os.environ.get("BETFAIR_APP_KEY", "")
BETFAIR_USERNAME = os.environ.get("BETFAIR_USERNAME", "")
BETFAIR_PASSWORD = os.environ.get("BETFAIR_PASSWORD", "")

# Certificate login (recommended by Betfair for bots/automated login) is used
# when both of these are set; otherwise the client falls back to interactive
# (non-cert) login.
BETFAIR_CERT_FILE = os.environ.get("BETFAIR_CERT_FILE", "")
BETFAIR_CERT_KEY = os.environ.get("BETFAIR_CERT_KEY", "")

BETFAIR_IDENTITY_INTERACTIVE_URL = "https://identitysso.betfair.com/api/login"
BETFAIR_IDENTITY_CERT_URL = "https://identitysso-cert.betfair.com/api/certlogin"
BETFAIR_BETTING_ENDPOINT = (
    "https://api.betfair.com/exchange/betting/json-rpc/v1"
)

# --- Model artifacts ---
MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "models"))
MODEL_FILE = MODEL_DIR / os.environ.get("MODEL_FILENAME", "outcome_model.keras")
SCALER_FILE = MODEL_DIR / os.environ.get("SCALER_FILENAME", "feature_scaler.pkl")

# --- Model / feature schema ---
N_OUTCOMES = int(os.environ.get("N_OUTCOMES", "3"))  # e.g. home / draw / away
FEATURES_PER_SELECTION = [
    "back_price",
    "lay_price",
    "implied_prob",
    "overround_adj_prob",
    "matched_volume",
    "price_move_5min",
]

# Minimum (model_prob - market_prob) edge to flag a selection as a "value" bet.
VALUE_EDGE_THRESHOLD = float(os.environ.get("VALUE_EDGE_THRESHOLD", "0.02"))


def has_betfair_credentials() -> bool:
    if not (BETFAIR_APP_KEY and BETFAIR_USERNAME and BETFAIR_PASSWORD):
        return False
    return True


def has_cert_login() -> bool:
    return bool(BETFAIR_CERT_FILE and BETFAIR_CERT_KEY)
