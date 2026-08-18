"""Run the Betfair + Keras prediction Flask API.

Usage:
    python run_betting_server.py
"""

from __future__ import annotations

from betting_predictor.app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
