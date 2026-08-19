"""Flask app: Betfair market data + Keras outcome prediction, exposed as a
small JSON API.

Run:
    python -m betting_predictor.app
    # or: flask --app betting_predictor.app run
"""

from __future__ import annotations

import numpy as np
from flask import Flask, jsonify, request

from . import config, features, model as model_lib, synthetic_data
from .betfair_client import BetfairAPIError, BetfairClient

_betfair_client: BetfairClient | None = None
_model_cache: tuple | None = None  # (keras.Model, StandardScaler)


def get_betfair_client() -> BetfairClient:
    global _betfair_client
    if _betfair_client is None:
        _betfair_client = BetfairClient.from_env()
    return _betfair_client


def get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = model_lib.load()
    return _model_cache


def _predict_from_selection_rows(selection_rows: list[dict]) -> dict:
    model, scaler = get_model()
    vector = features.build_feature_vector(selection_rows, n_outcomes=config.N_OUTCOMES)
    X = np.array([vector], dtype="float32")
    probs = model_lib.predict_proba(model, scaler, X)[0]

    implied = [features.implied_probability(r["back_price"]) for r in selection_rows]
    market_probs = features.overround_adjusted_probabilities(implied) if implied else []

    selections_out = []
    for i, row in enumerate(selection_rows):
        model_p = float(probs[i]) if i < len(probs) else None
        market_p = market_probs[i] if i < len(market_probs) else None
        edge = (model_p - market_p) if (model_p is not None and market_p is not None) else None
        selections_out.append(
            {
                "selection_id": row.get("selection_id", i),
                "back_price": row["back_price"],
                "market_implied_prob": market_p,
                "model_prob": model_p,
                "edge": edge,
                "value_bet": bool(edge is not None and edge > config.VALUE_EDGE_THRESHOLD),
            }
        )
    return {"selections": selections_out}


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "betfair_credentials_configured": config.has_betfair_credentials(),
                "model_trained": config.MODEL_FILE.exists(),
            }
        )

    @app.get("/betfair/event-types")
    def event_types():
        try:
            client = get_betfair_client()
            return jsonify(client.list_event_types())
        except BetfairAPIError as exc:
            return jsonify({"error": str(exc)}), 502
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 501

    @app.get("/betfair/markets")
    def markets():
        event_type_id = request.args.get("event_type_id")
        market_type = request.args.get("market_type", "MATCH_ODDS")
        max_results = int(request.args.get("max_results", 20))
        if not event_type_id:
            return jsonify({"error": "event_type_id query param is required"}), 400
        try:
            client = get_betfair_client()
            catalogue = client.list_market_catalogue(
                filter_={
                    "eventTypeIds": [event_type_id],
                    "marketTypeCodes": [market_type],
                },
                max_results=max_results,
            )
            return jsonify(catalogue)
        except BetfairAPIError as exc:
            return jsonify({"error": str(exc)}), 502
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 501

    @app.get("/betfair/market/<market_id>/prices")
    def market_prices(market_id: str):
        try:
            client = get_betfair_client()
            books = client.list_market_book([market_id])
            return jsonify(books)
        except BetfairAPIError as exc:
            return jsonify({"error": str(exc)}), 502
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 501

    @app.post("/predict")
    def predict():
        """Predict outcome probabilities and flag value bets.

        Body options:
          {"market_id": "1.234567890"}                     -> fetches live Betfair prices
          {"selections": [{"back_price": 2.5, "lay_price": 2.6, ...}, ...]}
        """
        body = request.get_json(silent=True) or {}

        try:
            if "market_id" in body:
                client = get_betfair_client()
                books = client.list_market_book([body["market_id"]])
                if not books:
                    return jsonify({"error": "market not found"}), 404
                selection_rows = features.market_book_to_selection_rows(books[0])
            elif "selections" in body:
                selection_rows = []
                for i, sel in enumerate(body["selections"]):
                    selection_rows.append(
                        {
                            "selection_id": sel.get("selection_id", i),
                            "back_price": float(sel["back_price"]),
                            "lay_price": float(sel.get("lay_price", sel["back_price"])),
                            "matched_volume": float(sel.get("matched_volume", 0.0)),
                            "price_move_5min": float(sel.get("price_move_5min", 0.0)),
                        }
                    )
            else:
                return jsonify({"error": "provide 'market_id' or 'selections'"}), 400

            result = _predict_from_selection_rows(selection_rows)
            return jsonify(result)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 503
        except BetfairAPIError as exc:
            return jsonify({"error": str(exc)}), 502
        except (KeyError, ValueError) as exc:
            return jsonify({"error": f"invalid request: {exc}"}), 400

    @app.post("/train")
    def train_endpoint():
        """Retrain on freshly generated synthetic data (demo/dev use)."""
        body = request.get_json(silent=True) or {}
        n_events = int(body.get("events", 5000))
        epochs = int(body.get("epochs", 15))

        X, y = synthetic_data.generate_training_set(n_events=n_events, n_outcomes=config.N_OUTCOMES)
        result = model_lib.train(X, y, n_outcomes=config.N_OUTCOMES, epochs=epochs)
        model_lib.save(result.model, result.scaler)

        global _model_cache
        _model_cache = (result.model, result.scaler)

        return jsonify(
            {
                "events": n_events,
                "epochs": epochs,
                "val_accuracy": result.val_accuracy,
                "val_log_loss": result.val_log_loss,
            }
        )

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
