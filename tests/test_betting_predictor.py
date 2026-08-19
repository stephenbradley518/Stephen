"""Tests for the Betfair + Keras outcome predictor.

Keras training is real but kept tiny (few synthetic events, 1-2 epochs) so
the suite stays fast. No network calls are made — Betfair-hitting endpoints
are exercised only for their "not configured" error path.
"""

from __future__ import annotations

import numpy as np
import pytest

from betting_predictor import config, features, model as model_lib, synthetic_data
from betting_predictor.app import create_app


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def test_implied_probability():
    assert features.implied_probability(2.0) == pytest.approx(0.5)
    assert features.implied_probability(4.0) == pytest.approx(0.25)


def test_implied_probability_rejects_invalid_price():
    with pytest.raises(ValueError):
        features.implied_probability(1.0)


def test_overround_adjusted_probabilities_sum_to_one():
    implied = [features.implied_probability(p) for p in (2.0, 3.5, 4.0)]
    adjusted = features.overround_adjusted_probabilities(implied)
    assert sum(adjusted) == pytest.approx(1.0)


def test_build_feature_vector_pads_missing_selections():
    rows = [
        {"selection_id": 0, "back_price": 2.0, "lay_price": 2.02, "matched_volume": 100.0, "price_move_5min": 0.01},
    ]
    vector = features.build_feature_vector(rows, n_outcomes=3)
    n_feat = len(config.FEATURES_PER_SELECTION)
    assert len(vector) == 3 * n_feat
    # Second and third selections are zero-padded.
    assert vector[n_feat : 2 * n_feat] == [0.0] * n_feat
    assert vector[2 * n_feat : 3 * n_feat] == [0.0] * n_feat


def test_market_book_to_selection_rows_skips_illiquid_runners():
    market_book = {
        "runners": [
            {
                "selectionId": 1,
                "ex": {"availableToBack": [{"price": 2.0}], "availableToLay": [{"price": 2.02}]},
                "totalMatched": 500.0,
            },
            {
                "selectionId": 2,
                "ex": {"availableToBack": [], "availableToLay": [{"price": 5.0}]},
                "totalMatched": 0.0,
            },
        ]
    }
    rows = features.market_book_to_selection_rows(market_book)
    assert len(rows) == 1
    assert rows[0]["selection_id"] == 1


# ---------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------

def test_generate_training_set_shapes():
    X, y = synthetic_data.generate_training_set(n_events=32, n_outcomes=3, seed=1)
    n_feat = len(config.FEATURES_PER_SELECTION)
    assert X.shape == (32, 3 * n_feat)
    assert y.shape == (32, 3)
    # One-hot rows.
    assert np.all(y.sum(axis=1) == 1.0)


def test_generate_training_set_is_deterministic_with_seed():
    X1, y1 = synthetic_data.generate_training_set(n_events=16, seed=7)
    X2, y2 = synthetic_data.generate_training_set(n_events=16, seed=7)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_trained_model():
    X, y = synthetic_data.generate_training_set(n_events=64, n_outcomes=3, seed=3)
    result = model_lib.train(X, y, n_outcomes=3, epochs=2, batch_size=16, verbose=0)
    return result


def test_build_model_output_shape():
    m = model_lib.build_model(input_dim=18, n_outcomes=3)
    dummy = np.zeros((4, 18), dtype="float32")
    out = m.predict(dummy, verbose=0)
    assert out.shape == (4, 3)
    # Softmax outputs sum to 1 per row.
    np.testing.assert_allclose(out.sum(axis=1), np.ones(4), atol=1e-5)


def test_train_produces_reasonable_metrics(tiny_trained_model):
    assert 0.0 <= tiny_trained_model.val_accuracy <= 1.0
    assert tiny_trained_model.val_log_loss > 0.0


def test_save_and_load_roundtrip(tiny_trained_model, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(config, "MODEL_FILE", tmp_path / "m.keras")
    monkeypatch.setattr(config, "SCALER_FILE", tmp_path / "s.pkl")

    model_lib.save(tiny_trained_model.model, tiny_trained_model.scaler)
    loaded_model, loaded_scaler = model_lib.load()

    X = np.random.default_rng(0).normal(size=(2, 18)).astype("float32")
    p1 = model_lib.predict_proba(tiny_trained_model.model, tiny_trained_model.scaler, X)
    p2 = model_lib.predict_proba(loaded_model, loaded_scaler, X)
    np.testing.assert_allclose(p1, p2, atol=1e-5)


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    app = create_app()
    app.testing = True
    return app.test_client()


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert "betfair_credentials_configured" in body


def test_betfair_endpoints_report_not_configured_without_credentials(client, monkeypatch):
    monkeypatch.setattr(config, "BETFAIR_APP_KEY", "")
    monkeypatch.setattr(config, "BETFAIR_USERNAME", "")
    monkeypatch.setattr(config, "BETFAIR_PASSWORD", "")
    resp = client.get("/betfair/markets?event_type_id=1")
    assert resp.status_code == 501


def test_predict_with_raw_selections(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(config, "MODEL_FILE", tmp_path / "m.keras")
    monkeypatch.setattr(config, "SCALER_FILE", tmp_path / "s.pkl")

    X, y = synthetic_data.generate_training_set(n_events=48, n_outcomes=3, seed=5)
    result = model_lib.train(X, y, n_outcomes=3, epochs=2, batch_size=16)
    model_lib.save(result.model, result.scaler)

    import betting_predictor.app as app_module

    monkeypatch.setattr(app_module, "_model_cache", None)

    resp = client.post(
        "/predict",
        json={
            "selections": [
                {"back_price": 2.0, "lay_price": 2.02, "matched_volume": 1000, "price_move_5min": 0.0},
                {"back_price": 3.5, "lay_price": 3.6, "matched_volume": 500, "price_move_5min": 0.0},
                {"back_price": 4.2, "lay_price": 4.3, "matched_volume": 300, "price_move_5min": 0.0},
            ]
        },
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert len(body["selections"]) == 3
    probs = [s["model_prob"] for s in body["selections"]]
    assert sum(probs) == pytest.approx(1.0, abs=1e-4)


def test_predict_requires_market_id_or_selections(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(config, "MODEL_FILE", tmp_path / "missing.keras")
    monkeypatch.setattr(config, "SCALER_FILE", tmp_path / "missing.pkl")
    resp = client.post("/predict", json={})
    assert resp.status_code == 400
