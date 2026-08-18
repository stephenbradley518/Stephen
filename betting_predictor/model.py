"""Keras deep learning model for sporting-event outcome prediction.

A feedforward network over engineered market features (odds, overround-
adjusted implied probability, matched volume, short-term price drift). The
output is a softmax over `n_outcomes` classes (e.g. home/draw/away), trained
with categorical cross-entropy so predictions are calibrated probabilities
rather than plain classification labels — required to compare against the
market's own probabilities and detect value bets.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from . import config


def build_model(input_dim: int, n_outcomes: int = config.N_OUTCOMES) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(n_outcomes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


@dataclass
class TrainResult:
    model: keras.Model
    scaler: StandardScaler
    history: dict
    val_accuracy: float
    val_log_loss: float


def train(
    X: np.ndarray,
    y: np.ndarray,
    n_outcomes: int = config.N_OUTCOMES,
    epochs: int = 30,
    batch_size: int = 64,
    validation_split: float = 0.2,
    verbose: int = 0,
    seed: int = 42,
) -> TrainResult:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = build_model(input_dim=X.shape[1], n_outcomes=n_outcomes)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose,
    )

    val_loss, val_acc = model.evaluate(X_val_scaled, y_val, verbose=0)
    return TrainResult(
        model=model,
        scaler=scaler,
        history=history.history,
        val_accuracy=float(val_acc),
        val_log_loss=float(val_loss),
    )


def predict_proba(model: keras.Model, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled, verbose=0)


def save(model: keras.Model, scaler: StandardScaler) -> None:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(config.MODEL_FILE)
    with open(config.SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)


def load() -> tuple[keras.Model, StandardScaler]:
    if not config.MODEL_FILE.exists() or not config.SCALER_FILE.exists():
        raise FileNotFoundError(
            f"No trained model found at {config.MODEL_FILE}. Run training first "
            "(python -m betting_predictor.train)."
        )
    model = keras.models.load_model(config.MODEL_FILE)
    with open(config.SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler
