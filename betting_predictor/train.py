"""Train the outcome-prediction model on synthetic data and save it.

Usage:
    python -m betting_predictor.train
    python -m betting_predictor.train --events 50000 --epochs 40
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import log_loss

from . import config, model as model_lib, synthetic_data


def _market_baseline_log_loss(X: np.ndarray, y: np.ndarray, n_outcomes: int) -> float:
    """Log loss of just using the market's own overround-adjusted implied
    probabilities as the prediction — the bar the model needs to clear."""
    n_features = len(config.FEATURES_PER_SELECTION)
    adj_prob_idx = config.FEATURES_PER_SELECTION.index("overround_adj_prob")
    cols = [i * n_features + adj_prob_idx for i in range(n_outcomes)]
    market_probs = X[:, cols]
    market_probs = market_probs / market_probs.sum(axis=1, keepdims=True)
    return float(log_loss(y, market_probs))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=20000, help="Synthetic training events")
    parser.add_argument("--outcomes", type=int, default=config.N_OUTCOMES)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Generating {args.events} synthetic events ({args.outcomes} outcomes each)...")
    X, y = synthetic_data.generate_training_set(
        n_events=args.events, n_outcomes=args.outcomes, seed=args.seed
    )

    print("Training Keras model...")
    result = model_lib.train(
        X,
        y,
        n_outcomes=args.outcomes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1 if args.verbose else 0,
        seed=args.seed,
    )

    baseline_loss = _market_baseline_log_loss(X, y, args.outcomes)

    print("\n" + "=" * 60)
    print(f" Validation accuracy : {result.val_accuracy:.4f}")
    print(f" Validation log loss : {result.val_log_loss:.4f}")
    print(f" Market baseline loss: {baseline_loss:.4f}  (de-vigged market probs)")
    edge = baseline_loss - result.val_log_loss
    print(f" Model edge vs market: {edge:+.4f} (lower log loss is better)")
    print("=" * 60)

    model_lib.save(result.model, result.scaler)
    print(f"\nSaved model to {config.MODEL_FILE}")
    print(f"Saved scaler to {config.SCALER_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
