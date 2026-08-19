"""Synthetic market + outcome generator.

Training a real model requires either a live Betfair account streaming
historical market data, or Betfair's paid historical data service — neither
is available in this environment. This module generates a synthetic but
*structured* dataset that mimics real exchange dynamics closely enough to
exercise the whole pipeline end-to-end:

  1. Each event gets a latent "true" win-probability vector (Dirichlet).
  2. The market's displayed prices are the true probabilities distorted by
     bookmaker overround + noise + occasional mispricing — so the model has
     a genuine (if synthetic) edge to learn, not just a de-vig calculator.
  3. The actual outcome is sampled from the *true* probabilities.

Swap this module out for a real historical-data loader once you have access
to one; `train.py` and the model code don't care where `X, y` come from.
"""

from __future__ import annotations

import numpy as np

from . import config, features


def _simulate_market_for_event(
    true_probs: np.ndarray, overround: float, rng: np.random.Generator
) -> list[dict]:
    n = len(true_probs)
    # Market's perceived probabilities: true probs + mispricing noise,
    # renormalized, then inflated by the overround (bookmaker margin).
    noise = rng.normal(loc=0.0, scale=0.04, size=n)
    perceived = np.clip(true_probs + noise, 0.01, None)
    perceived = perceived / perceived.sum()
    # Sums to `overround`, not 1.0 — clipped so no selection implies a
    # probability >= 1.0 (which would mean a decimal price <= 1.0).
    market_probs = np.clip(perceived * overround, 1e-3, 0.99)

    rows = []
    for i in range(n):
        implied_p = market_probs[i]
        back_price = 1.0 / implied_p
        spread = back_price * rng.uniform(0.005, 0.03)
        lay_price = back_price + spread
        matched_volume = float(rng.lognormal(mean=8.0, sigma=1.2))
        # Simulate informed money nudging the price toward the true value
        # over the last 5 minutes.
        true_price = 1.0 / max(true_probs[i], 1e-6)
        drift = (true_price - back_price) * rng.uniform(0.0, 0.15)
        price_move_5min = drift + rng.normal(0.0, back_price * 0.01)
        rows.append(
            {
                "selection_id": i,
                "back_price": back_price,
                "lay_price": lay_price,
                "matched_volume": matched_volume,
                "price_move_5min": price_move_5min,
            }
        )
    return rows


def generate_training_set(
    n_events: int,
    n_outcomes: int = config.N_OUTCOMES,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, y): X is (n_events, n_outcomes * n_features_per_selection)
    float32, y is (n_events, n_outcomes) one-hot float32."""
    rng = np.random.default_rng(seed)

    X = np.zeros((n_events, n_outcomes * len(config.FEATURES_PER_SELECTION)), dtype="float32")
    y = np.zeros((n_events, n_outcomes), dtype="float32")

    for i in range(n_events):
        # Dirichlet alpha < 1 skews toward one clear favorite sometimes,
        # alpha > 1 toward closer contests — mix both for realism.
        alpha = rng.choice([0.6, 1.0, 2.5], size=n_outcomes)
        true_probs = rng.dirichlet(alpha)
        overround = rng.uniform(1.03, 1.12)  # typical exchange-ish margin

        rows = _simulate_market_for_event(true_probs, overround, rng)
        X[i] = features.build_feature_vector(rows, n_outcomes=n_outcomes)

        outcome = rng.choice(n_outcomes, p=true_probs)
        y[i, outcome] = 1.0

    return X, y
