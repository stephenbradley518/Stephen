"""Turn raw Betfair market data into the numeric feature vectors the Keras
model consumes.

Betfair prices are decimal odds. The exchange's "overround" (the sum of
implied probabilities across all selections being > 1) reflects the market's
built-in margin, so we compute both the raw implied probability and an
overround-adjusted ("de-vigged") probability — the model is trained to spot
edges relative to the de-vigged market, which is the closest thing to a fair
value baseline the market itself offers.
"""

from __future__ import annotations

from . import config


def implied_probability(decimal_price: float) -> float:
    if decimal_price <= 1.0:
        raise ValueError("decimal_price must be > 1.0")
    return 1.0 / decimal_price


def overround_adjusted_probabilities(implied_probs: list[float]) -> list[float]:
    """Rescale implied probabilities so they sum to 1 (remove the overround)."""
    total = sum(implied_probs)
    if total <= 0:
        raise ValueError("implied_probs must sum to a positive number")
    return [p / total for p in implied_probs]


def selection_features(
    back_price: float,
    lay_price: float,
    implied_prob: float,
    overround_adj_prob: float,
    matched_volume: float,
    price_move_5min: float,
) -> list[float]:
    """Order MUST match config.FEATURES_PER_SELECTION."""
    return [
        back_price,
        lay_price,
        implied_prob,
        overround_adj_prob,
        matched_volume,
        price_move_5min,
    ]


def market_book_to_selection_rows(
    market_book: dict, price_history: dict[int, float] | None = None
) -> list[dict]:
    """Extract one row of raw values per runner/selection from a Betfair
    `listMarketBook` result entry.

    `price_history` optionally maps selectionId -> best-back price from ~5
    minutes ago, used to compute price_move_5min (market drift signal). If
    omitted, price movement is reported as 0.0 (no signal).
    """
    price_history = price_history or {}
    rows = []
    for runner in market_book.get("runners", []):
        selection_id = runner["selectionId"]
        ex = runner.get("ex", {})
        backs = ex.get("availableToBack", [])
        lays = ex.get("availableToLay", [])
        best_back = backs[0]["price"] if backs else None
        best_lay = lays[0]["price"] if lays else None
        matched = runner.get("totalMatched", 0.0) or 0.0

        if best_back is None or best_lay is None:
            # No liquidity on one side — skip, can't derive a fair mid price.
            continue

        prev_price = price_history.get(selection_id, best_back)
        rows.append(
            {
                "selection_id": selection_id,
                "back_price": best_back,
                "lay_price": best_lay,
                "matched_volume": matched,
                "price_move_5min": best_back - prev_price,
            }
        )
    return rows


def build_feature_vector(selection_rows: list[dict], n_outcomes: int = config.N_OUTCOMES) -> list[float]:
    """Build the flat feature vector for one market (event) from its
    per-selection raw rows. Pads/truncates to `n_outcomes` selections so the
    model always sees a fixed-size input."""
    implied = [implied_probability(r["back_price"]) for r in selection_rows]
    adjusted = overround_adjusted_probabilities(implied) if implied else []

    vector: list[float] = []
    for i in range(n_outcomes):
        if i < len(selection_rows):
            r = selection_rows[i]
            vector.extend(
                selection_features(
                    back_price=r["back_price"],
                    lay_price=r["lay_price"],
                    implied_prob=implied[i],
                    overround_adj_prob=adjusted[i],
                    matched_volume=r["matched_volume"],
                    price_move_5min=r["price_move_5min"],
                )
            )
        else:
            vector.extend([0.0] * len(config.FEATURES_PER_SELECTION))
    return vector


def feature_names(n_outcomes: int = config.N_OUTCOMES) -> list[str]:
    names = []
    for i in range(n_outcomes):
        names.extend(f"sel{i}_{f}" for f in config.FEATURES_PER_SELECTION)
    return names
