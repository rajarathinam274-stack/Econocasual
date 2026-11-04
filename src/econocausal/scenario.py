"""Pricing scenario analysis under explicit, non-causal elasticity assumptions."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def price_scenarios(
    data: pd.DataFrame,
    price_col: str,
    quantity_col: str,
    price_changes: Sequence[float],
    elasticities: Sequence[float],
) -> pd.DataFrame:
    """Calculate revenue scenarios using user-supplied constant elasticities.

    ``price_changes`` are fractional changes, so ``0.1`` means a 10% increase.
    The elasticity is an assumption supplied by the caller; this function does
    not estimate it and therefore does not make a causal claim.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    missing = [col for col in (price_col, quantity_col) if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    changes = [float(value) for value in price_changes]
    elasticities = [float(value) for value in elasticities]
    if not changes or not elasticities:
        raise ValueError("price_changes and elasticities must not be empty")
    if any(change <= -1 for change in changes):
        raise ValueError("price changes must be greater than -1")

    frame = data[[price_col, quantity_col]].apply(pd.to_numeric, errors="coerce").dropna()
    frame = frame[(frame[price_col] > 0) & (frame[quantity_col] >= 0)].copy()
    if frame.empty:
        raise ValueError("At least one row with positive price and nonnegative quantity is required")
    baseline_revenue = float((frame[price_col] * frame[quantity_col]).sum())
    rows: list[dict[str, Any]] = []
    for change in changes:
        new_price_factor = 1 + change
        for elasticity in elasticities:
            quantity_factor = new_price_factor ** elasticity
            scenario_quantity = float((frame[quantity_col] * quantity_factor).sum())
            scenario_revenue = float((frame[price_col] * new_price_factor * frame[quantity_col] * quantity_factor).sum())
            rows.append(
                {
                    "price_change": change,
                    "elasticity_assumption": elasticity,
                    "baseline_revenue": baseline_revenue,
                    "scenario_quantity": scenario_quantity,
                    "scenario_revenue": scenario_revenue,
                    "revenue_change": scenario_revenue - baseline_revenue,
                    "interpretation": (
                        "Scenario arithmetic under a supplied elasticity assumption; "
                        "not an estimated or identified causal effect."
                    ),
                }
            )
    return pd.DataFrame(rows)
