"""Bootstrap robustness diagnostics for descriptive, non-causal associations."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def bootstrap_correlation(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, Any]:
    """Estimate a Pearson correlation and percentile bootstrap interval.

    This is a descriptive association diagnostic. The interval is not a causal
    effect interval and cannot address unobserved confounding or selection.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    missing = [col for col in (x_col, y_col) if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if x_col == y_col:
        raise ValueError("x_col and y_col must differ")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    frame = data[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 3:
        raise ValueError("At least three complete numeric rows are required")
    if frame[x_col].nunique() < 2 or frame[y_col].nunique() < 2:
        raise ValueError("Both variables must vary")

    values = frame.to_numpy(dtype=float)
    estimate = float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    samples = values[indices]
    x_centered = samples[:, :, 0] - samples[:, :, 0].mean(axis=1, keepdims=True)
    y_centered = samples[:, :, 1] - samples[:, :, 1].mean(axis=1, keepdims=True)
    denominator = np.sqrt((x_centered**2).sum(axis=1) * (y_centered**2).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = (x_centered * y_centered).sum(axis=1) / denominator
    correlations = correlations[np.isfinite(correlations)]
    alpha = 1 - confidence_level
    return {
        "x_column": x_col,
        "y_column": y_col,
        "n_complete": int(len(frame)),
        "estimate": estimate,
        "lower": float(np.quantile(correlations, alpha / 2)),
        "upper": float(np.quantile(correlations, 1 - alpha / 2)),
        "n_resamples": int(len(correlations)),
        "confidence_level": confidence_level,
        "random_state": random_state,
        "interpretation": (
            "Descriptive association only; this bootstrap interval is not a causal effect "
            "interval and does not remove unobserved confounding."
        ),
    }
