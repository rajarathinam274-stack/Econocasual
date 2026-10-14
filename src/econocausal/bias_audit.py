"""Descriptive bias diagnostics; these do not establish causal identification."""
from __future__ import annotations
import pandas as pd


def data_quality_audit(data: pd.DataFrame) -> dict[str, object]:
    """Return missingness, duplication, and cardinality diagnostics."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    total = int(data.size)
    missing = int(data.isna().sum().sum())
    return {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_cells": missing,
        "missing_cell_rate": float(missing / total) if total else 0.0,
        "unique_values_by_column": {str(c): int(data[c].nunique(dropna=True)) for c in data.columns},
    }


def subgroup_error_audit(data: pd.DataFrame, actual_col: str, predicted_col: str,
                         group_col: str) -> pd.DataFrame:
    """Compare MAE, mean residual, and sample size across observed groups."""
    required = {actual_col, predicted_col, group_col}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = data[[actual_col, predicted_col, group_col]].dropna().copy()
    frame["residual"] = frame[actual_col] - frame[predicted_col]
    frame["absolute_error"] = frame["residual"].abs()
    result = frame.groupby(group_col, dropna=False).agg(
        n=("residual", "size"), mae=("absolute_error", "mean"), mean_residual=("residual", "mean")
    ).reset_index()
    return result.sort_values("n", ascending=False).reset_index(drop=True)


def causal_readiness_checks(data: pd.DataFrame, treatment_col: str,
                            outcome_col: str, covariates: list[str]) -> dict[str, object]:
    """Basic feasibility checks only; cannot prove absence of unobserved confounding."""
    required = [treatment_col, outcome_col, *covariates]
    absent = [c for c in required if c not in data.columns]
    if absent:
        raise ValueError(f"Missing columns: {absent}")
    usable = data[required].dropna()
    treatment = pd.to_numeric(usable[treatment_col], errors="coerce").dropna()
    return {
        "rows_total": int(len(data)),
        "rows_complete_case": int(len(usable)),
        "complete_case_rate": float(len(usable) / len(data)) if len(data) else 0.0,
        "treatment_unique_values": int(treatment.nunique()),
        "treatment_std": float(treatment.std()) if len(treatment) > 1 else 0.0,
        "causal_claim_permitted": False,
        "warning": "These checks do not establish exchangeability, overlap, or valid causal identification.",
    }
