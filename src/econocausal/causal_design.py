"""Conservative diagnostics for whether observed data support a causal design."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def assess_causal_design(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: Sequence[str] = (),
) -> dict[str, Any]:
    """Report observable design diagnostics without asserting identification.

    The report is intentionally descriptive. It cannot establish exchangeability,
    valid treatment assignment, no interference, or a causal effect.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    covariates = list(covariate_cols)
    required = [treatment_col, outcome_col, *covariates]
    missing = list(dict.fromkeys(col for col in required if col not in data.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if treatment_col == outcome_col:
        raise ValueError("treatment_col and outcome_col must differ")
    if len(set(covariates)) != len(covariates):
        raise ValueError("covariate_cols must not contain duplicates")

    frame = data[required].copy()
    treatment = pd.to_numeric(frame[treatment_col], errors="coerce")
    outcome = pd.to_numeric(frame[outcome_col], errors="coerce")
    finite_treatment = treatment[np.isfinite(treatment)]
    finite_outcome = outcome[np.isfinite(outcome)]
    treatment_values = sorted(float(value) for value in finite_treatment.unique())
    outcome_values = sorted(float(value) for value in finite_outcome.unique())
    binary = len(treatment_values) == 2
    group_counts = {
        str(value): int((treatment == value).sum()) for value in treatment_values
    }
    usable = frame.loc[treatment.notna() & outcome.notna()].copy()
    covariate_missing = {
        col: int(frame[col].isna().sum()) for col in covariates
    }
    complete_case_rows = int(frame.dropna().shape[0])

    return {
        "rows": int(len(frame)),
        "complete_case_rows": complete_case_rows,
        "treatment": {
            "column": treatment_col,
            "finite_rows": int(len(finite_treatment)),
            "unique_values": len(treatment_values),
            "values": treatment_values,
            "binary": binary,
            "group_counts": group_counts,
        },
        "outcome": {
            "column": outcome_col,
            "finite_rows": int(len(finite_outcome)),
            "unique_values": len(outcome_values),
        },
        "covariates": {
            "columns": covariates,
            "missing_cells": covariate_missing,
        },
        "usable_treatment_outcome_rows": int(len(usable)),
        "common_support_check": {
            "status": "descriptive_only",
            "note": "Treatment support is reported; this does not establish overlap or positivity.",
        },
        "causal_claim_permitted": False,
        "warning": (
            "Observed-data checks do not establish exchangeability, valid treatment assignment, "
            "no interference, or a causal effect. DML cannot remove unobserved confounding."
        ),
    }
