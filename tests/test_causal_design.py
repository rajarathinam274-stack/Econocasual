import pandas as pd
import pytest

from econocausal.baseline import fit_demand_baseline
from econocausal.causal_design import assess_causal_design


def test_assessment_reports_support_without_identification_claim():
    data = pd.DataFrame(
        {
            "price": [1.0, 1.5, 2.0, 2.5],
            "demand": [10, 8, 7, 5],
            "season": [1, 1, 2, 2],
        }
    )
    report = assess_causal_design(data, "price", "demand", ["season"])
    assert report["rows"] == 4
    assert report["treatment"]["unique_values"] == 4
    assert report["usable_treatment_outcome_rows"] == 4
    assert report["causal_claim_permitted"] is False
    assert "unobserved confounding" in report["warning"]


def test_binary_treatment_reports_group_counts_and_missingness():
    data = pd.DataFrame(
        {
            "treated": [0, 0, 1, 1, None],
            "outcome": [1, 2, 3, 4, 5],
            "region": ["a", "a", "b", None, "b"],
        }
    )
    report = assess_causal_design(data, "treated", "outcome", ["region"])
    assert report["treatment"]["binary"] is True
    assert report["treatment"]["group_counts"] == {"0.0": 2, "1.0": 2}
    assert report["covariates"]["missing_cells"]["region"] == 1


def test_assessment_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing columns"):
        assess_causal_design(pd.DataFrame({"price": [1]}), "price", "demand")


def test_baseline_is_predictive_only():
    data = pd.DataFrame(
        {
            "demand": [1, 2, 3, 4, 5, 6, 7, 8],
            "price": [8, 7, 6, 5, 4, 3, 2, 1],
            "season": ["a", "a", "b", "b", "a", "a", "b", "b"],
        }
    )
    result = fit_demand_baseline(data, "demand", ["price", "season"], random_state=0)
    assert result["n_train"] + result["n_test"] == len(data)
    assert result["mae"] >= 0
    assert "not a causal" in result["interpretation"]
