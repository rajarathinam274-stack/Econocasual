import pandas as pd
import pytest
from econocausal.validation import validate_transactions
from econocausal.bias_audit import data_quality_audit, causal_readiness_checks, subgroup_error_audit


def test_validation_removes_duplicates_invalid_sales_and_returns():
    df = pd.DataFrame({"Quantity": [2, 2, -1, 0, 3], "UnitPrice": [5, 5, 4, 3, 0]})
    clean, report = validate_transactions(df)
    assert len(clean) == 1
    assert report.rows_input == 5
    assert report.duplicate_rows == 1
    assert report.cancellation_rows == 1


def test_validation_requires_columns():
    with pytest.raises(ValueError):
        validate_transactions(pd.DataFrame({"quantity": [1]}))


def test_quality_audit_handles_empty_frame():
    result = data_quality_audit(pd.DataFrame())
    assert result["rows"] == 0
    assert result["missing_cell_rate"] == 0.0


def test_causal_readiness_never_claims_identification():
    df = pd.DataFrame({"price": [1, 2, 3], "demand": [4, 3, 2], "season": [1, 1, 2]})
    result = causal_readiness_checks(df, "price", "demand", ["season"])
    assert result["treatment_unique_values"] == 3
    assert result["causal_claim_permitted"] is False


def test_subgroup_error_audit():
    df = pd.DataFrame({"y": [2, 4, 5], "pred": [1, 4, 7], "group": ["a", "a", "b"]})
    result = subgroup_error_audit(df, "y", "pred", "group")
    assert set(result["group"]) == {"a", "b"}
    assert "mae" in result.columns
