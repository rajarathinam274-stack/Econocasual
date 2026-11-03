import pandas as pd
import pytest

from econocausal.robustness import bootstrap_correlation


def test_bootstrap_correlation_is_reproducible_and_non_causal():
    data = pd.DataFrame({"price": [1, 2, 3, 4, 5], "demand": [10, 8, 7, 5, 4]})
    first = bootstrap_correlation(data, "price", "demand", n_resamples=200, random_state=7)
    second = bootstrap_correlation(data, "price", "demand", n_resamples=200, random_state=7)
    assert first == second
    assert first["n_complete"] == 5
    assert first["lower"] <= first["estimate"] <= first["upper"]
    assert "not a causal effect" in first["interpretation"]


def test_bootstrap_correlation_coerces_and_drops_missing_values():
    data = pd.DataFrame({"x": [1, 2, "bad", 4, 5], "y": [1, 3, 2, 4, None]})
    result = bootstrap_correlation(data, "x", "y", n_resamples=25)
    assert result["n_complete"] == 3
    assert result["n_resamples"] <= 25


def test_bootstrap_correlation_validates_input():
    with pytest.raises(ValueError, match="At least three"):
        bootstrap_correlation(pd.DataFrame({"x": [1, 2], "y": [2, 3]}), "x", "y")
    with pytest.raises(ValueError, match="Missing columns"):
        bootstrap_correlation(pd.DataFrame({"x": [1, 2, 3]}), "x", "y")
