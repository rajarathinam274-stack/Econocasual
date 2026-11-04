import pandas as pd
import pytest

from econocausal.scenario import price_scenarios


def test_price_scenarios_calculates_revenue_under_elasticity_assumption():
    data = pd.DataFrame({"price": [10.0], "quantity": [100.0]})
    result = price_scenarios(data, "price", "quantity", [0.10], [-1.0])
    row = result.iloc[0]
    assert row["baseline_revenue"] == 1000.0
    assert row["scenario_quantity"] == pytest.approx(100 / 1.1)
    assert row["scenario_revenue"] == pytest.approx(1000.0)
    assert "not an estimated" in row["interpretation"]


def test_price_scenarios_returns_full_change_elasticity_grid():
    data = pd.DataFrame({"price": [2.0, 4.0], "quantity": [5.0, 3.0]})
    result = price_scenarios(data, "price", "quantity", [-0.1, 0.1], [-2.0, -1.0])
    assert len(result) == 4
    assert set(result["price_change"]) == {-0.1, 0.1}
    assert set(result["elasticity_assumption"]) == {-2.0, -1.0}


def test_price_scenarios_validates_data_and_changes():
    with pytest.raises(ValueError, match="Missing columns"):
        price_scenarios(pd.DataFrame({"price": [1]}), "price", "quantity", [0.1], [-1])
    with pytest.raises(ValueError, match="greater than -1"):
        price_scenarios(pd.DataFrame({"price": [1], "quantity": [1]}), "price", "quantity", [-1], [-1])
