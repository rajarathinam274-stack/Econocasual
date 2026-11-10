import json
from pathlib import Path

import pandas as pd

from scripts.demand_baseline import main


def test_demand_baseline_cli_writes_metrics_without_model(tmp_path, monkeypatch):
    source = tmp_path / "demand.csv"
    pd.DataFrame(
        {
            "demand": [1, 2, 3, 4, 5, 6, 7, 8],
            "price": [8, 7, 6, 5, 4, 3, 2, 1],
            "season": ["a", "a", "b", "b", "a", "a", "b", "b"],
        }
    ).to_csv(source, index=False)
    output = tmp_path / "baseline.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "demand_baseline.py", str(source), "--outcome", "demand",
            "--feature", "price", "--feature", "season",
            "--seed", "0", "--output", str(output),
        ],
    )

    assert main() == 0
    metrics = json.loads(output.read_text(encoding="utf-8"))
    assert metrics["n_train"] + metrics["n_test"] == 8
    assert metrics["mae"] >= 0
    assert "not a causal" in metrics["interpretation"]
    assert "model" not in metrics
