import json
from pathlib import Path

import pandas as pd

from scripts.reproducibility_smoke import run_pipeline


def test_reproducibility_pipeline_writes_manifest_and_artifacts(tmp_path):
    source = tmp_path / "transactions.csv"
    pd.DataFrame(
        {
            "InvoiceNo": ["1", "2", "3", "4", "5"],
            "Quantity": [2, 1, 3, 2, -1],
            "InvoiceDate": ["2011-12-01", "2011-12-02", "2011-12-03", "2011-12-04", "2011-12-05"],
            "UnitPrice": [5, 3, 4, 2, 4],
        }
    ).to_csv(source, index=False)

    manifest = run_pipeline(source, tmp_path / "outputs")

    assert manifest["raw_rows"] == 5
    assert manifest["positive_sales_rows"] == 4
    assert manifest["scenario_rows"] == 4
    assert "do not identify a causal" in manifest["scientific_note"]
    assert (tmp_path / "outputs/manifest.json").is_file()
    assert (tmp_path / "outputs/price_scenarios.csv").is_file()
    assert (tmp_path / "outputs/eda/eda_summary.json").is_file()
    assert json.loads((tmp_path / "outputs/manifest.json").read_text())["eda_revenue"] == 29.0
