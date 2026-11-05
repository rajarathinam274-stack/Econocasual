"""Run the local EconoCausal pipeline and write a reproducibility manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from econocausal.eda import plot_eda  # noqa: E402
from econocausal.preprocessing import preprocess_transactions  # noqa: E402
from econocausal.robustness import bootstrap_correlation  # noqa: E402
from econocausal.scenario import price_scenarios  # noqa: E402


def run_pipeline(csv_path: Path, output_dir: Path) -> dict[str, object]:
    """Run descriptive pipeline components and persist their outputs."""
    raw = pd.read_csv(csv_path)
    transactions = preprocess_transactions(raw)
    eda = plot_eda(transactions, output_dir / "eda", top_n=10)
    sensitivity = bootstrap_correlation(
        transactions, "unit_price", "quantity", n_resamples=200, random_state=42
    )
    scenarios = price_scenarios(
        transactions, "unit_price", "quantity", [-0.10, 0.10], [-2.0, -1.0]
    )
    scenarios.to_csv(output_dir / "price_scenarios.csv", index=False)
    manifest = {
        "input_csv": str(csv_path),
        "output_dir": str(output_dir),
        "raw_rows": int(len(raw)),
        "positive_sales_rows": int(len(transactions)),
        "eda_revenue": float(eda["positive_sales_revenue"]),
        "association_sensitivity": sensitivity,
        "scenario_rows": int(len(scenarios)),
        "scientific_note": (
            "All outputs are descriptive or assumption-based. They do not identify a causal price effect."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reproducibility"))
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"Input CSV does not exist: {args.csv}")
    print(json.dumps(run_pipeline(args.csv, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
