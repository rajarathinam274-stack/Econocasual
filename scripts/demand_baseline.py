"""Fit the predictive demand baseline and write holdout metrics as JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from econocausal.baseline import fit_demand_baseline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--feature", action="append", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/demand_baseline.json"))
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"Input CSV does not exist: {args.csv}")
    result = fit_demand_baseline(
        pd.read_csv(args.csv), args.outcome, args.feature,
        test_size=args.test_size, random_state=args.seed,
    )
    metrics = {key: value for key, value in result.items() if key != "model"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
