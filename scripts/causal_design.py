"""Write conservative causal-design diagnostics for a local CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from econocausal.causal_design import assess_causal_design  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Input CSV")
    parser.add_argument("--treatment", required=True, help="Treatment column")
    parser.add_argument("--outcome", required=True, help="Outcome column")
    parser.add_argument("--covariate", action="append", default=[], help="Observed covariate column; repeatable")
    parser.add_argument("--output", type=Path, default=Path("outputs/causal_design.json"))
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"Input CSV does not exist: {args.csv}")

    report = assess_causal_design(
        pd.read_csv(args.csv), args.treatment, args.outcome, args.covariate
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
