"""Compute a descriptive bootstrap correlation interval for two CSV columns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from econocausal.robustness import bootstrap_correlation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--x", required=True, help="Numeric explanatory/association column")
    parser.add_argument("--y", required=True, help="Numeric outcome/association column")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"Input CSV does not exist: {args.csv}")
    result = bootstrap_correlation(
        pd.read_csv(args.csv), args.x, args.y,
        n_resamples=args.resamples, random_state=args.seed,
    )
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
