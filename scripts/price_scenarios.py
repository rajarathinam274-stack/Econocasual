"""Run non-causal pricing scenarios from a CSV using explicit elasticity assumptions."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from econocausal.scenario import price_scenarios  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--price", required=True)
    parser.add_argument("--quantity", required=True)
    parser.add_argument("--price-change", type=float, action="append", required=True)
    parser.add_argument("--elasticity", type=float, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/price_scenarios.csv"))
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"Input CSV does not exist: {args.csv}")
    result = price_scenarios(
        pd.read_csv(args.csv), args.price, args.quantity,
        args.price_change, args.elasticity,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_json(orient="records", indent=2))
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
