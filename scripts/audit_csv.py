"""Run transaction validation and data-quality audit on a local CSV."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from econocausal.validation import validate_transactions  # noqa: E402
from econocausal.bias_audit import data_quality_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Input transaction CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for audit outputs")
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"Input CSV does not exist: {args.csv}")
    data = pd.read_csv(args.csv)
    cleaned, report = validate_transactions(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = args.output_dir / "clean_transactions.csv"
    report_path = args.output_dir / "data_quality_report.json"
    cleaned.to_csv(cleaned_path, index=False)
    report_payload = {"validation": report.to_dict(), "quality": data_quality_audit(data)}
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(json.dumps({"cleaned_csv": str(cleaned_path), "report_json": str(report_path), **report.to_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
