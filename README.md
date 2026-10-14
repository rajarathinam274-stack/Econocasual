# EconoCausal — Dynamic Pricing via Double Machine Learning

A Python research project to investigate price–demand relationships, with explicit checks for data quality and causal-identification limitations. The current implementation contains data validation, descriptive bias diagnostics, a predictive baseline, tests, a transaction-audit CLI, and a GitHub Actions test workflow. **A validated DML estimate is not implemented yet.**

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run tests

```bash
# Windows PowerShell
$env:PYTHONPATH = "src"
pytest -q
# macOS/Linux
PYTHONPATH=src pytest -q
```

Pushes to `main` and pull requests targeting `main` trigger `.github/workflows/python-tests.yml`.

## Run a transaction audit

The input CSV must contain columns named `Quantity` and `UnitPrice` (case-insensitive).

```bash
python scripts/audit_csv.py path/to/transactions.csv --output-dir outputs
```

The command writes `clean_transactions.csv` and `data_quality_report.json` into the output directory. The current cleaning rule excludes non-positive quantities and non-positive/missing prices from the positive-sales frame; negative quantities are counted as potential returns/cancellations in the report and should not be silently interpreted as demand.

## Current modules

- `src/econocausal/validation.py` — transaction validation and cleaning report.
- `src/econocausal/bias_audit.py` — missingness/duplication checks, subgroup prediction-error audit, and preliminary causal-readiness checks.
- `src/econocausal/baseline.py` — holdout Ridge prediction baseline with MAE, RMSE, and mean residual; prediction is not causal inference.
- `scripts/audit_csv.py` — command-line data audit and cleaned CSV export.
- `tests/test_core.py` — tests for validation and audit utilities.

## Research guardrails

A transaction dataset containing only observed purchases may omit times when customers saw a price but did not buy. That selection problem can undermine price-elasticity estimates. A DML model cannot by itself eliminate unobserved confounding, invalid treatment assignment, or lack of overlap. Before interpreting any coefficient causally, the project must document the unit of analysis, treatment/outcome definitions, observed confounders, overlap, and assumptions. If the data cannot support identification, report predictive/associational findings and limitations rather than a causal claim.

## Dataset

Candidate: UCI Online Retail. Review the source, data dictionary, and licence/terms before downloading or redistributing: https://archive.ics.uci.edu/dataset/352/online+retail. No dataset is bundled in this repository yet.
