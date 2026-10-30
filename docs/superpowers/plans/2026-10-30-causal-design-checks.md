# Causal Design Checks Implementation Plan

> **For agentic workers:** Implement this plan task-by-task with runnable checks and a separate commit for the real code increment.

**Goal:** Add a conservative, executable causal-design assessment that reports data support without claiming identification, alongside tests for the existing non-causal prediction baseline.

**Architecture:** Add one focused `causal_design` module that validates treatment, outcome, and observed covariate columns, computes finite-support and common-support diagnostics, and always returns an explicit non-identification warning. Add a small CLI for JSON output and tests covering valid, invalid, binary-treatment, and baseline behavior. Do not add DML because this repository has no verified treatment-assignment or outcome data that support a defensible causal estimand.

**Tech Stack:** Python 3.11, pandas, scikit-learn, pytest, argparse, JSON.

---

### Task 1: Add causal-design assessment module

**Files:**
- Create: `src/econocausal/causal_design.py`

- [ ] Implement `assess_causal_design(data, treatment_col, outcome_col, covariate_cols)` with explicit column validation, numeric finite-support diagnostics, covariate missingness, binary-treatment group sizes, and an unconditional `causal_claim_permitted: False` result.

### Task 2: Add an executable CLI

**Files:**
- Create: `scripts/causal_design.py`

- [ ] Read a CSV, run the assessment, and write a JSON report with a non-causal interpretation warning.

### Task 3: Add runnable tests

**Files:**
- Create: `tests/test_causal_design.py`

- [ ] Test complete numeric data, missing columns, binary-treatment support, and the existing `fit_demand_baseline` predictive-only contract.

### Task 4: Run checks and commit

- [ ] Install only the declared dependencies if needed.
- [ ] Run `PYTHONPATH=src pytest -q` and a CLI smoke test on a temporary CSV.
- [ ] Inspect the JSON output and git diff.
- [ ] Commit only the implementation and tests as one milestone commit, then push `main` if authenticated write access succeeds.
