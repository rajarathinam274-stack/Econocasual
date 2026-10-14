"""Validation utilities for transaction-level retail data."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import pandas as pd


@dataclass
class ValidationReport:
    rows_input: int
    rows_output: int
    duplicate_rows: int
    missing_cells: int
    invalid_price_rows: int
    invalid_quantity_rows: int
    cancellation_rows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_transactions(data: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate common Online Retail-style columns and return cleaned rows.

    Required columns: Quantity and UnitPrice (case-insensitive). Negative
    quantities are retained in the input report as potential returns/cancellations,
    but excluded from the clean positive-sales frame. Invalid/nonpositive prices
    are excluded. Duplicate full rows are dropped.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    frame = data.copy()
    lookup = {str(c).strip().lower().replace(" ", ""): c for c in frame.columns}
    if "quantity" not in lookup or "unitprice" not in lookup:
        raise ValueError("Required columns not found: Quantity and UnitPrice")
    q_col, p_col = lookup["quantity"], lookup["unitprice"]
    rows_input = len(frame)
    missing_cells = int(frame.isna().sum().sum())
    duplicate_rows = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().copy()
    frame[q_col] = pd.to_numeric(frame[q_col], errors="coerce")
    frame[p_col] = pd.to_numeric(frame[p_col], errors="coerce")
    invalid_price = int((frame[p_col].isna() | (frame[p_col] <= 0)).sum())
    invalid_quantity = int((frame[q_col].isna() | (frame[q_col] == 0)).sum())
    cancellation = int((frame[q_col] < 0).sum())
    clean = frame.loc[
        frame[q_col].notna() & (frame[q_col] > 0)
        & frame[p_col].notna() & (frame[p_col] > 0)
    ].copy()
    report = ValidationReport(
        rows_input=rows_input,
        rows_output=len(clean),
        duplicate_rows=duplicate_rows,
        missing_cells=missing_cells,
        invalid_price_rows=invalid_price,
        invalid_quantity_rows=invalid_quantity,
        cancellation_rows=cancellation,
    )
    return clean, report
