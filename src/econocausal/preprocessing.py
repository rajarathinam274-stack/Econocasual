"""Preprocessing for validated Online Retail-style transactions."""
from __future__ import annotations

import pandas as pd


def preprocess_transactions(data: pd.DataFrame) -> pd.DataFrame:
    """Return a canonical, analysis-ready positive-sales transaction frame.

    The function requires the standard transaction fields, normalizes their names,
    parses dates and numeric values, and derives line revenue and calendar fields.
    It does not infer unobserved purchases or make causal claims.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    aliases = {
        "invoiceno": "invoice_no",
        "invoice no": "invoice_no",
        "stockcode": "stock_code",
        "stock code": "stock_code",
        "description": "description",
        "quantity": "quantity",
        "invoicedate": "invoice_date",
        "invoice date": "invoice_date",
        "unitprice": "unit_price",
        "unit price": "unit_price",
        "customerid": "customer_id",
        "customer id": "customer_id",
        "country": "country",
    }
    rename = {}
    for column in data.columns:
        key = str(column).strip().lower().replace("_", " ")
        key = " ".join(key.split())
        compact = key.replace(" ", "")
        if key in aliases:
            rename[column] = aliases[key]
        elif compact in aliases:
            rename[column] = aliases[compact]
    frame = data.rename(columns=rename).copy()
    required = {"quantity", "unit_price", "invoice_date"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Required columns not found: {missing}")

    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
    frame["unit_price"] = pd.to_numeric(frame["unit_price"], errors="coerce")
    frame["invoice_date"] = pd.to_datetime(frame["invoice_date"], errors="coerce")
    frame = frame.loc[
        frame["quantity"].notna()
        & (frame["quantity"] > 0)
        & frame["unit_price"].notna()
        & (frame["unit_price"] > 0)
        & frame["invoice_date"].notna()
    ].copy()
    frame["line_revenue"] = frame["quantity"] * frame["unit_price"]
    frame["invoice_month"] = frame["invoice_date"].dt.to_period("M").astype(str)
    frame["invoice_year"] = frame["invoice_date"].dt.year.astype("int64")
    frame["invoice_weekday"] = frame["invoice_date"].dt.day_name()
    return frame.reset_index(drop=True)
