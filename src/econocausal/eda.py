"""Reusable exploratory data analysis for transaction-level retail data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _column_map(data: pd.DataFrame) -> dict[str, Any]:
    return {str(c).strip().lower().replace(" ", ""): c for c in data.columns}


def summarize_transactions(data: pd.DataFrame, top_n: int = 10) -> dict[str, Any]:
    """Summarize observed retail transactions without making causal claims.

    Required columns: InvoiceDate, Quantity, UnitPrice. InvoiceNo, StockCode,
    and Description are optional. Sales charts/totals include only rows with a
    valid date, positive quantity, and positive unit price. Returns/cancellations
    are counted separately and not silently treated as positive demand.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    mapping = _column_map(data)
    required = ["invoicedate", "quantity", "unitprice"]
    missing = [c for c in required if c not in mapping]
    if missing:
        raise ValueError(f"Required EDA columns missing: {missing}")

    date_col, qty_col, price_col = (mapping[c] for c in required)
    product_col, invoice_col = mapping.get("description"), mapping.get("invoiceno")
    frame = data.copy()
    frame["_date"] = pd.to_datetime(frame[date_col], errors="coerce")
    frame["_quantity"] = pd.to_numeric(frame[qty_col], errors="coerce")
    frame["_unit_price"] = pd.to_numeric(frame[price_col], errors="coerce")
    frame["_line_revenue"] = frame["_quantity"] * frame["_unit_price"]
    valid = (
        frame["_date"].notna()
        & np.isfinite(frame["_quantity"])
        & np.isfinite(frame["_unit_price"])
        & (frame["_quantity"] > 0)
        & (frame["_unit_price"] > 0)
    )
    positive = frame.loc[valid].copy()
    positive["_month"] = positive["_date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        positive.groupby("_month", as_index=False)
        .agg(revenue=("_line_revenue", "sum"),
             units=("_quantity", "sum"),
             transaction_lines=("_line_revenue", "size"))
        .sort_values("_month").reset_index(drop=True)
    )
    monthly["_month"] = monthly["_month"].dt.strftime("%Y-%m")

    if product_col is not None:
        products = (
            positive.groupby(product_col, dropna=False, as_index=False)
            .agg(revenue=("_line_revenue", "sum"),
                 units=("_quantity", "sum"),
                 transaction_lines=("_line_revenue", "size"))
            .sort_values("revenue", ascending=False).head(top_n).reset_index(drop=True)
        )
        products[product_col] = products[product_col].fillna("(missing)").astype(str)
    else:
        products = pd.DataFrame(columns=["product", "revenue", "units", "transaction_lines"])

    cancellation_rows = 0
    if invoice_col is not None:
        cancellation_rows = int(
            frame[invoice_col].astype("string").str.strip().str.upper().str.startswith("C", na=False).sum()
        )
    missing_cells = int(data.isna().sum().sum())
    cell_count = int(data.shape[0] * data.shape[1])
    stock_col = mapping.get("stockcode")
    unique_products = (
        int(frame[stock_col].nunique(dropna=True)) if stock_col
        else int(frame[product_col].nunique(dropna=True)) if product_col else None
    )
    return {
        "row_count": int(len(data)),
        "column_count": int(data.shape[1]),
        "missing_cells": missing_cells,
        "missing_cell_rate": float(missing_cells / cell_count) if cell_count else 0.0,
        "duplicate_rows": int(data.duplicated().sum()),
        "invalid_date_rows": int(frame["_date"].isna().sum()),
        "negative_quantity_rows": int((frame["_quantity"] < 0).sum()),
        "zero_quantity_rows": int((frame["_quantity"] == 0).sum()),
        "nonpositive_price_rows": int((frame["_unit_price"] <= 0).sum()),
        "cancellation_invoice_rows": cancellation_rows,
        "positive_sales_rows": int(len(positive)),
        "unique_invoices": int(frame[invoice_col].nunique(dropna=True)) if invoice_col else None,
        "unique_products": unique_products,
        "date_min": frame["_date"].min().isoformat() if frame["_date"].notna().any() else None,
        "date_max": frame["_date"].max().isoformat() if frame["_date"].notna().any() else None,
        "positive_sales_revenue": float(positive["_line_revenue"].sum()),
        "positive_sales_units": float(positive["_quantity"].sum()),
        "monthly": monthly,
        "top_products": products,
        "interpretation_warning": (
            "Descriptive statistics cover observed transaction records only. They omit non-purchase opportunities "
            "and do not identify causal price effects."
        ),
    }


def plot_eda(data: pd.DataFrame, output_dir: str | Path, top_n: int = 10) -> dict[str, Any]:
    """Write PNG plots, CSV summaries, and a JSON quality/EDA report."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = summarize_transactions(data, top_n=top_n)
    monthly = result["monthly"].copy()
    if not monthly.empty:
        monthly["_month_dt"] = pd.to_datetime(monthly["_month"])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(monthly["_month_dt"], monthly["revenue"], marker="o")
        ax.set(title="Monthly observed positive-sales revenue", xlabel="Month", ylabel="Revenue (GBP)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "monthly_revenue.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(monthly["_month_dt"], monthly["units"], marker="o")
        ax.set(title="Monthly observed positive-sales quantity", xlabel="Month", ylabel="Units")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "monthly_units.png", dpi=140)
        plt.close(fig)

    mapping = _column_map(data)
    prices = pd.to_numeric(data[mapping["unitprice"]], errors="coerce")
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if not prices.empty:
        upper = float(prices.quantile(0.99))
        displayed = prices[prices <= upper]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(displayed, bins=40)
        ax.set(title="Unit-price distribution (positive prices, capped at 99th percentile)",
               xlabel="Unit price (GBP)", ylabel="Transaction lines")
        fig.tight_layout()
        fig.savefig(output / "unit_price_distribution.png", dpi=140)
        plt.close(fig)
        result["price_plot_upper_cutoff_p99"] = upper

    product_col = mapping.get("description")
    products = result["top_products"]
    if product_col and not products.empty:
        ordered = products.sort_values("revenue", ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(ordered))))
        ax.barh(ordered[product_col].astype(str), ordered["revenue"])
        ax.set(title=f"Top {len(ordered)} products by observed positive-sales revenue",
               xlabel="Revenue (GBP)", ylabel="Product")
        fig.tight_layout()
        fig.savefig(output / "top_products_revenue.png", dpi=140)
        plt.close(fig)

    monthly.to_csv(output / "monthly_summary.csv", index=False)
    products.to_csv(output / "top_products.csv", index=False)
    serializable = {k: v for k, v in result.items() if k not in {"monthly", "top_products"}}
    (output / "eda_summary.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    result["output_dir"] = str(output)
    return result
