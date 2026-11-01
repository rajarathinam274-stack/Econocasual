import pandas as pd
import pytest

from econocausal.preprocessing import preprocess_transactions


def test_preprocess_canonicalizes_fields_and_derives_features():
    data = pd.DataFrame(
        {
            "InvoiceNo": ["100", "101", "102"],
            "Quantity": [2, -1, "bad"],
            "UnitPrice": [5, 4, 3],
            "InvoiceDate": ["2011-12-01 08:00", "2011-12-01 09:00", "not-a-date"],
        }
    )
    result = preprocess_transactions(data)
    assert len(result) == 1
    assert result.loc[0, "invoice_no"] == "100"
    assert result.loc[0, "line_revenue"] == 10
    assert result.loc[0, "invoice_month"] == "2011-12"
    assert result.loc[0, "invoice_year"] == 2011
    assert result.loc[0, "invoice_weekday"] == "Thursday"


def test_preprocess_accepts_spaced_column_names():
    data = pd.DataFrame(
        {
            "Invoice Date": ["2011-01-02"],
            "Quantity": [1],
            "Unit Price": [2.5],
        }
    )
    result = preprocess_transactions(data)
    assert result.loc[0, "line_revenue"] == 2.5


def test_preprocess_requires_transaction_fields():
    with pytest.raises(ValueError, match="Required columns"):
        preprocess_transactions(pd.DataFrame({"Quantity": [1], "UnitPrice": [2]}))
