"""A predictive baseline, explicitly separate from causal estimation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def fit_demand_baseline(data: pd.DataFrame, outcome_col: str,
                        feature_cols: list[str], test_size: float = 0.2,
                        random_state: int = 42) -> dict[str, object]:
    """Fit a Ridge prediction model and return holdout metrics and fitted model."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    absent = [c for c in [outcome_col, *feature_cols] if c not in data.columns]
    if absent:
        raise ValueError(f"Missing columns: {absent}")
    frame = data[[outcome_col, *feature_cols]].copy()
    frame[outcome_col] = pd.to_numeric(frame[outcome_col], errors="coerce")
    frame = frame.dropna(subset=[outcome_col])
    if len(frame) < 5:
        raise ValueError("At least five rows with a numeric outcome are required")
    X, y = frame[feature_cols], frame[outcome_col]
    if len(X) < 2:
        raise ValueError("Not enough rows to split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    numeric = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in feature_cols if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical))
    model = Pipeline([("features", ColumnTransformer(transformers)), ("ridge", Ridge())])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "model": model,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mean_residual": float(np.mean(y_test.to_numpy() - pred)),
        "interpretation": "Predictive metrics only; not a causal price-effect estimate.",
    }
