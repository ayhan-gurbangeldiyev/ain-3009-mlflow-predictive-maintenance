"""Data loading and preprocessing for the AI4I 2020 predictive maintenance set.

The dataset contains 10,000 rows of synthetic machine sensor readings. The
modelling target is the binary ``Machine failure`` flag. The individual failure
mode columns (TWF, HDF, PWF, OSF, RNF) are dropped because they directly encode
the target and would leak label information into the features.
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DATA_PATH, RANDOM_STATE

TARGET = "Machine failure"
DROP_COLS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]


def load_dataframe(path: str | None = None) -> pd.DataFrame:
    """Read the raw CSV into a DataFrame."""
    return pd.read_csv(path or DATA_PATH)


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric columns and one-hot encode the machine ``Type``."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def make_pipeline(estimator) -> Pipeline:
    """Wrap an estimator behind the shared preprocessing step."""
    return Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])


def load_and_split(path: str | None = None, test_size: float = 0.2):
    """Return a stratified train/test split of features and target."""
    df = load_dataframe(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df[TARGET].astype(int)
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


def _validate(path: str | None) -> None:
    """Print a short data quality summary used during development."""
    df = load_dataframe(path)
    print(f"Shape: {df.shape}")
    print(f"Target column present: {TARGET in df.columns}")
    failure_rate = df[TARGET].mean()
    print(f"Failure rate (class imbalance): {failure_rate:.4f}")
    print(f"Missing values total: {int(df.isna().sum().sum())}")
    print(f"Machine types: {df['Type'].value_counts().to_dict()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the dataset.")
    parser.add_argument("--data", default=None, help="Path to ai4i2020.csv")
    args = parser.parse_args()
    _validate(args.data)
