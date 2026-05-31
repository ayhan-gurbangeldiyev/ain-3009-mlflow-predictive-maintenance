"""Exploratory data analysis and dataset visualizations.

Generates presentation-ready figures (PNG) and summary tables (CSV) describing
the AI4I 2020 dataset: class imbalance, feature distributions by failure,
correlations, and machine-type composition. Figures are written to
``reports/figures/`` so they can be dropped straight into the slide deck.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import REPORTS_DIR
from data_preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    load_dataframe,
)

FIG_DIR = REPORTS_DIR / "figures"


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"saved {path}")


def summary_table(df) -> None:
    """Write a numeric summary table used as a slide table."""
    summary = df[NUMERIC_FEATURES + [TARGET]].describe().round(2).T
    out = REPORTS_DIR / "data_summary.csv"
    summary.to_csv(out)
    print(f"saved {out}")


def class_balance(df) -> None:
    counts = df[TARGET].value_counts().sort_index()
    rate = df[TARGET].mean() * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["No failure (0)", "Failure (1)"], counts.values,
                  color=["#4caf50", "#e53935"])
    ax.bar_label(bars)
    ax.set_title(f"Class imbalance — failure rate {rate:.1f}%")
    ax.set_ylabel("Count")
    _save(fig, "01_class_imbalance.png")


def type_distribution(df) -> None:
    counts = df[CATEGORICAL_FEATURES[0]].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index, counts.values, color="#1e88e5")
    ax.bar_label(bars)
    ax.set_title("Machine type composition")
    ax.set_ylabel("Count")
    _save(fig, "02_machine_types.png")


def feature_distributions(df) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feature in zip(axes.ravel(), NUMERIC_FEATURES):
        for label, color in [(0, "#4caf50"), (1, "#e53935")]:
            ax.hist(df[df[TARGET] == label][feature], bins=30, alpha=0.6,
                    color=color, label=f"failure={label}", density=True)
        ax.set_title(feature)
        ax.legend(fontsize=8)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Numeric feature distributions by failure", fontsize=14)
    _save(fig, "03_feature_distributions.png")


def correlation_heatmap(df) -> None:
    corr = df[NUMERIC_FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=7)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_title("Feature correlation")
    _save(fig, "04_correlation_heatmap.png")


def main(data_path: str | None) -> None:
    df = load_dataframe(data_path)
    summary_table(df)
    class_balance(df)
    type_distribution(df)
    feature_distributions(df)
    correlation_heatmap(df)
    print("EDA visualizations complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EDA figures.")
    parser.add_argument("--data", default=None, help="Path to ai4i2020.csv")
    args = parser.parse_args()
    main(args.data)
