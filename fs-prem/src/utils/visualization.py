"""
src/utils/visualization.py
---------------------------
Publication-quality plot helpers for FS-PREM.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.metrics import roc_curve, auc


PALETTE = {
    "portex":     "#2196F3",
    "baseline":   "#FF5722",
    "sequential": "#4CAF50",
    "conformal":  "#9C27B0",
    "neutral":    "#607D8B",
}


def plot_portex_distribution(portex_scores: pd.Series, threshold: float, ax=None):
    """Histogram of PORTEX risk scores with 95th-percentile threshold line."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(portex_scores, bins=40, color=PALETTE["portex"], alpha=0.75, edgecolor="white")
    ax.axvline(threshold, color="crimson", linestyle="--", linewidth=1.8,
               label=f"95th percentile ({threshold:.2f})")
    ax.set_xlabel("PORTEX Risk Score (0–100)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Distribution of PORTEX Risk Scores", fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_feature_importance(coefs: pd.Series, ax=None):
    """Horizontal bar chart of PORTEX-GLM coefficients."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    colors = [PALETTE["portex"] if v >= 0 else PALETTE["baseline"] for v in coefs.values]
    ax.barh(coefs.index, coefs.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient", fontsize=11)
    ax.set_title("PORTEX-GLM Feature Importance", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_loso_auc_comparison(results: dict, ax=None):
    """Bar chart comparing mean LOSO AUC across models."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    aucs = [v["mean_auc"] for v in results.values()]
    stds = [v["std_auc"] for v in results.values()]
    colors = [
        PALETTE["portex"] if "PORTEX" in n
        else PALETTE["sequential"] if any(k in n for k in ["GRU", "Conv"])
        else PALETTE["baseline"]
        for n in names
    ]
    bars = ax.bar(names, aucs, yerr=stds, capsize=5, color=colors,
                  alpha=0.85, edgecolor="white")
    ax.set_ylabel("Test AUC (LOSO)", fontsize=11)
    ax.set_title("Model Comparison — Leave-One-Storm-Out AUC", fontsize=13, fontweight="bold")
    ax.set_ylim(0.5, 1.0)
    ax.tick_params(axis="x", rotation=25)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    return ax


def plot_conformal_coverage(grid_df: pd.DataFrame, ax=None):
    """Line plot of empirical coverage vs window size for each alpha."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    for alpha, grp in grid_df.groupby("alpha"):
        ax.plot(grp["window"], grp["coverage"], marker="o",
                label=f"α={alpha}")
        ax.axhline(1 - alpha, linestyle="--", alpha=0.4)
    ax.set_xlabel("Window Size", fontsize=11)
    ax.set_ylabel("Empirical Coverage", fontsize=11)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title("Conformal Coverage vs Window Size", fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def save_figure(fig, path: str, dpi: int = 150):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[viz] Saved → {path}")
