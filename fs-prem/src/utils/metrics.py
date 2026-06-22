"""
src/utils/metrics.py
--------------------
Evaluation metric helpers for FS-PREM experiments.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)


def compute_all_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Return a dict of standard classification metrics."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auc":       roc_auc_score(y_true, y_proba),
        "pr_auc":    average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def summarise_loso(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean ± std across LOSO folds."""
    numeric = fold_results.select_dtypes(include=[np.number])
    summary = numeric.agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]
    summary["mean±std"] = summary.apply(
        lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1
    )
    return summary
