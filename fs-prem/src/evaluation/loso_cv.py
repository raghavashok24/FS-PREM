"""
src/evaluation/loso_cv.py
--------------------------
Leave-One-Storm-Out (LOSO) cross-validation for FS-PREM.

Each storm's advisories are entirely withheld from training, simulating
deployment conditions where each storm presents novel dynamics.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix,
)
from typing import List, Dict


def run_loso_cv(
    model,
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "Event_Label",
    group_col: str = "STORMS_ACTIVE",
    threshold: float = 0.5,
    verbose: bool = True,
) -> Dict:
    """
    Perform Leave-One-Storm-Out cross-validation.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Must implement fit() and predict_proba().
    df : pd.DataFrame
    feature_cols : list[str]
    label_col : str
    group_col : str
    threshold : float
        Decision threshold for precision/recall/F1.
    verbose : bool

    Returns
    -------
    dict with keys: fold_results, mean_auc, std_auc, mean_f1, all_preds
    """
    logo = LeaveOneGroupOut()
    X = df[feature_cols]
    y = df[label_col].values
    groups = df[group_col].values

    fold_results = []
    all_preds = np.zeros(len(df))
    all_true = np.zeros(len(df))

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        storm_name = groups[test_idx[0]]

        # Skip folds with no positive examples
        if y_test.sum() == 0:
            if verbose:
                print(f"  [fold {fold_idx:02d}] Storm={storm_name} — no positives, skipping")
            continue

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= threshold).astype(int)

        auc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

        fold_results.append({
            "storm": storm_name,
            "auc": auc,
            "pr_auc": pr_auc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "n_test": len(y_test),
            "n_positive": int(y_test.sum()),
        })

        all_preds[test_idx] = proba
        all_true[test_idx] = y_test

        if verbose:
            print(f"  [fold {fold_idx:02d}] Storm={storm_name:<25} "
                  f"AUC={auc:.3f}  F1={f1:.3f}  "
                  f"TP={tp} FN={fn} FP={fp}")

    results_df = pd.DataFrame(fold_results)
    mean_auc = results_df["auc"].mean()
    std_auc = results_df["auc"].std()

    if verbose:
        print(f"\n── LOSO Summary ──────────────────────────────")
        print(f"  Folds evaluated : {len(results_df)}")
        print(f"  Mean AUC        : {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"  Mean F1         : {results_df['f1'].mean():.4f}")
        print(f"─────────────────────────────────────────────\n")

    return {
        "fold_results": results_df,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "mean_f1": results_df["f1"].mean(),
        "all_preds": all_preds,
        "all_true": all_true,
    }
