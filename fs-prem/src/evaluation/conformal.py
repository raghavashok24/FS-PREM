"""
src/evaluation/conformal.py
----------------------------
Rolling-window conformal prediction wrapper for FS-PREM.

Provides distribution-free, calibrated uncertainty intervals
P(y ∈ C_{p,t}) ≥ 1 − α at each advisory update, valid even
under inter-storm distributional shifts.
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import List, Tuple


class RollingConformalWrapper:
    """
    Mondrian / rolling-window conformal predictor wrapping any
    sklearn-compatible classifier.

    At each advisory step the calibration queue is updated and a new
    prediction interval is emitted with nominal coverage 1 − alpha.

    Parameters
    ----------
    base_model : sklearn estimator
        Fitted classifier exposing predict_proba().
    alpha : float
        Desired miscoverage level (e.g. 0.05 → 95% coverage).
    window : int
        Rolling calibration window size (number of past advisories).
    """

    def __init__(self, base_model, alpha: float = 0.05, window: int = 100):
        self.base_model = base_model
        self.alpha = alpha
        self.window = window
        self._cal_scores: deque = deque(maxlen=window)

    def _nonconformity(self, proba: float, label: int) -> float:
        """Regression-style nonconformity: |y - p(y=1)|."""
        return abs(label - proba)

    def update_calibration(self, proba: float, true_label: int):
        """Add a new calibration score to the rolling window."""
        self._cal_scores.append(self._nonconformity(proba, true_label))

    def predict_set(self, proba: float) -> Tuple[List[int], float]:
        """
        Compute the conformal prediction set for a new advisory.

        Returns
        -------
        prediction_set : list[int]
            Labels whose nonconformity score ≤ q_hat.
        q_hat : float
            Empirical (1 - alpha) quantile of calibration scores.
        """
        if len(self._cal_scores) < 2:
            return [0, 1], 1.0  # uninformative until warm-up

        q_hat = np.quantile(list(self._cal_scores), 1 - self.alpha)
        pred_set = [
            label for label in [0, 1]
            if self._nonconformity(proba, label) <= q_hat
        ]
        return pred_set, float(q_hat)

    def evaluate_coverage(
        self,
        probas: np.ndarray,
        true_labels: np.ndarray,
    ) -> dict:
        """
        Compute empirical coverage and average prediction set size
        over a sequence of advisory predictions.

        Parameters
        ----------
        probas : np.ndarray  shape (n,)
        true_labels : np.ndarray  shape (n,)

        Returns
        -------
        dict with 'coverage', 'avg_set_size', 'q_hat_mean'
        """
        covered = 0
        set_sizes = []
        q_hats = []

        for p, y in zip(probas, true_labels):
            pred_set, q_hat = self.predict_set(p)
            covered += int(y in pred_set)
            set_sizes.append(len(pred_set))
            q_hats.append(q_hat)
            self.update_calibration(p, y)

        n = len(probas)
        return {
            "coverage": covered / n,
            "avg_set_size": np.mean(set_sizes),
            "q_hat_mean": np.mean(q_hats),
            "n": n,
        }


def evaluate_conformal_grid(
    base_model,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list,
    alpha_values: List[float] = None,
    window_sizes: List[int] = None,
) -> pd.DataFrame:
    """
    Grid search over alpha and window size; return coverage + efficiency table.
    """
    alpha_values = alpha_values or [0.05, 0.10]
    window_sizes = window_sizes or [60, 80, 100, 120, 144]

    rows = []
    probas_test = base_model.predict_proba(X_test[feature_cols])[:, 1]
    probas_cal = base_model.predict_proba(X_cal[feature_cols])[:, 1]

    for alpha in alpha_values:
        for window in window_sizes:
            cp = RollingConformalWrapper(base_model, alpha=alpha, window=window)
            # Warm-up on calibration set
            for p, y in zip(probas_cal, y_cal):
                cp.update_calibration(p, int(y))
            # Evaluate on test set
            result = cp.evaluate_coverage(probas_test, y_test.values)
            rows.append({
                "alpha": alpha,
                "window": window,
                "coverage": result["coverage"],
                "avg_set_size": result["avg_set_size"],
                "target_coverage": 1 - alpha,
            })

    return pd.DataFrame(rows)
