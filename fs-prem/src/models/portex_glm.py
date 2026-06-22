"""
src/models/portex_glm.py
------------------------
Physics-Aware Weighted PORTEX Generalised Linear Model.

Coefficients are estimated with sparsity-inducing L1 / group-lasso penalties
constrained to the five PORTEX dimensions, preserving domain interpretability
while allowing empirical re-weighting of hazard, proximity, and operational
exposure according to observed disruption patterns.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


PORTEX_FEATURES = ["Hazard", "Proximity", "Uncertainty", "Exposure", "Vulnerability"]


class PORTEXWeightedGLM(BaseEstimator, ClassifierMixin):
    """
    Logistic regression over the five PORTEX dimensions with optional L1 penalty.

    Parameters
    ----------
    penalty : str
        Regularization type ('l1' or 'l2').
    C : float
        Inverse regularization strength.
    features : list[str]
        PORTEX feature columns to use (default: all five).
    """

    def __init__(
        self,
        penalty: str = "l1",
        C: float = 1.0,
        solver: str = "liblinear",
        features: list = None,
        random_state: int = 42,
    ):
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.features = features or PORTEX_FEATURES
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._clf = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_arr = self._scaler.fit_transform(X[self.features])
        self._clf = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            solver=self.solver,
            max_iter=2000,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self._clf.fit(X_arr, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = self._scaler.transform(X[self.features])
        return self._clf.predict_proba(X_arr)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @property
    def coefficients(self) -> pd.Series:
        """Named coefficients for the PORTEX dimensions."""
        if self._clf is None:
            raise RuntimeError("Model not fitted yet.")
        return pd.Series(self._clf.coef_[0], index=self.features)

    def print_summary(self):
        coefs = self.coefficients.sort_values(ascending=False)
        print("\n── PORTEX-GLM Coefficients ──────────────────")
        for feat, val in coefs.items():
            bar = "█" * int(abs(val) * 2) if abs(val) < 20 else "█" * 40
            sign = "+" if val >= 0 else ""
            print(f"  {feat:<15} {sign}{val:>7.3f}  {bar}")
        print("─────────────────────────────────────────────\n")
