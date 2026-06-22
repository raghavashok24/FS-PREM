"""
src/models/baselines.py
-----------------------
Standard classifier wrappers used as FS-PREM baselines:
Logistic Regression, Random Forest, Gradient Boosting, SVM.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def get_logistic_regression(**kwargs) -> Pipeline:
    params = dict(max_iter=1000, class_weight="balanced",
                  random_state=RANDOM_STATE, C=1.0)
    params.update(kwargs)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**params)),
    ])


def get_random_forest(**kwargs) -> Pipeline:
    params = dict(n_estimators=100, max_depth=8, class_weight="balanced",
                  random_state=RANDOM_STATE)
    params.update(kwargs)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(**params)),
    ])


def get_gradient_boosting(**kwargs) -> Pipeline:
    params = dict(n_estimators=100, learning_rate=0.05,
                  max_depth=4, random_state=RANDOM_STATE)
    params.update(kwargs)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(**params)),
    ])


def get_svm(**kwargs) -> Pipeline:
    params = dict(probability=True, kernel="rbf",
                  class_weight="balanced", random_state=RANDOM_STATE)
    params.update(kwargs)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(**params)),
    ])


BASELINE_REGISTRY = {
    "LogisticRegression": get_logistic_regression,
    "RandomForest":       get_random_forest,
    "GradientBoosting":   get_gradient_boosting,
    "SVM":                get_svm,
}
