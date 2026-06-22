"""
src/data/preprocessing.py
--------------------------
PORTEX index computation, feature engineering, NaN imputation,
and binary event label generation for the FS-PREM benchmark.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# PORTEX index
# ---------------------------------------------------------------------------

def compute_hazard(wind_speed: pd.Series, gale_radius: pd.Series) -> pd.Series:
    """
    Non-linear wind + gale-radius intensity proxy.

        H = (V/74)^3  +  (2/3)(R/60)(V/74)^2
    """
    v = wind_speed.clip(lower=0)
    r = gale_radius.clip(lower=0)
    return (v / 74) ** 3 + (2 / 3) * (r / 60) * (v / 74) ** 2


def compute_proximity(distance_km: pd.Series, gale_radius_km: pd.Series) -> pd.Series:
    """
    Distance-modulated hazard gate.

        P = max(0, 1 - d / (1.5 * R_km))
    """
    denom = 1.5 * gale_radius_km.clip(lower=1e-3)
    return (1 - distance_km / denom).clip(lower=0)


def compute_portex(
    df: pd.DataFrame,
    hazard_col: str = "CMEHI_proxy",
    proximity_col: str = "proximity_factor",
    uncertainty_col: str = "uncertainty_factor",
    exposure_col: str = "exposure_score",
    vulnerability_col: str = "vulnerability_score",
    weights: dict = None,
) -> pd.DataFrame:
    """
    Compute (or re-compute) the PORTEX disruption index.

    PORTEX_{p,t} = 100 * [
        0.5  * (H * P)   +
        0.2  * U_norm    +
        0.2  * E         +
        0.1  * V
    ]

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with pre-encoded PORTEX columns.
    weights : dict, optional
        Override default component weights.

    Returns
    -------
    pd.DataFrame
        Input df with added PORTEX columns.
    """
    w = {"hazard_proximity": 0.5, "uncertainty": 0.2, "exposure": 0.2, "vulnerability": 0.1}
    if weights:
        w.update(weights)

    df = df.copy()

    # Map to canonical names
    df["Hazard"] = df[hazard_col]
    df["Proximity"] = df[proximity_col]
    df["Uncertainty"] = df[uncertainty_col]
    df["Exposure"] = df[exposure_col]
    df["Vulnerability"] = df[vulnerability_col]

    # Normalize Hazard and Uncertainty to [0, 1] using robust percentiles
    def robust_norm(s: pd.Series) -> pd.Series:
        lo, hi = s.quantile(0.05), s.quantile(0.95)
        return ((s - lo) / (hi - lo + 1e-9)).clip(0, 1)

    H_norm = robust_norm(df["Hazard"])
    U_norm = robust_norm(df["Uncertainty"])

    df["PORTEX_risk_recalc"] = 100 * (
        w["hazard_proximity"] * (H_norm * df["Proximity"]) +
        w["uncertainty"]      * U_norm +
        w["exposure"]         * df["Exposure"] +
        w["vulnerability"]    * df["Vulnerability"]
    )

    print(f"[preprocessing] PORTEX recomputed. Range: "
          f"[{df['PORTEX_risk_recalc'].min():.2f}, {df['PORTEX_risk_recalc'].max():.2f}]")
    return df


def create_event_labels(
    df: pd.DataFrame,
    portex_col: str = "PORTEX_risk_recalc",
    threshold_percentile: float = 0.95,
) -> pd.DataFrame:
    """
    Create binary disruption labels at a given percentile threshold.

    Parameters
    ----------
    df : pd.DataFrame
    portex_col : str
        PORTEX score column to threshold.
    threshold_percentile : float
        Quantile cutoff for positive label (default 0.95 → ~5% positive rate).

    Returns
    -------
    pd.DataFrame
        df with 'Event_Label' column added.
    """
    df = df.copy()
    threshold = df[portex_col].quantile(threshold_percentile)
    df["Event_Label"] = (df[portex_col] >= threshold).astype(int)

    n_pos = df["Event_Label"].sum()
    pct = 100 * n_pos / len(df)
    print(f"[preprocessing] Threshold={threshold:.3f} (p{threshold_percentile*100:.0f}), "
          f"Positives={n_pos:,} ({pct:.1f}%)")
    return df


def impute_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric NaNs with column medians."""
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        median = df[col].median()
        n_filled = df[col].isna().sum()
        df[col] = df[col].fillna(median)
        if n_filled > 0:
            print(f"[preprocessing] Imputed {n_filled} NaNs in '{col}' with median={median:.4f}")
    return df


FEATURE_COLS = ["Hazard", "Proximity", "Uncertainty", "Exposure", "Vulnerability"]
GROUP_COL = "STORMS_ACTIVE"
LABEL_COL = "Event_Label"
