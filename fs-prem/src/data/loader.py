"""
src/data/loader.py
------------------
Utilities for loading and merging IBTrACS storm advisories with
AIS vessel movement records into the FS-PREM benchmark dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_portex_dataset(ais_path: str) -> pd.DataFrame:
    """
    Load the pre-merged advisory–port dataset.

    Parameters
    ----------
    ais_path : str
        Path to the master_port_storm_dataset_portex_encoded.csv

    Returns
    -------
    pd.DataFrame
        Raw merged dataset with PORTEX-encoded features.
    """
    df = pd.read_csv(ais_path)
    print(f"[loader] Loaded {len(df):,} rows × {df.shape[1]} cols")
    print(f"[loader] Ports: {sorted(df['PORT'].unique())}")
    print(f"[loader] Storms: {df['SID'].nunique()} unique storm IDs")
    return df


def load_ibtracs(ibtracs_path: str, basin: str = "NA") -> pd.DataFrame:
    """
    Load IBTrACS storm track data and filter to specified basin.

    Parameters
    ----------
    ibtracs_path : str
        Path to the IBTrACS CSV file.
    basin : str
        Storm basin code (default 'NA' = North Atlantic).

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(ibtracs_path, low_memory=False, skiprows=[1])
    if basin:
        df = df[df["BASIN"] == basin].copy()
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    df = df.dropna(subset=["ISO_TIME", "LAT", "LON"])
    print(f"[loader] IBTrACS: {len(df):,} track records, basin={basin}")
    return df


# ---------------------------------------------------------------------------
# Port coordinate reference
# ---------------------------------------------------------------------------

PORT_COORDS = {
    "Houston":        (29.7604, -95.3698),
    "New_Orleans":    (29.9511, -90.0715),
    "Mobile":         (30.6954, -88.0399),
    "Tampa":          (27.9506, -82.4572),
    "Corpus_Christi": (27.8006, -97.3964),
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometres between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))
