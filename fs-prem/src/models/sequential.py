"""
src/models/sequential.py
------------------------
Compact sequential architectures for temporal PORTEX modeling.

Both models use hidden dimensions ≤ 16 to satisfy real-time inference
constraints on streaming advisory data.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input


def build_compact_gru(
    seq_len: int,
    n_features: int,
    hidden_dim: int = 8,
    dropout: float = 0.2,
) -> Model:
    """
    Compact GRU for sequential PORTEX advisory streams.

    Architecture:
        GRU(hidden_dim) → Dropout → Dense(1, sigmoid)

    Parameters
    ----------
    seq_len : int
        Number of advisory time steps per sequence.
    n_features : int
        Number of PORTEX input features per step.
    hidden_dim : int
        GRU hidden state size (≤16 for real-time feasibility).
    dropout : float
        Dropout rate after GRU.

    Returns
    -------
    tf.keras.Model
    """
    inp = Input(shape=(seq_len, n_features), name="portex_sequence")
    x = layers.GRU(hidden_dim, return_sequences=False, name="gru")(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.BatchNormalization()(x)
    out = layers.Dense(1, activation="sigmoid", name="risk_score")(x)
    model = Model(inputs=inp, outputs=out, name="CompactGRU")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["AUC"],
    )
    return model


def build_compact_conv1d(
    seq_len: int,
    n_features: int,
    filters: int = 8,
    kernel_size: int = 3,
    dropout: float = 0.2,
) -> Model:
    """
    Compact 1-D CNN for local pattern detection in advisory sequences.

    Architecture:
        Conv1D(filters, kernel) → GlobalMaxPool → Dropout → Dense(1, sigmoid)

    Parameters
    ----------
    seq_len : int
    n_features : int
    filters : int
        Number of convolution filters (≤16 for efficiency).
    kernel_size : int
        Convolution window size.
    dropout : float
    """
    inp = Input(shape=(seq_len, n_features), name="portex_sequence")
    x = layers.Conv1D(filters, kernel_size, activation="relu",
                      padding="same", name="conv1d")(inp)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.BatchNormalization()(x)
    out = layers.Dense(1, activation="sigmoid", name="risk_score")(x)
    model = Model(inputs=inp, outputs=out, name="CompactConv1D")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["AUC"],
    )
    return model


def build_sequences(
    df,
    feature_cols: list,
    label_col: str,
    group_col: str,
    seq_len: int = 10,
):
    """
    Convert advisory-level dataframe rows into fixed-length sequences
    grouped by storm–port pair for GRU / Conv1D training.

    Returns
    -------
    X : np.ndarray  shape (n_sequences, seq_len, n_features)
    y : np.ndarray  shape (n_sequences,)
    groups : np.ndarray  shape (n_sequences,)
    """
    X_seqs, y_seqs, groups = [], [], []
    for key, grp in df.groupby(group_col):
        grp = grp.sort_values("DATE") if "DATE" in grp.columns else grp
        feats = grp[feature_cols].values
        labels = grp[label_col].values
        for i in range(seq_len, len(grp)):
            X_seqs.append(feats[i - seq_len : i])
            y_seqs.append(labels[i])
            groups.append(key)
    return np.array(X_seqs), np.array(y_seqs), np.array(groups)
