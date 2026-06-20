"""Data pipeline for the AI4I 2020 predictive-maintenance dataset.

Responsibilities: load, run data-quality checks, engineer physically meaningful features, and
hand back a clean (X, y) for modelling. Kept separate from training so the same pipeline runs in
the model, the predictor, and the desktop tool.

Dataset: UCI AI4I 2020 Predictive Maintenance (10,000 rows, 3.4% failures). Sensors are air and
process temperature, rotational speed, torque and tool wear, plus a product quality type (L/M/H).
The five failure-mode flags (TWF, HDF, PWF, OSF, RNF) are components of the label and are excluded
from the model inputs to avoid leakage; they are used only as the target for failure-type analysis.
"""

import numpy as np
import pandas as pd

TARGET = "Machine failure"
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
TYPE_MAP = {"L": 0, "M": 1, "H": 2}  # ordinal product quality

# The raw sensor columns we model on (after renaming to clean names).
SENSORS = ["air_temp_K", "process_temp_K", "rot_speed_rpm", "torque_Nm", "tool_wear_min"]
FEATURES = ["type", "air_temp_K", "process_temp_K", "rot_speed_rpm", "torque_Nm",
            "tool_wear_min", "power_W", "temp_diff_K"]

_RENAME = {
    "Air temperature [K]": "air_temp_K", "Process temperature [K]": "process_temp_K",
    "Rotational speed [rpm]": "rot_speed_rpm", "Torque [Nm]": "torque_Nm",
    "Tool wear [min]": "tool_wear_min",
}


def load_data(path="data/ai4i2020.csv"):
    df = pd.read_csv(path)
    return df.rename(columns=_RENAME)


def quality_report(df):
    """Return a dict of data-quality findings (and the kind of thing the JD calls 'data quality checks')."""
    rep = {
        "rows": int(len(df)),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "failure_rate_pct": round(float(df[TARGET].mean()) * 100, 2),
        "out_of_range": {},
    }
    # Simple physical range checks on the sensors.
    ranges = {"air_temp_K": (290, 310), "process_temp_K": (300, 320),
              "rot_speed_rpm": (1000, 3000), "torque_Nm": (0, 80), "tool_wear_min": (0, 260)}
    for col, (lo, hi) in ranges.items():
        if col in df:
            rep["out_of_range"][col] = int(((df[col] < lo) | (df[col] > hi)).sum())
    return rep


def engineer_features(df):
    """Add physically meaningful features. Power and temperature gradient are the physics behind
    two of the real failure modes (power and heat-dissipation failures)."""
    df = df.copy()
    df["type"] = df["Type"].map(TYPE_MAP).astype(int)
    # Mechanical power P = torque * angular velocity, with rpm -> rad/s.
    df["power_W"] = df["torque_Nm"] * df["rot_speed_rpm"] * 2 * np.pi / 60.0
    df["temp_diff_K"] = df["process_temp_K"] - df["air_temp_K"]
    return df


def get_xy(df):
    df = engineer_features(df)
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)
    return X, y, list(FEATURES)


def failure_type_frame(df):
    """For the failure-type model: rows that failed, with the dominant failure mode as the label."""
    df = engineer_features(df)
    failed = df[df[TARGET] == 1].copy()
    # Dominant mode = the first flagged failure mode for that row.
    def dominant(row):
        for m in FAILURE_MODES:
            if row[m] == 1:
                return m
        return "UNKNOWN"
    failed["failure_type"] = failed.apply(dominant, axis=1)
    failed = failed[failed["failure_type"] != "UNKNOWN"]
    return failed[FEATURES].astype(float), failed["failure_type"]
