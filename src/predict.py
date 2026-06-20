"""Load the trained models and assess a single machine reading into a maintenance decision.

This is the inference / decision-support layer the desktop tool and any service would call.
"""

import json
import os

import joblib
import pandas as pd

from .pipeline import engineer_features, FEATURES, SENSORS, TYPE_MAP

# Status thresholds on the failure probability.
WATCH, ALARM = 0.15, 0.50


def load_artifacts(models_dir="models"):
    return {
        "clf": joblib.load(os.path.join(models_dir, "failure_classifier.joblib")),
        "iso": joblib.load(os.path.join(models_dir, "anomaly_detector.joblib")),
        "ftype": joblib.load(os.path.join(models_dir, "failure_type_classifier.joblib")),
        "meta": json.load(open(os.path.join(models_dir, "meta.json"))),
    }


def _row(reading):
    """Build the engineered feature row from a raw sensor reading dict."""
    t = reading.get("Type", reading.get("type", "L"))
    if t in (0, 1, 2):
        t = {v: k for k, v in TYPE_MAP.items()}[t]
    raw = pd.DataFrame([{
        "Type": t,
        "air_temp_K": float(reading["air_temp_K"]),
        "process_temp_K": float(reading["process_temp_K"]),
        "rot_speed_rpm": float(reading["rot_speed_rpm"]),
        "torque_Nm": float(reading["torque_Nm"]),
        "tool_wear_min": float(reading["tool_wear_min"]),
    }])
    return engineer_features(raw)


def assess(reading, art=None):
    art = art or load_artifacts()
    feat = _row(reading)
    X = feat[FEATURES].astype(float)

    prob = float(art["clf"].predict_proba(X)[:, 1][0])
    anomaly = bool(art["iso"].predict(feat[SENSORS])[0] == -1)
    likely_type = str(art["ftype"].predict(X)[0])

    if prob >= ALARM:
        status, colour = "ALARM", "red"
    elif prob >= WATCH or anomaly:
        status, colour = "WATCH", "orange"
    else:
        status, colour = "OK", "green"

    return {
        "failure_probability": round(prob, 3),
        "predicted_failure": prob >= ALARM,
        "anomaly": anomaly,
        "status": status,
        "colour": colour,
        "likely_failure_type": likely_type,
    }


if __name__ == "__main__":
    art = load_artifacts()
    print("Normal reading:", assess(
        {"Type": "L", "air_temp_K": 298, "process_temp_K": 308, "rot_speed_rpm": 1500, "torque_Nm": 40, "tool_wear_min": 80}, art))
    print("Heavy/worn reading:", assess(
        {"Type": "L", "air_temp_K": 304, "process_temp_K": 313, "rot_speed_rpm": 1300, "torque_Nm": 65, "tool_wear_min": 220}, art))
