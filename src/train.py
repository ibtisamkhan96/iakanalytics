"""Train, evaluate, benchmark, and save the predictive-maintenance models.

Three models:
  1. Failure classifier (binary): will this machine fail? Compares Logistic Regression, Random
     Forest, and Histogram Gradient Boosting, all class-weighted for the 3.4% imbalance.
  2. Anomaly detector: Isolation Forest on the sensors, unsupervised, for novelty not in the labels.
  3. Failure-type classifier: among failures, which mode (tool wear, heat, power, overstrain).

Saves the chosen pipeline, the anomaly model, the failure-type model, metrics.json, and plots.
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .pipeline import load_data, quality_report, get_xy, failure_type_frame, FEATURES, SENSORS

MODELS_DIR = "models"
REPORTS_DIR = "reports"


def _scaled(numeric, estimator):
    pre = ColumnTransformer([("num", StandardScaler(), numeric)], remainder="passthrough")
    return Pipeline([("pre", pre), ("clf", estimator)])


def main(data_path="data/ai4i2020.csv"):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = load_data(data_path)
    qr = quality_report(df)
    print("Data quality:", json.dumps(qr))

    X, y, feats = get_xy(df)
    numeric = [f for f in feats if f != "type"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    candidates = {
        "logistic_regression": _scaled(numeric, LogisticRegression(max_iter=1000, class_weight="balanced")),
        "random_forest": _scaled(numeric, RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)),
        "hist_gradient_boosting": _scaled(numeric, HistGradientBoostingClassifier(class_weight="balanced", random_state=42)),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    bench, best_name, best_ap = {}, None, -1
    for name, pipe in candidates.items():
        ap = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="average_precision", n_jobs=-1)
        roc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        bench[name] = {"cv_pr_auc": round(ap.mean(), 4), "cv_pr_auc_std": round(ap.std(), 4),
                       "cv_roc_auc": round(roc.mean(), 4)}
        print(f"  {name}: PR-AUC {ap.mean():.4f} +/- {ap.std():.4f}, ROC-AUC {roc.mean():.4f}")
        if ap.mean() > best_ap:
            best_ap, best_name = ap.mean(), name

    print(f"Best model: {best_name}")
    best = candidates[best_name].fit(X_tr, y_tr)
    proba = best.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    test_metrics = {
        "test_roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "test_pr_auc": round(float(average_precision_score(y_te, proba)), 4),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "report": classification_report(y_te, pred, output_dict=True, zero_division=0),
    }
    print("Test ROC-AUC:", test_metrics["test_roc_auc"], "PR-AUC:", test_metrics["test_pr_auc"])

    # ---- Anomaly detector (unsupervised, on sensors) ----
    iso = Pipeline([("scaler", StandardScaler()),
                    ("iso", IsolationForest(contamination=0.04, random_state=42))])
    iso.fit(X_tr[SENSORS])
    flags = iso.predict(X_te[SENSORS])  # -1 anomaly, 1 normal
    is_anom = (flags == -1)
    anomaly_metrics = {
        "flagged_pct": round(float(is_anom.mean()) * 100, 2),
        "recall_of_failures": round(float(is_anom[y_te.values == 1].mean()), 3),
    }
    print("Anomaly detector flags", anomaly_metrics["flagged_pct"], "%, catches",
          anomaly_metrics["recall_of_failures"], "of failures unsupervised")

    # ---- Failure-type classifier ----
    Xf, yf = failure_type_frame(df)
    ft = _scaled([f for f in FEATURES if f != "type"], RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
    ft.fit(Xf, yf)
    type_acc = round(float(cross_val_score(ft, Xf, yf, cv=3, scoring="accuracy").mean()), 3)
    print("Failure-type classifier CV accuracy:", type_acc, "over classes", sorted(set(yf)))

    # ---- Feature importance plot (from the fitted classifier if available) ----
    try:
        clf = best.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            order = np.argsort(clf.feature_importances_)[::-1]
            plt.figure(figsize=(7, 4))
            plt.bar([feats[i] for i in order], clf.feature_importances_[order], color="#3b6fd4")
            plt.xticks(rotation=40, ha="right"); plt.ylabel("importance"); plt.title("Feature importance")
            plt.tight_layout(); plt.savefig(f"{REPORTS_DIR}/feature_importance.png", dpi=120); plt.close()
    except Exception:
        pass
    PrecisionRecallDisplay.from_predictions(y_te, proba); plt.title("Precision-Recall (failure)")
    plt.tight_layout(); plt.savefig(f"{REPORTS_DIR}/pr_curve.png", dpi=120); plt.close()
    RocCurveDisplay.from_predictions(y_te, proba); plt.title("ROC (failure)")
    plt.tight_layout(); plt.savefig(f"{REPORTS_DIR}/roc_curve.png", dpi=120); plt.close()

    # ---- Persist ----
    joblib.dump(best, f"{MODELS_DIR}/failure_classifier.joblib")
    joblib.dump(iso, f"{MODELS_DIR}/anomaly_detector.joblib")
    joblib.dump(ft, f"{MODELS_DIR}/failure_type_classifier.joblib")
    meta = {"features": feats, "sensors": SENSORS, "best_model": best_name,
            "data_quality": qr, "benchmark": bench, "test_metrics": test_metrics,
            "anomaly": anomaly_metrics, "failure_type_cv_accuracy": type_acc,
            "failure_type_classes": sorted(set(yf))}
    with open(f"{MODELS_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(f"{REPORTS_DIR}/metrics.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved models and reports.")
    return meta


if __name__ == "__main__":
    main()
