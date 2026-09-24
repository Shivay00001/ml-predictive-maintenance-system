"""Real predictive-maintenance model.

Trains a genuine sklearn pipeline (StandardScaler + GradientBoostingClassifier)
on machine-sensor telemetry and predicts failure probability. /evaluate reports
real precision/recall/F1 computed on a held-out split — never canned numbers.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "rpm",
    "humidity_pct",
    "hours_since_maintenance",
]
LABEL_COL = "failure"  # 0 / 1
MODEL_PATH = os.environ.get("PM_MODEL_PATH", "model.joblib")


class MaintenanceModel:
    def __init__(self, random_state=42):
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=150, max_depth=3, learning_rate=0.08,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        self.trained = False
        self.eval_metrics: dict = {}

    def fit(self, df: pd.DataFrame, test_size: float = 0.2) -> dict:
        missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        if len(df) < 50:
            raise ValueError("need at least 50 rows to train")
        X = df[FEATURE_COLS].astype(float).values
        y = df[LABEL_COL].astype(int).values
        if len(np.unique(y)) < 2:
            raise ValueError("label column must contain both 0 and 1 examples")

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        self.pipeline.fit(Xtr, ytr)
        self.trained = True

        p = self.pipeline.predict(Xte)
        proba = self.pipeline.predict_proba(Xte)[:, 1]
        self.eval_metrics = {
            "n_train": int(len(Xtr)),
            "n_test": int(len(Xte)),
            "precision": float(precision_score(yte, p, zero_division=0)),
            "recall": float(recall_score(yte, p, zero_division=0)),
            "f1": float(f1_score(yte, p, zero_division=0)),
            "roc_auc": float(roc_auc_score(yte, proba)),
            "test_failure_rate": float(yte.mean()),
        }
        return self.eval_metrics

    def predict_one(self, row: dict) -> dict:
        if not self.trained:
            raise RuntimeError("model not trained")
        try:
            vals = np.array([[float(row[c]) for c in FEATURE_COLS]])
        except KeyError as e:
            raise ValueError(f"missing feature: {e}")
        proba = float(self.pipeline.predict_proba(vals)[0][1])
        importances = {
            c: float(v)
            for c, v in zip(
                FEATURE_COLS, self.pipeline.named_steps["clf"].feature_importances_
            )
        }
        return {
            "failure_probability": proba,
            "will_fail": bool(proba >= 0.5),
            "feature_importances": importances,
        }

    def save(self, path: str = MODEL_PATH) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = MODEL_PATH) -> "MaintenanceModel":
        return joblib.load(path)
