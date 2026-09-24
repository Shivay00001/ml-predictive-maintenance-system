import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_demo_data import make_demo_csv  # noqa: E402
from model import FEATURE_COLS, MaintenanceModel  # noqa: E402


def test_train_evaluate_predict(tmp_path):
    csv = str(tmp_path / "demo.csv")
    make_demo_csv(csv, n=1500, failure_rate=0.1)
    df = pd.read_csv(csv)

    model = MaintenanceModel()
    metrics = model.fit(df)
    assert model.trained
    for k in ("precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= metrics[k] <= 1.0, k
    # the synthetic pattern is learnable: recall must be meaningfully high
    assert metrics["recall"] > 0.7
    assert metrics["roc_auc"] > 0.8

    degraded = df[df["failure"] == 1].iloc[0].to_dict()
    healthy = df[df["failure"] == 0].iloc[0].to_dict()
    r_bad = model.predict_one(degraded)
    r_ok = model.predict_one(healthy)
    assert r_bad["failure_probability"] > r_ok["failure_probability"]
    assert set(r_bad["feature_importances"]) == set(FEATURE_COLS)
    assert abs(sum(r_bad["feature_importances"].values()) - 1.0) < 1e-6

    p = str(tmp_path / "m.joblib")
    model.save(p)
    loaded = MaintenanceModel.load(p)
    assert (
        abs(
            loaded.predict_one(degraded)["failure_probability"]
            - r_bad["failure_probability"]
        )
        < 1e-9
    )
