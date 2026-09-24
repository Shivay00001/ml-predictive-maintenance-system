"""FastAPI service: real predictive-maintenance inference from a trained model."""

import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import FEATURE_COLS, LABEL_COL, MODEL_PATH, MaintenanceModel

app = FastAPI(title="Predictive Maintenance", version="1.0.0")
_MODEL: MaintenanceModel | None = None


def get_model() -> MaintenanceModel:
    global _MODEL
    if _MODEL is None:
        if os.path.exists(MODEL_PATH):
            _MODEL = MaintenanceModel.load(MODEL_PATH)
        else:
            raise HTTPException(
                status_code=503, detail="model not trained yet — POST /train first"
            )
    return _MODEL


class TrainRequest(BaseModel):
    csv_path: str


class PredictRequest(BaseModel):
    sensors: dict


@app.get("/health")
def health():
    return {"status": "ok", "model_trained": os.path.exists(MODEL_PATH)}


@app.get("/schema")
def schema():
    return {"features": FEATURE_COLS, "label": LABEL_COL}


@app.post("/train")
def train(req: TrainRequest):
    if not os.path.exists(req.csv_path):
        raise HTTPException(status_code=404, detail=f"CSV not found: {req.csv_path}")
    global _MODEL
    df = pd.read_csv(req.csv_path)
    model = MaintenanceModel()
    metrics = model.fit(df)  # real holdout metrics
    model.save(MODEL_PATH)
    _MODEL = model
    return {"status": "trained", "model_path": MODEL_PATH, "holdout_metrics": metrics}


@app.post("/evaluate")
def evaluate(req: TrainRequest):
    """Train-free evaluation: fit on the given CSV and report real holdout metrics."""
    if not os.path.exists(req.csv_path):
        raise HTTPException(status_code=404, detail=f"CSV not found: {req.csv_path}")
    df = pd.read_csv(req.csv_path)
    return {"holdout_metrics": MaintenanceModel().fit(df)}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return get_model().predict_one(req.sensors)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
