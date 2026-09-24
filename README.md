# ml-predictive-maintenance-system

**Real** predictive-maintenance model (rebuilt 2026-09-24 — the previous version
was a simulated-inference stub and has been deleted). Trains a genuine sklearn
pipeline (`StandardScaler` + `GradientBoostingClassifier`) on machine-sensor
telemetry and serves real failure probabilities.

## Expected CSV columns

| column | meaning |
|---|---|
| `temperature_c` | bearing/motor temperature |
| `vibration_mm_s` | vibration velocity |
| `pressure_bar` | line pressure |
| `rpm` | shaft speed |
| `humidity_pct` | ambient humidity |
| `hours_since_maintenance` | run hours since last service |
| `failure` | label, 0 / 1 (required) |

## Endpoints

- `POST /train {"csv_path": ...}` — trains, saves model, returns **real holdout
  metrics** (precision / recall / F1 / ROC-AUC on a stratified 20% split)
- `POST /evaluate {"csv_path": ...}` — same metrics without saving
- `POST /predict {"sensors": {...}}` — real `failure_probability` + feature importances
- `GET /health`, `GET /schema`

## Quickstart

```bash
pip install -r requirements.txt
python generate_demo_data.py   # SYNTHETIC demo sensors — for testing only
uvicorn app:app --port 8002
curl -X POST localhost:8002/train -H 'Content-Type: application/json' \
  -d '{"csv_path": "demo_sensors.csv"}'
curl -X POST localhost:8002/predict -H 'Content-Type: application/json' \
  -d '{"sensors": {"temperature_c": 95, "vibration_mm_s": 15, "pressure_bar": 5.2,
      "rpm": 1300, "humidity_pct": 50, "hours_since_maintenance": 1900}}'
pytest -q
```

## Honest limits

- Demo CSV is **synthetic**; real machines need real labeled telemetry.
- Metrics are from one holdout split — use cross-validation for production claims.
- Screening tool, not a safety-certified system.
