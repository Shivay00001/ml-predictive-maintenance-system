"""DEMO DATA GENERATOR — SYNTHETIC DATA, NOT REAL SENSOR TELEMETRY.

Simulates machine-sensor readings with a known degradation->failure pattern so
the model can be tested end-to-end. NEVER present this file as real data.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(11)


def make_demo_csv(path: str, n: int = 4000, failure_rate: float = 0.08) -> str:
    wear = RNG.uniform(0, 1, n)  # hidden degradation state
    fail = (wear > 1 - failure_rate).astype(int)

    df = pd.DataFrame(
        {
            "temperature_c": 65 + 30 * wear + RNG.normal(0, 3, n),
            "vibration_mm_s": 2 + 14 * wear**2 + RNG.normal(0, 0.8, n),
            "pressure_bar": 8 - 3 * wear + RNG.normal(0, 0.4, n),
            "rpm": 1450 + RNG.normal(0, 40, n) - 150 * wear,
            "humidity_pct": 45 + RNG.normal(0, 8, n),
            "hours_since_maintenance": RNG.uniform(0, 2000, n) * (0.3 + 0.7 * wear),
            "failure": fail,
        }
    )
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    out = make_demo_csv("demo_sensors.csv")
    print(f"wrote SYNTHETIC DEMO data -> {out} (NOT real sensor telemetry)")
