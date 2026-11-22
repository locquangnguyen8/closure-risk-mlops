"""Synthetic daily estuary data with a known drift.

The physics is deliberately simple but causal: rain raises river flow, flow
lowers salinity, and the closure rules from the monitoring program turn those
into closures. From DRIFT_START the climate gets wetter and the catchment
responds more strongly to rain (for example after land clearing), so a model
trained on earlier years under-predicts closures. That is the drift the
monitoring loop has to catch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ESTUARIES = {"merrin": 0.8, "kallow": 0.6, "tasset": 0.45}  # flow response per mm of rain
START, END = "2019-01-01", "2026-10-31"
DRIFT_START = pd.Timestamp("2025-07-01")
RAIN_TRIGGER_MM = 30.0
SALINITY_TRIGGER_PSU = 22.0


def _closures(rain: np.ndarray, salinity: np.ndarray) -> np.ndarray:
    """Apply the closure rules day by day.

    Close when rain > 30 mm in a day or salinity < 22 PSU for two consecutive days.
    Reopen after salinity has been above 22 PSU for three consecutive days and at
    least five days have passed since closing (standing in for the E. coli test).
    """
    closed = np.zeros(len(rain), dtype=bool)
    is_closed, days_closed, fresh_run, low_run = False, 0, 0, 0
    for t in range(len(rain)):
        low_run = low_run + 1 if salinity[t] < SALINITY_TRIGGER_PSU else 0
        if not is_closed and (rain[t] > RAIN_TRIGGER_MM or low_run >= 2):
            is_closed, days_closed, fresh_run = True, 0, 0
        elif is_closed:
            days_closed += 1
            fresh_run = fresh_run + 1 if salinity[t] > SALINITY_TRIGGER_PSU else 0
            if fresh_run >= 3 and days_closed >= 5:
                is_closed = False
        closed[t] = is_closed
    return closed


def simulate(seed: int = 42, start: str = START, end: str = END) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    season = np.sin(2 * np.pi * (dates.dayofyear.to_numpy() - 30) / 365.25)
    drift = (dates >= DRIFT_START).astype(float)

    # One regional weather series shared by all estuaries, with local variation.
    wet = rng.random(n) < 0.32 + 0.10 * season
    regional = np.where(wet, rng.gamma(0.75, 9.0, n), 0.0) * (1 + 0.35 * drift)

    frames = []
    for name, response in ESTUARIES.items():
        rain = np.clip(regional * rng.lognormal(0, 0.25, n), 0, None)
        k = response * (1 + 0.5 * drift)
        flow = np.empty(n)
        flow[0] = 5.0
        for t in range(1, n):
            flow[t] = 0.75 * flow[t - 1] + k[t] * rain[t] + 1.25 + rng.normal(0, 0.3)
        flow = np.clip(flow, 0.5, None)
        salinity = np.clip(34.5 - 4.2 * np.log1p(flow) + rng.normal(0, 1.0, n), 0, 36)
        temp = 21 + 5 * np.cos(2 * np.pi * (dates.dayofyear.to_numpy() - 30) / 365.25) + rng.normal(0, 1.2, n)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "estuary": name,
                    "rain_mm": rain.round(1),
                    "flow_m3s": flow.round(2),
                    "salinity_psu": salinity.round(2),
                    "water_temp_c": temp.round(2),
                    "closed": _closures(rain, salinity),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
