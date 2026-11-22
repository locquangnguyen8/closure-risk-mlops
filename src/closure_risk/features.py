"""Feature engineering and the 7-day-ahead closure target.

Every feature uses only information available at the end of day t, plus a
7-day rain forecast, which a real system would take from a weather service.
The forecast is simulated as the true future rain with error that grows with
lead time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .simulate import ESTUARIES

HORIZON_DAYS = 7
FEATURES = [
    "rain_1d",
    "rain_3d",
    "rain_7d",
    "rain_14d",
    "rain_forecast_3d",
    "rain_forecast_4_7d",
    "rain_forecast_max_day",
    "flow",
    "flow_7d_mean",
    "salinity",
    "salinity_3d_min",
    "salinity_change_3d",
    "water_temp",
    "days_since_closure",
    "month_sin",
    "month_cos",
    *[f"estuary_{e}" for e in ESTUARIES],
]


def build_features(raw: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Return one row per estuary-day that is currently open, with features and the target.

    Days when an area is already closed are excluded: the useful question is
    whether an open area will close, not whether a closed one stays closed.
    """
    rng = np.random.default_rng(seed)
    out = []
    for name, g in raw.sort_values("date").groupby("estuary", sort=False):
        g = g.reset_index(drop=True)
        rain = g["rain_mm"]
        closed = g["closed"].astype(int)
        f = pd.DataFrame({"date": g["date"], "estuary": name})
        f["rain_1d"] = rain
        f["rain_3d"] = rain.rolling(3, min_periods=1).sum()
        f["rain_7d"] = rain.rolling(7, min_periods=1).sum()
        f["rain_14d"] = rain.rolling(14, min_periods=1).sum()
        # Forecast of day t+k: true rain with multiplicative error that grows with lead time k.
        fc = pd.DataFrame({k: rain.shift(-k).fillna(0) * rng.lognormal(0, 0.2 + 0.08 * k, len(g)) for k in range(1, 8)})
        f["rain_forecast_3d"] = fc[[1, 2, 3]].sum(axis=1).round(1)
        f["rain_forecast_4_7d"] = fc[[4, 5, 6, 7]].sum(axis=1).round(1)
        f["rain_forecast_max_day"] = fc.max(axis=1).round(1)
        f["flow"] = g["flow_m3s"]
        f["flow_7d_mean"] = g["flow_m3s"].rolling(7, min_periods=1).mean()
        f["salinity"] = g["salinity_psu"]
        f["salinity_3d_min"] = g["salinity_psu"].rolling(3, min_periods=1).min()
        f["salinity_change_3d"] = g["salinity_psu"].diff(3).fillna(0)
        f["water_temp"] = g["water_temp_c"]
        last_closed = g["date"].where(closed == 1).ffill()
        f["days_since_closure"] = (g["date"] - last_closed).dt.days.fillna(365).clip(upper=365)
        month = g["date"].dt.month
        f["month_sin"] = np.sin(2 * np.pi * month / 12)
        f["month_cos"] = np.cos(2 * np.pi * month / 12)
        for e in ESTUARIES:
            f[f"estuary_{e}"] = int(e == name)
        # Target: closed on any of the next 7 days.
        future_closed = closed[::-1].rolling(HORIZON_DAYS, min_periods=1).max()[::-1].shift(-1)
        f["target"] = future_closed
        f["is_open"] = closed == 0
        out.append(f)
    df = pd.concat(out, ignore_index=True)
    # The last 7 days have no complete label window yet, exactly as in production.
    labelled = df["date"] <= df["date"].max() - pd.Timedelta(days=HORIZON_DAYS)
    df = df[df["is_open"] & labelled].drop(columns="is_open")
    df["target"] = df["target"].astype(int)
    return df.sort_values(["date", "estuary"]).reset_index(drop=True)
