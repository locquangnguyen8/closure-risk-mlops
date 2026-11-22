import numpy as np
import pandas as pd

from closure_risk.features import FEATURES, HORIZON_DAYS, build_features
from closure_risk.simulate import DRIFT_START, _closures, simulate


def test_simulation_is_reproducible():
    a, b = simulate(seed=1, end="2019-06-30"), simulate(seed=1, end="2019-06-30")
    pd.testing.assert_frame_equal(a, b)


def test_closure_rules():
    rain = np.array([0, 35, 0, 0, 0, 0, 0, 0, 0, 0], float)  # one day over 30 mm
    salinity = np.full(10, 30.0)
    closed = _closures(rain, salinity)
    assert not closed[0] and closed[1]
    assert closed[1:6].all()  # stays closed for at least 5 days
    assert not closed[-1]  # reopens after 3 fresh days

    low = np.array([30, 21, 21, 30, 30, 30, 30, 30, 30, 30], float)  # 2 consecutive days < 22 PSU
    assert _closures(np.zeros(10), low)[2] and not _closures(np.zeros(10), low)[1]


def test_drift_makes_closures_more_common(raw):
    before = raw[raw["date"] < DRIFT_START]["closed"].mean()
    after = raw[raw["date"] >= DRIFT_START]["closed"].mean()
    assert after > before * 1.3


def test_features_use_no_future_data_except_the_forecast(raw):
    feats = build_features(raw)
    # Changing rain *after* day t must not change any feature at t except the rain forecasts.
    cutoff = pd.Timestamp("2022-03-15")
    altered = raw.copy()
    altered.loc[altered["date"] > cutoff, "rain_mm"] *= 3
    f2 = build_features(altered)
    keep = [c for c in FEATURES if not c.startswith("rain_forecast")]
    a = feats[feats["date"] <= cutoff - pd.Timedelta(days=HORIZON_DAYS)][keep].reset_index(drop=True)
    b = f2[f2["date"] <= cutoff - pd.Timedelta(days=HORIZON_DAYS)][keep].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_only_open_days_with_complete_labels(raw, feats):
    open_days = raw[~raw["closed"]].set_index(["date", "estuary"])
    assert feats.set_index(["date", "estuary"]).index.isin(open_days.index).all()
    assert feats["date"].max() <= raw["date"].max() - pd.Timedelta(days=HORIZON_DAYS)
    assert set(feats["target"].unique()) == {0, 1}
