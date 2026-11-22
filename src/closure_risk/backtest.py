"""Rolling-origin backtest and model selection.

Train on everything before a month, test on that month, and repeat for 12
months. Training data stops 7 days before each test month (an embargo), because
the target looks 7 days ahead and would otherwise leak test-period closures
into training labels.

The candidate with the best overall balanced accuracy becomes the production
model family; on a tie (within 0.005) the simpler model wins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import HORIZON_DAYS
from .metrics import classification_metrics
from .model import ClosureRiskModel

EMBARGO = pd.Timedelta(days=HORIZON_DAYS)
CANDIDATES = ["logistic", "xgboost"]  # simplest first: it wins ties
TIE_MARGIN = 0.005


def rule_baseline(df: pd.DataFrame) -> np.ndarray:
    """What a person might do without a model: flag a forecast day over the rain trigger, or already-low salinity."""
    return ((df["rain_forecast_max_day"] > 30) | (df["salinity_3d_min"] < 24)).astype(float).to_numpy()


def month_starts(first: str, n: int) -> list[pd.Timestamp]:
    return list(pd.date_range(first, periods=n, freq="MS"))


def rolling_backtest(df: pd.DataFrame, first_test_month: str = "2024-07-01", n_folds: int = 12) -> dict:
    preds = {name: [] for name in [*CANDIDATES, "rule"]}
    truth, folds = [], []
    for start in month_starts(first_test_month, n_folds):
        end = start + pd.offsets.MonthBegin(1)
        train = df[df["date"] < start - EMBARGO]
        test = df[(df["date"] >= start) & (df["date"] < end)]
        fold = {"month": start.strftime("%Y-%m"), "positives": int(test["target"].sum()), "n": len(test)}
        for family in CANDIDATES:
            p = ClosureRiskModel.train(train, family).predict_proba(test)
            preds[family].append(p)
            fold[family] = classification_metrics(test["target"], p)["balanced_accuracy"]
        preds["rule"].append(rule_baseline(test))
        truth.append(test["target"].to_numpy())
        folds.append(fold)

    y = np.concatenate(truth)
    overall = {name: classification_metrics(y, np.concatenate(p)) for name, p in preds.items()}
    best = max(CANDIDATES, key=lambda f: overall[f]["balanced_accuracy"])
    selected = next(
        f for f in CANDIDATES if overall[f]["balanced_accuracy"] >= overall[best]["balanced_accuracy"] - TIE_MARGIN
    )
    return {"overall": overall, "folds": folds, "selected_family": selected}
