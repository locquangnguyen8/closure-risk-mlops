"""Month-by-month production simulation: score, monitor, retrain, shadow-test, promote.

Each month the champion model scores live data. When the month's labels are in:
- performance is checked (balanced accuracy over the month's last 28 days), and
- input drift is checked: PSI of key features over the last 90 days against the
  champion's last year of training data. (One month holds only ~70 open days,
  and PSI on samples that small exceeds the alarm threshold even without drift.)
If either crosses its threshold, a challenger is trained on all labelled data so far
and run in shadow for the next month. It replaces the champion only if it scores
better on that month, which neither model was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .backtest import EMBARGO
from .metrics import classification_metrics, psi
from .model import ClosureRiskModel

DRIFT_FEATURES = ["rain_7d", "flow_7d_mean", "salinity_3d_min"]
DRIFT_WINDOW = pd.Timedelta(days=90)


@dataclass
class Policy:
    min_balanced_accuracy: float = 0.70
    max_psi: float = 0.25
    min_improvement: float = 0.01  # challenger must beat the champion by at least this much


@dataclass
class MonitoringRun:
    months: list[dict] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    champion: ClosureRiskModel | None = None


def _reference(df: pd.DataFrame, model: ClosureRiskModel) -> pd.DataFrame:
    """The last year of the model's training data, used as the drift baseline."""
    through = pd.Timestamp(model.card.trained_through)
    return df[(df["date"] <= through) & (df["date"] > through - pd.Timedelta(days=365))]


def simulate_production(
    df: pd.DataFrame,
    live_from: str = "2025-07-01",
    policy: Policy | None = None,
    on_model=None,
    family: str = "logistic",
) -> MonitoringRun:
    """`on_model(model, role)` is called for every trained model, e.g. to log it to MLflow."""
    policy = policy or Policy()
    live_from = pd.Timestamp(live_from)
    champion = ClosureRiskModel.train(df[df["date"] < live_from - EMBARGO], family, version="v1")
    if on_model:
        on_model(champion, "champion")
    challenger: ClosureRiskModel | None = None
    run = MonitoringRun(
        events=[f"{live_from:%Y-%m}: champion v1 ({family}) deployed, trained through {champion.card.trained_through}"]
    )
    versions = 1

    last = df["date"].max()
    for start in pd.date_range(live_from, last, freq="MS"):
        end = start + pd.offsets.MonthBegin(1)
        month = df[(df["date"] >= start) & (df["date"] < end)]
        if month.empty:
            continue
        tag = f"{start:%Y-%m}"
        live = classification_metrics(month["target"], champion.predict_proba(month))
        recent = month[month["date"] >= end - pd.Timedelta(days=28)]
        recent_bal = classification_metrics(recent["target"], champion.predict_proba(recent))["balanced_accuracy"]
        ref = _reference(df, champion)
        window = df[(df["date"] >= end - DRIFT_WINDOW) & (df["date"] < end)]
        drift = {f: psi(ref[f], window[f]) for f in DRIFT_FEATURES}
        row = {
            "month": tag,
            "model": champion.card.version,
            **live,
            "balanced_accuracy_28d": recent_bal,
            "max_psi": max(drift.values()),
            "psi": drift,
            "action": "",
        }

        # 1. A challenger from last month has now run in shadow on this month: decide.
        if challenger is not None:
            shadow = classification_metrics(month["target"], challenger.predict_proba(month))
            gain = shadow["balanced_accuracy"] - live["balanced_accuracy"]
            if gain >= policy.min_improvement:
                run.events.append(
                    f"{tag}: promoted {challenger.card.version} (shadow balanced accuracy "
                    f"{shadow['balanced_accuracy']:.3f} vs champion {live['balanced_accuracy']:.3f})"
                )
                row["action"] = f"promote {challenger.card.version}"
                champion = challenger
                if on_model:
                    on_model(champion, "champion")
            else:
                run.events.append(
                    f"{tag}: kept {champion.card.version}; challenger {challenger.card.version} "
                    f"did not improve enough ({gain:+.3f})"
                )
                row["action"] = f"discard {challenger.card.version}"
            challenger = None

        # 2. Check the (possibly new) champion's health and decide whether to train a challenger.
        else:
            reasons = []
            if recent_bal < policy.min_balanced_accuracy:
                reasons.append(f"28-day balanced accuracy {recent_bal:.3f} < {policy.min_balanced_accuracy}")
            if row["max_psi"] > policy.max_psi:
                worst = max(drift, key=drift.get)
                reasons.append(f"PSI {drift[worst]:.2f} on {worst} > {policy.max_psi}")
            if reasons and end <= last:
                versions += 1
                challenger = ClosureRiskModel.train(df[df["date"] < end - EMBARGO], family, version=f"v{versions}")
                if on_model:
                    on_model(challenger, "challenger")
                run.events.append(
                    f"{tag}: retraining triggered ({'; '.join(reasons)}); "
                    f"challenger {challenger.card.version} goes to shadow next month"
                )
                row["action"] = f"train {challenger.card.version}"
        run.months.append(row)

    run.champion = champion
    return run
