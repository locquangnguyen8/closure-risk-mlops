import numpy as np
import pandas as pd
import pytest

from closure_risk.backtest import EMBARGO, rolling_backtest
from closure_risk.metrics import psi
from closure_risk.model import ClosureRiskModel
from closure_risk.monitor import Policy, simulate_production


def test_psi_detects_shift_and_ignores_noise():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.05
    assert psi(ref, rng.normal(1, 1, 5000)) > 0.25


@pytest.mark.parametrize("family", ["xgboost", "logistic"])
def test_model_round_trip_and_explanations(feats, tmp_path, family):
    train = feats[feats["date"] < "2023-01-01"]
    sample = feats[feats["date"] >= "2023-01-01"].head(50)
    model = ClosureRiskModel.train(train, family, version="test")
    model.save(tmp_path)
    loaded = ClosureRiskModel.load(tmp_path)
    np.testing.assert_allclose(model.predict_proba(sample), loaded.predict_proba(sample))
    assert loaded.card.version == "test" and loaded.card.family == family
    assert loaded.card.trained_through == "2022-12-31"
    drivers = loaded.explain(sample.head(1))[0]
    assert len(drivers) == 3 and abs(drivers[0]["contribution"]) >= abs(drivers[-1]["contribution"])


def test_backtest_models_beat_the_rule_and_one_is_selected(feats):
    result = rolling_backtest(feats, n_folds=4)
    rule = result["overall"]["rule"]["balanced_accuracy"]
    for family in ("logistic", "xgboost"):
        assert result["overall"][family]["balanced_accuracy"] > rule + 0.1
    assert result["selected_family"] in {"logistic", "xgboost"}
    assert len(result["folds"]) == 4


def test_logistic_explanations_add_up_to_the_prediction(feats):
    model = ClosureRiskModel.train(feats[feats["date"] < "2023-01-01"], "logistic")
    row = feats.iloc[[5000]]
    contribs = model.linear.contributions(row[model.card.features].to_numpy(float))
    logit = contribs.sum() + model.linear.intercept
    np.testing.assert_allclose(1 / (1 + np.exp(-logit)), model.predict_proba(row))


def test_backtest_respects_the_embargo(feats, monkeypatch):
    seen = []
    original = ClosureRiskModel.train.__func__

    def spy(cls, df, family="xgboost", version=None):
        seen.append(df["date"].max())
        return original(cls, df, family, version)

    monkeypatch.setattr(ClosureRiskModel, "train", classmethod(spy))
    rolling_backtest(feats, first_test_month="2024-07-01", n_folds=2)
    assert seen[0] < pd.Timestamp("2024-07-01") - EMBARGO + pd.Timedelta(days=1)


def test_monitoring_detects_drift_and_retrains(feats):
    logged = []
    run = simulate_production(feats, on_model=lambda m, role: logged.append((m.card.version, role)))
    actions = [m["action"] for m in run.months]
    assert any(a.startswith("train") for a in actions), "drift should trigger retraining"
    assert logged[0] == ("v1", "champion")
    assert any(role == "challenger" for _, role in logged)
    # A promotion or discard always follows a training month.
    for i, a in enumerate(actions[:-1]):
        if a.startswith("train"):
            assert actions[i + 1].split()[0] in {"promote", "discard"}


def test_strict_policy_never_promotes_without_improvement(feats):
    run = simulate_production(feats, policy=Policy(min_improvement=1.0))
    assert run.champion.card.version == "v1"
    assert not any(m["action"].startswith("promote") for m in run.months)
