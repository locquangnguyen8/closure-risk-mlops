from fastapi.testclient import TestClient

from closure_risk.api import _EXAMPLE, create_app
from closure_risk.model import ClosureRiskModel


def _client(feats):
    return TestClient(create_app(ClosureRiskModel.train(feats[feats["date"] < "2024-01-01"], version="api-test")))


def test_health_reports_model_version(feats):
    r = _client(feats).get("/health")
    assert r.json()["model_version"] == "api-test"


def test_predict_returns_probability_and_drivers(feats):
    calm = _EXAMPLE.model_copy(
        update={
            "rain_1d": 0,
            "rain_3d": 0,
            "rain_7d": 2,
            "rain_14d": 5,
            "rain_forecast_3d": 0,
            "rain_forecast_4_7d": 0,
            "rain_forecast_max_day": 0,
            "flow": 3,
            "flow_7d_mean": 3,
            "salinity": 31,
            "salinity_3d_min": 30.5,
            "salinity_change_3d": 0.2,
            "days_since_closure": 200,
        }
    )
    storm = _EXAMPLE.model_copy(
        update={"rain_forecast_3d": 80, "rain_forecast_max_day": 55, "salinity_3d_min": 22.5, "salinity": 22.8}
    )
    r = _client(feats).post("/predict", json=[calm.model_dump(), storm.model_dump()])
    assert r.status_code == 200
    low, high = r.json()
    assert high["probability"] > low["probability"]
    assert high["at_risk"] and not low["at_risk"]
    assert len(high["drivers"]) == 3


def test_invalid_input_is_rejected(feats):
    bad = _EXAMPLE.model_dump() | {"salinity": -5}
    assert _client(feats).post("/predict", json=[bad]).status_code == 422
