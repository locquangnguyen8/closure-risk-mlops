"""Prediction service: POST /predict returns 7-day closure risk with its main drivers."""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .features import FEATURES
from .model import ClosureRiskModel
from .simulate import ESTUARIES


class Observation(BaseModel):
    """Conditions at the end of a day for one open harvest area."""

    estuary: Literal["merrin", "kallow", "tasset"]
    date: str = Field(examples=["2026-10-20"])
    rain_1d: float = Field(ge=0)
    rain_3d: float = Field(ge=0)
    rain_7d: float = Field(ge=0)
    rain_14d: float = Field(ge=0)
    rain_forecast_3d: float = Field(ge=0, description="Forecast rain for days 1-3, mm")
    rain_forecast_4_7d: float = Field(ge=0, description="Forecast rain for days 4-7, mm")
    rain_forecast_max_day: float = Field(ge=0, description="Wettest single forecast day in the next 7, mm")
    flow: float = Field(ge=0)
    flow_7d_mean: float = Field(ge=0)
    salinity: float = Field(ge=0, le=45)
    salinity_3d_min: float = Field(ge=0, le=45)
    salinity_change_3d: float
    water_temp: float
    days_since_closure: float = Field(ge=0)


class Driver(BaseModel):
    feature: str
    value: float
    contribution: float


class Prediction(BaseModel):
    estuary: str
    probability: float
    at_risk: bool
    drivers: list[Driver]
    model_version: str


def _frame(observations: list[Observation]) -> pd.DataFrame:
    df = pd.DataFrame([o.model_dump() for o in observations])
    month = pd.to_datetime(df["date"]).dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    for e in ESTUARIES:
        df[f"estuary_{e}"] = (df["estuary"] == e).astype(int)
    return df


def create_app(model: ClosureRiskModel | None = None) -> FastAPI:
    app = FastAPI(title="closure-risk", version="0.1.0")
    app.state.model = model or ClosureRiskModel.load(os.getenv("MODEL_DIR", "models/champion"))

    @app.get("/health")
    def health():
        card = app.state.model.card
        return {"status": "ok", "model_version": card.version, "trained_through": card.trained_through}

    @app.get("/model")
    def model_card():
        return app.state.model.card

    @app.post("/predict", response_model=list[Prediction])
    def predict(observations: list[Observation]):
        m: ClosureRiskModel = app.state.model
        df = _frame(observations)
        proba = m.predict_proba(df)
        drivers = m.explain(df)
        return [
            Prediction(
                estuary=o.estuary,
                probability=round(float(p), 4),
                at_risk=bool(p >= m.card.threshold),
                drivers=d,
                model_version=m.card.version,
            )
            for o, p, d in zip(observations, proba, drivers, strict=True)
        ]

    assert set(FEATURES) <= set(_frame([_EXAMPLE]).columns), "API features out of sync with model features"
    return app


_EXAMPLE = Observation(
    estuary="merrin",
    date="2026-10-20",
    rain_1d=12.0,
    rain_3d=40.0,
    rain_7d=55.0,
    rain_14d=70.0,
    rain_forecast_3d=35.0,
    rain_forecast_4_7d=10.0,
    rain_forecast_max_day=22.0,
    flow=28.0,
    flow_7d_mean=18.0,
    salinity=23.5,
    salinity_3d_min=23.1,
    salinity_change_3d=-4.2,
    water_temp=20.5,
    days_since_closure=40,
)
