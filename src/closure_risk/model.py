"""Model training, scoring, explanation and persistence for two model families.

`xgboost`: gradient-boosted trees; explanations are TreeSHAP values from XGBoost itself.
`logistic`: standardised logistic regression; explanations are each feature's exact
contribution to the log-odds, coef * (x - mean) / scale.

Both are saved as JSON (no pickles), and serving needs neither scikit-learn nor MLflow.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xgboost as xgb

from .features import FEATURES

Family = Literal["xgboost", "logistic"]

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 5,
    "eval_metric": "logloss",
    "random_state": 0,
    "n_jobs": 4,
}
LOGISTIC_PARAMS = {"C": 1.0, "class_weight": "balanced", "max_iter": 2000}


@dataclass
class ModelCard:
    version: str
    family: str
    trained_through: str
    n_training_rows: int
    positive_rate: float
    threshold: float = 0.5  # both families are trained class-balanced, so 0.5 targets balanced accuracy
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    params: dict = field(default_factory=dict)


class _Linear:
    """Numpy-only logistic regression scorer (what gets saved and served)."""

    def __init__(self, mean, scale, coef, intercept):
        self.mean, self.scale = np.asarray(mean), np.asarray(scale)
        self.coef, self.intercept = np.asarray(coef), float(intercept)

    def contributions(self, X: np.ndarray) -> np.ndarray:
        return self.coef * (X - self.mean) / self.scale

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self.contributions(X).sum(axis=1) + self.intercept
        return 1 / (1 + np.exp(-z))

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
        }


class ClosureRiskModel:
    def __init__(
        self,
        card: ModelCard,
        booster: xgb.XGBClassifier | None = None,
        linear: _Linear | None = None,
        sklearn_model=None,
    ):
        self.card, self.booster, self.linear = card, booster, linear
        self.sklearn_model = sklearn_model  # kept in memory after training so MLflow can log the native model

    @classmethod
    def train(cls, df: pd.DataFrame, family: Family = "xgboost", version: str | None = None) -> ClosureRiskModel:
        X, y = df[FEATURES].to_numpy(float), df["target"].to_numpy()
        card = ModelCard(
            version=version or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
            family=family,
            trained_through=str(df["date"].max().date()),
            n_training_rows=len(df),
            positive_rate=round(float(y.mean()), 4),
        )
        if family == "xgboost":
            pos = max(int(y.sum()), 1)
            card.params = dict(XGB_PARAMS)
            booster = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=(len(y) - pos) / pos)
            booster.fit(df[FEATURES], y)
            return cls(card, booster=booster)
        if family == "logistic":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            card.params = dict(LOGISTIC_PARAMS)
            pipe = make_pipeline(StandardScaler(), LogisticRegression(**LOGISTIC_PARAMS)).fit(X, y)
            scaler, lr = pipe[0], pipe[1]
            linear = _Linear(scaler.mean_, scaler.scale_, lr.coef_[0], lr.intercept_[0])
            return cls(card, linear=linear, sklearn_model=pipe)
        raise ValueError(f"Unknown model family {family!r}")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.booster is not None:
            return self.booster.predict_proba(df[FEATURES])[:, 1]
        return self.linear.predict_proba(df[FEATURES].to_numpy(float))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df) >= self.card.threshold).astype(int)

    def explain(self, df: pd.DataFrame, top: int = 3) -> list[list[dict]]:
        """Top per-row contributions to the log-odds of closure."""
        if self.booster is not None:
            contribs = self.booster.get_booster().predict(xgb.DMatrix(df[FEATURES]), pred_contribs=True)[:, :-1]
        else:
            contribs = self.linear.contributions(df[FEATURES].to_numpy(float))
        out = []
        for row, values in zip(df[FEATURES].to_dict("records"), contribs, strict=True):
            order = np.argsort(-np.abs(values))[:top]
            out.append(
                [
                    {
                        "feature": FEATURES[i],
                        "value": float(row[FEATURES[i]]),
                        "contribution": round(float(values[i]), 3),
                    }
                    for i in order
                ]
            )
        return out

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for old in ("model.json", "linear.json"):
            (d / old).unlink(missing_ok=True)
        if self.booster is not None:
            self.booster.save_model(d / "model.json")
        else:
            (d / "linear.json").write_text(json.dumps(self.linear.to_dict()), encoding="utf-8")
        (d / "model_card.json").write_text(json.dumps(asdict(self.card), indent=2), encoding="utf-8")
        return d

    @classmethod
    def load(cls, directory: str | Path) -> ClosureRiskModel:
        d = Path(directory)
        card = ModelCard(**json.loads((d / "model_card.json").read_text(encoding="utf-8")))
        if card.family == "xgboost":
            booster = xgb.XGBClassifier()
            booster.load_model(d / "model.json")
            return cls(card, booster=booster)
        return cls(card, linear=_Linear(**json.loads((d / "linear.json").read_text(encoding="utf-8"))))
