"""Classification metrics and population stability index (PSI) for drift."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, precision_score, recall_score, roc_auc_score


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    both = len(np.unique(y_true)) == 2
    return {
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, pred)), 4) if both else float("nan"),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4) if both else float("nan"),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
    }


def psi(reference, current, bins: int = 10, smoothing: float = 0.5) -> float:
    """Population stability index. Rule of thumb: < 0.1 stable, 0.1-0.25 moderate, > 0.25 significant shift.

    Bins are reference deciles. Each bin gets `smoothing` extra counts so that an
    empty bin in a small sample does not blow up the log term.
    """
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature (e.g. mostly zero rain): fall back to a zero / non-zero split
        edges = np.array([-np.inf, reference.min(), np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] + smoothing
    cur = np.histogram(current, edges)[0] + smoothing
    ref, cur = ref / ref.sum(), cur / cur.sum()
    return round(float(np.sum((cur - ref) * np.log(cur / ref))), 4)
