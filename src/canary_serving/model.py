"""Train the three model versions and the serving path that turns a request into a score.

- v1      logistic regression on the seven raw features (the current production model).
- v2-good the same plus the pass ratio approved / enrolled (slightly better, see results).
- v2-bad  the v2-good weights shipped with two classic serving bugs:
          1. the serving code assembles the feature vector in the wrong order
             (sem1_grade and sem1_approved swapped), which silently shifts every score;
          2. the guard for students with zero enrolled units was removed, so those requests
             raise ZeroDivisionError and the API answers HTTP 500.
  Offline training metrics for v2-bad look identical to v2-good, because the weights are the
  same. Only production signals (errors, score distribution) can reveal it, which is the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, Dataset, generate

VERSIONS: tuple[str, ...] = ("v1", "v2-good", "v2-bad")
BAD_ORDER: list[str] = [
    "sem1_grade" if f == "sem1_approved" else "sem1_approved" if f == "sem1_grade" else f
    for f in FEATURES
]


@dataclass
class ModelBundle:
    """Everything the server needs for one version: weights, serving config, drift reference."""

    version: str
    estimator: Pipeline
    feature_order: list[str]
    use_ratio: bool
    guard_zero_enrolled: bool
    reference_X: np.ndarray  # training inputs in FEATURES order, for input drift
    reference_scores: np.ndarray  # scores on those inputs, for output drift
    metrics: dict[str, float] = field(default_factory=dict)

    def vector(self, student: dict[str, float]) -> list[float]:
        """The serving-side feature assembly. This is where v2-bad is broken."""
        values = [float(student[name]) for name in self.feature_order]
        if self.use_ratio:
            enrolled, approved = float(student["sem1_enrolled"]), float(student["sem1_approved"])
            if self.guard_zero_enrolled and enrolled == 0:
                values.append(0.0)
            else:
                values.append(approved / enrolled)  # ZeroDivisionError when unguarded
        return values

    def predict_one(self, student: dict[str, float]) -> float:
        x = np.array([self.vector(student)])
        return float(self.estimator.predict_proba(x)[0, 1])


def _design(X: np.ndarray, use_ratio: bool) -> np.ndarray:
    """Training-side features (always correct): raw columns plus the guarded pass ratio."""
    if not use_ratio:
        return X
    enrolled, approved = (
        X[:, FEATURES.index("sem1_enrolled")],
        X[:, FEATURES.index("sem1_approved")],
    )
    ratio = np.divide(approved, enrolled, out=np.zeros(len(X)), where=enrolled > 0)
    return np.column_stack([X, ratio])


def _fit(train: Dataset, use_ratio: bool) -> Pipeline:
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return pipe.fit(_design(train.X, use_ratio), train.y)


def evaluate(bundle: ModelBundle, data: Dataset) -> dict[str, float]:
    """Score through the serving path, so serving bugs show up here as they would in production."""
    scores, labels, errors = [], [], 0
    for row, label in zip(data.rows(), data.y, strict=True):
        try:
            scores.append(bundle.predict_one(row))
            labels.append(int(label))
        except ZeroDivisionError:
            errors += 1
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "error_rate": errors / len(data.y),
        "mean_score": float(np.mean(scores)),
    }


def train_all(seed: int = 42, n_train: int = 6000, n_test: int = 3000) -> dict[str, ModelBundle]:
    """Train v1 and v2 on the same seeded data and derive v2-bad from v2-good."""
    train, test = generate(n_train, seed=seed), generate(n_test, seed=seed + 1)
    ref_X = train.X[:2000]
    bundles: dict[str, ModelBundle] = {}
    for version, use_ratio in (("v1", False), ("v2-good", True)):
        est = _fit(train, use_ratio)
        bundles[version] = ModelBundle(
            version=version,
            estimator=est,
            feature_order=list(FEATURES),
            use_ratio=use_ratio,
            guard_zero_enrolled=True,
            reference_X=ref_X,
            reference_scores=est.predict_proba(_design(ref_X, use_ratio))[:, 1],
        )
    bundles["v2-bad"] = replace(
        bundles["v2-good"], version="v2-bad", feature_order=BAD_ORDER, guard_zero_enrolled=False
    )
    for bundle in bundles.values():
        bundle.metrics = evaluate(bundle, test)
    return bundles


def save_all(bundles: dict[str, ModelBundle], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for version, bundle in bundles.items():
        path = out_dir / f"{version}.joblib"
        joblib.dump(bundle, path)
        paths.append(path)
    return paths


def load(model_dir: Path, version: str) -> ModelBundle:
    """Load one version. Only load files you built yourself: joblib files can run code."""
    if version not in VERSIONS:
        raise ValueError(f"unknown model version {version!r}; expected one of {VERSIONS}")
    bundle = joblib.load(model_dir / f"{version}.joblib")
    if not isinstance(bundle, ModelBundle) or bundle.version != version:
        raise ValueError(f"{model_dir / version}.joblib does not contain model {version}")
    return bundle
