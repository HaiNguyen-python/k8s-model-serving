"""Model versions: determinism, the planted v1 -> v2 gain, and the v2-bad serving bugs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from canary_serving.data import FEATURES, UCI_COLUMNS, generate, load_uci
from canary_serving.model import ModelBundle, load, save_all, train_all


def test_generator_is_deterministic_and_plants_zero_enrolment() -> None:
    a, b = generate(2000, seed=3), generate(2000, seed=3)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y)
    zero = a.X[:, FEATURES.index("sem1_enrolled")] == 0
    assert 0.03 < zero.mean() < 0.09
    assert a.y[zero].mean() > a.y[~zero].mean()  # zero enrolment is a real risk signal


def test_training_is_deterministic(bundles: dict[str, ModelBundle]) -> None:
    again = train_all(seed=42, n_train=3000, n_test=1500)
    for v in ("v1", "v2-good"):
        coef_a = bundles[v].estimator[-1].coef_
        coef_b = again[v].estimator[-1].coef_
        assert np.allclose(coef_a, coef_b)


def test_v2_good_beats_v1_and_v2_bad_is_caught_by_serving_path(
    bundles: dict[str, ModelBundle],
) -> None:
    v1, good, bad = (bundles[v].metrics for v in ("v1", "v2-good", "v2-bad"))
    assert good["auc"] > v1["auc"]
    assert v1["error_rate"] == good["error_rate"] == 0.0
    assert 0.03 < bad["error_rate"] < 0.09  # the zero-enrolment students crash it
    assert bad["auc"] < good["auc"] - 0.02  # swapped features hurt ranking
    # same weights: the bug is in serving, not training
    assert bundles["v2-bad"].estimator is bundles["v2-good"].estimator


def test_v2_bad_raises_only_on_zero_enrolment(bundles: dict[str, ModelBundle]) -> None:
    row = generate(1, seed=0).rows()[0] | {"sem1_enrolled": 0.0, "sem1_approved": 0.0}
    assert bundles["v2-good"].predict_one(row) >= 0
    with pytest.raises(ZeroDivisionError):
        bundles["v2-bad"].predict_one(row)


def test_save_and_load_roundtrip(bundles: dict[str, ModelBundle], tmp_path: Path) -> None:
    save_all(bundles, tmp_path)
    row = generate(1, seed=5).rows()[0]
    loaded = load(tmp_path, "v2-good")
    assert loaded.predict_one(row) == pytest.approx(bundles["v2-good"].predict_one(row))
    with pytest.raises(ValueError, match="unknown model version"):
        load(tmp_path, "v3")
    (tmp_path / "v1.joblib").write_bytes((tmp_path / "v2-good.joblib").read_bytes())
    with pytest.raises(ValueError, match="does not contain"):
        load(tmp_path, "v1")


def test_load_uci_maps_columns_and_target(tmp_path: Path) -> None:
    header = [*UCI_COLUMNS.values(), "Marital status", "Target"]
    rows = [
        ["19", "1", "0", "1", "6", "6", "14.5", "1", "Graduate"],
        ["23", "0", "1", "0", "0", "0", "0", "1", "Dropout"],
    ]
    path = tmp_path / "data.csv"
    path.write_text("\n".join(";".join(r) for r in [header, *rows]) + "\n")
    data = load_uci(path)
    assert data.X.shape == (2, len(FEATURES))
    assert list(data.y) == [0, 1]
    assert data.rows()[0]["sem1_grade"] == 14.5
    path.write_text("Target;Age at enrollment\nDropout;20\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_uci(path)
