"""PSI maths, the drift report, Prometheus text parsing and window deltas across pods."""

from __future__ import annotations

import math

import numpy as np
import pytest
from prometheus_client import generate_latest

from canary_serving.data import FEATURES, generate
from canary_serving.drift import bin_counts, drift_report, psi, quantile_edges, status
from canary_serving.metrics import Snapshot, parse, server_metrics, window_delta


def test_psi_matches_hand_calculation() -> None:
    expected, actual = [50, 50], [80, 20]
    hand = (0.8 - 0.5) * math.log(0.8 / 0.5) + (0.2 - 0.5) * math.log(0.2 / 0.5)
    assert psi(expected, actual) == pytest.approx(hand)
    assert psi(actual, expected) == pytest.approx(hand)  # PSI is symmetric


def test_psi_is_zero_for_same_shape_and_ignores_scale() -> None:
    assert psi([1, 2, 3], [10, 20, 30]) == pytest.approx(0.0)


def test_empty_bin_gives_finite_large_psi() -> None:
    value = psi([50, 50], [100, 0])
    assert math.isfinite(value) and value > 1


@pytest.mark.parametrize(("e", "a"), [([1, 2], [1, 2, 3]), ([], []), ([0, 0], [1, 1])])
def test_psi_rejects_invalid_inputs(e: list[float], a: list[float]) -> None:
    with pytest.raises(ValueError):
        psi(e, a)


def test_binary_column_collapses_to_two_bins_and_edges_go_left() -> None:
    ref = np.array([0.0] * 70 + [1.0] * 30)
    edges = quantile_edges(ref)
    assert list(edges) == [0.0, 1.0]
    assert list(bin_counts(ref, edges)) == [70.0, 30.0, 0.0]


def test_status_thresholds() -> None:
    assert [status(v) for v in (0.1, 0.1001, 0.25, 0.2501)] == ["ok", "warn", "warn", "alert"]


def test_drift_report_quiet_on_same_population_loud_on_shifted_cohort() -> None:
    ref = generate(3000, seed=1)
    scores_ref = ref.X[:, FEATURES.index("sem1_grade")] / 20  # any score-like column works
    same, shifted = generate(3000, seed=2), generate(3000, seed=2, shift=1.0)
    quiet = drift_report(ref.X, scores_ref, same.X, same.X[:, 6] / 20)
    loud = {r.name: r for r in drift_report(ref.X, scores_ref, shifted.X, shifted.X[:, 6] / 20)}
    assert all(r.status == "ok" for r in quiet)
    assert loud["sem1_grade"].status == "alert"
    assert loud["sem1_enrolled"].status == "ok"  # enrolment was not shifted
    assert len(loud) == len(FEATURES) + 1


def _exported(ok: int, errors: int, version: str = "v1") -> Snapshot:
    m = server_metrics()
    for _ in range(ok):
        m.requests.labels(version, "200").inc()
        m.latency.labels(version).observe(0.003)
        m.score.labels(version).observe(0.35)
    for _ in range(errors):
        m.requests.labels(version, "500").inc()
        m.latency.labels(version).observe(0.2)
    return parse(generate_latest(m.registry).decode())


def test_parse_reads_counts_and_cumulative_buckets() -> None:
    s = _exported(ok=8, errors=2)
    assert s.requests == {"200": 8.0, "500": 2.0}
    assert s.total == 10 and s.server_errors == 2
    assert s.latency[0.005] == 8 and s.latency[0.25] == 10 and s.latency[math.inf] == 10
    assert s.score[0.3] == 0 and s.score[0.4] == 8


def test_window_delta_handles_new_gone_and_restarted_pods() -> None:
    a1, a2 = Snapshot({"200": 10.0}), Snapshot({"200": 25.0})
    before = {
        "pod-a": a1,
        "pod-gone": Snapshot({"200": 99.0}),
        "pod-restart": Snapshot({"200": 50.0}),
    }
    after = {"pod-a": a2, "pod-new": Snapshot({"200": 7.0}), "pod-restart": Snapshot({"200": 3.0})}
    delta = window_delta(before, after)
    # 15 from pod-a + 7 from the new pod + 3 since the restart (not -47); nothing from the gone pod
    assert delta.requests["200"] == 25.0


def test_snapshots_add_up() -> None:
    total = _exported(3, 1) + _exported(2, 0, version="v2-good")
    assert total.requests == {"200": 5.0, "500": 1.0}
