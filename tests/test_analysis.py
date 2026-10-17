"""Decision logic: wait / pass / fail, threshold boundaries, histogram maths."""

from __future__ import annotations

import math

import pytest

from canary_serving.analysis import (
    Action,
    Thresholds,
    analyse,
    histogram_quantile,
    latency_regressed,
    score_psi,
)
from canary_serving.metrics import Snapshot

INF = math.inf


def snap(
    ok: float,
    errors: float = 0.0,
    scores: dict[float, float] | None = None,
    latency: dict[float, float] | None = None,
    client_errors: float = 0.0,
) -> Snapshot:
    """A snapshot with `ok` 200s, `errors` 500s; scores spread evenly unless given."""
    total = ok + errors + client_errors
    requests = {"200": ok, "500": errors, "422": client_errors}
    lat = latency or {0.005: total, 0.01: total, INF: total}
    if scores is None:
        scores = {round(0.1 * i, 1): ok * i / 10 for i in range(1, 11)}
        scores[INF] = ok
    return Snapshot(requests, lat, scores)


def test_histogram_quantile_interpolates_like_promql() -> None:
    buckets = {0.1: 50.0, 0.2: 90.0, 0.5: 100.0, INF: 100.0}
    # rank 95 is half-way between 90 (le 0.2) and 100 (le 0.5) -> 0.35
    assert histogram_quantile(0.95, buckets) == pytest.approx(0.35)
    assert histogram_quantile(0.5, buckets) == pytest.approx(0.1)


def test_histogram_quantile_edge_cases() -> None:
    assert math.isnan(histogram_quantile(0.95, {}))
    assert math.isnan(histogram_quantile(0.95, {0.1: 0.0, INF: 0.0}))
    # everything slower than the largest bucket: the best we can say is "at least 0.5"
    assert histogram_quantile(0.95, {0.1: 0.0, 0.5: 0.0, INF: 10.0}) == 0.5
    with pytest.raises(ValueError):
        histogram_quantile(1.5, {0.1: 1.0})


def test_insufficient_traffic_waits_even_if_canary_looks_terrible() -> None:
    limits = Thresholds(min_requests=200)
    verdict = analyse(snap(1000), snap(100, errors=99), limits)
    assert verdict.action is Action.WAIT
    assert "need 200" in verdict.reasons[0]


def test_min_requests_boundary() -> None:
    limits = Thresholds(min_requests=200)
    assert analyse(snap(1000), snap(199), limits).action is Action.WAIT
    assert analyse(snap(1000), snap(200), limits).action is Action.PASS
    assert analyse(snap(199), snap(1000), limits).action is Action.WAIT  # stable side too


def test_error_rate_boundary_is_inclusive() -> None:
    limits = Thresholds(min_requests=100, error_rate_margin=0.01)
    stable = snap(1000)
    assert analyse(stable, snap(990, errors=10), limits).action is Action.PASS  # exactly 1.00%
    verdict = analyse(stable, snap(989, errors=11), limits)  # 1.10%
    assert verdict.action is Action.FAIL
    assert "error rate" in verdict.reasons[0]


def test_error_margin_is_relative_to_stable() -> None:
    limits = Thresholds(min_requests=100)
    assert analyse(snap(950, errors=50), snap(945, errors=55), limits).action is Action.PASS


def test_client_errors_do_not_count_against_the_canary() -> None:
    limits = Thresholds(min_requests=100)
    assert analyse(snap(1000), snap(500, client_errors=500), limits).action is Action.PASS


def test_psi_boundary_is_inclusive() -> None:
    stable = snap(1000)
    shifted = {
        0.1: 0.0,
        0.2: 0.0,
        0.3: 0.0,
        0.4: 0.0,
        0.5: 0.0,
        0.6: 300.0,
        0.7: 500.0,
        0.8: 700.0,
        0.9: 900.0,
        1.0: 1000.0,
        INF: 1000.0,
    }
    canary = snap(1000, scores=shifted)
    value = score_psi(stable, canary)
    assert value > 0.2
    assert analyse(stable, canary, Thresholds(max_score_psi=value)).action is Action.PASS
    assert analyse(stable, canary, Thresholds(max_score_psi=value - 1e-9)).action is Action.FAIL


def test_all_requests_failing_has_no_scores_but_still_fails() -> None:
    canary = snap(0, errors=300, scores={})
    assert score_psi(snap(1000), canary) == 0.0
    assert analyse(snap(1000), canary).action is Action.FAIL


@pytest.mark.parametrize(
    ("stable", "canary", "regressed"),
    [
        (0.001, 0.002, False),  # 2x slower but only 1 ms: noise, not a regression
        (0.050, 0.070, False),  # 20 ms slower but only 1.4x
        (0.050, 0.100, True),  # 2x and 50 ms
        (0.050, 0.075, False),  # exactly 1.5x and exactly 25 ms: on the limit passes
        (math.nan, 0.5, False),  # no stable latency data: cannot judge
    ],
)
def test_latency_needs_relative_and_absolute_gap(
    stable: float, canary: float, regressed: bool
) -> None:
    assert latency_regressed(stable, canary, Thresholds()) is regressed


def test_slow_canary_fails_with_latency_reason() -> None:
    slow = {0.05: 0.0, 0.25: 0.0, 0.5: 1000.0, INF: 1000.0}
    verdict = analyse(snap(1000), snap(1000, latency=slow), Thresholds())
    assert verdict.action is Action.FAIL
    assert verdict.reasons == (verdict.reasons[0],) and "latency" in verdict.reasons[0]
