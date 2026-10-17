"""Controller loop, kubectl backend (with fakes) and the local in-process simulation."""

from __future__ import annotations

import math
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from canary_serving.analysis import Action, Thresholds
from canary_serving.kube import KubectlBackend
from canary_serving.metrics import PodSnapshots, Snapshot
from canary_serving.model import ModelBundle
from canary_serving.report import build, plot_checks, plot_timeline, results_markdown
from canary_serving.rollout import Outcome, replica_split, run_rollout
from canary_serving.simulate import simulate

INF = math.inf
LIMITS = Thresholds(min_requests=150)


def counts(ok: float, errors: float = 0.0) -> Snapshot:
    scores = {round(0.1 * i, 1): ok * i / 10 for i in range(1, 11)} | {INF: ok}
    lat = {0.005: ok + errors, INF: ok + errors}
    return Snapshot({"200": ok, "500": errors}, lat, scores)


class FakeBackend:
    """Each window adds `per_window` requests split by canary weight; canary errors at `bad`."""

    def __init__(self, per_window: int = 1000, bad: float = 0.0) -> None:
        self.per_window, self.bad = per_window, bad
        self.weight = 0
        self.stable_total = self.canary_ok = self.canary_err = 0.0
        self.calls: list[str] = []
        self.canary_pod = 0  # new pod name per step, like a scaled Deployment

    def set_canary_weight(self, percent: int) -> None:
        self.weight = percent
        self.calls.append(f"weight {percent}")
        self.canary_pod += 1
        self.canary_ok = self.canary_err = 0.0

    def wait_window(self) -> None:
        canary = self.per_window * self.weight / 100
        self.stable_total += self.per_window - canary
        self.canary_err += canary * self.bad
        self.canary_ok += canary * (1 - self.bad)

    def scrape(self) -> tuple[PodSnapshots, PodSnapshots]:
        stable = {"stable-0": counts(self.stable_total)} if self.weight < 100 else {}
        return stable, {f"canary-{self.canary_pod}": counts(self.canary_ok, self.canary_err)}

    def promote(self) -> None:
        self.calls.append("promote")

    def rollback(self) -> None:
        self.calls.append("rollback")


@pytest.mark.parametrize(
    ("total", "percent", "expected"),
    [
        (10, 10, (9, 1)),
        (10, 50, (5, 5)),
        (10, 100, (0, 10)),
        (10, 0, (10, 0)),
        (4, 10, (3, 1)),
        (10, 99, (1, 9)),
        (2, 1, (1, 1)),
    ],
)
def test_replica_split(total: int, percent: int, expected: tuple[int, int]) -> None:
    assert replica_split(total, percent) == expected


def test_replica_split_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        replica_split(0, 10)
    with pytest.raises(ValueError):
        replica_split(10, 120)


def test_healthy_canary_walks_all_steps_and_promotes() -> None:
    backend = FakeBackend()
    result = run_rollout(backend, limits=Thresholds(min_requests=200))
    assert result.outcome is Outcome.PROMOTED
    assert backend.calls == ["weight 10", "weight 50", "weight 100", "promote"]
    # at 10% the first window has 100 canary requests -> wait, the second has 200 -> pass
    assert [w.verdict.action.value for w in result.windows] == ["wait", "pass", "pass", "pass"]


def test_stable_baseline_survives_100_percent_step() -> None:
    result = run_rollout(FakeBackend(), limits=Thresholds(min_requests=200))
    last = result.windows[-1].verdict.comparison
    assert last.stable_requests == 900 + 900 + 500  # stable has no pods at 100%, history is kept


def test_failing_canary_rolls_back_at_first_step() -> None:
    backend = FakeBackend(bad=0.05)
    result = run_rollout(backend, limits=Thresholds(min_requests=200))
    assert result.outcome is Outcome.ROLLED_BACK
    assert backend.calls == ["weight 10", "rollback"]
    assert result.reason.startswith("step 10%: error rate")


def test_never_enough_traffic_rolls_back_instead_of_hanging() -> None:
    backend = FakeBackend(per_window=100)
    result = run_rollout(backend, limits=Thresholds(min_requests=200), max_windows_per_step=3)
    assert result.outcome is Outcome.ROLLED_BACK
    assert len(result.windows) == 3
    assert "not enough traffic after 3 windows" in result.reason


class FakeKubectl:
    def __init__(self, metrics_text: str, broken_pod: str = "") -> None:
        self.commands: list[list[str]] = []
        self.metrics_text, self.broken_pod = metrics_text, broken_pod

    def __call__(self, args: Sequence[str]) -> str:
        args = list(args)
        self.commands.append(args)
        if args[:2] == ["get", "--raw"]:
            if self.broken_pod and self.broken_pod in args[2]:
                raise subprocess.CalledProcessError(1, args)
            return self.metrics_text
        if "pods" in args:
            track = next(a for a in args if a.startswith("app=")).split("track=")[1]
            return f"{track}-a {track}-b"
        if "jsonpath" in " ".join(args) and "MODEL_VERSION" in " ".join(args):
            return "v2-good"
        return ""


METRICS = 'canary_requests_total{model_version="v1",status="200"} 5.0\n'


def test_kubectl_scales_canary_up_before_stable_down() -> None:
    fake = FakeKubectl(METRICS)
    KubectlBackend(total_replicas=10, runner=fake).set_canary_weight(10)
    scales = [c for c in fake.commands if "scale" in c]
    assert scales[0][-2:] == ["deployment/dropout-model-canary", "--replicas=1"]
    assert scales[1][-2:] == ["deployment/dropout-model-stable", "--replicas=9"]
    assert sum("rollout" in c for c in fake.commands) == 2  # waits for both


def test_kubectl_scrape_uses_pod_proxy_and_skips_unready_pods() -> None:
    fake = FakeKubectl(METRICS, broken_pod="canary-b")
    stable, canary = KubectlBackend(runner=fake).scrape()
    assert set(stable) == {"stable-a", "stable-b"} and set(canary) == {"canary-a"}
    assert stable["stable-a"].total == 5
    raw = [c[2] for c in fake.commands if c[:2] == ["get", "--raw"]]
    assert raw[0] == "/api/v1/namespaces/canary-demo/pods/stable-a:8000/proxy/metrics"


def test_kubectl_promote_copies_canary_version_then_scales() -> None:
    fake = FakeKubectl(METRICS)
    KubectlBackend(runner=fake).promote()
    set_env = next(c for c in fake.commands if "set" in c)
    assert set_env[-2:] == ["deployment/dropout-model-stable", "MODEL_VERSION=v2-good"]
    scales = [c[-1] for c in fake.commands if "scale" in c]
    assert scales == ["--replicas=10", "--replicas=0"]  # stable up first, then canary down


def test_kubectl_rollback_restores_stable_before_removing_canary() -> None:
    fake = FakeKubectl(METRICS)
    KubectlBackend(runner=fake, total_replicas=4).rollback()
    scales = [(c[-2], c[-1]) for c in fake.commands if "scale" in c]
    assert scales == [
        ("deployment/dropout-model-stable", "--replicas=4"),
        ("deployment/dropout-model-canary", "--replicas=0"),
    ]


def test_good_release_is_promoted(bundles: dict[str, ModelBundle]) -> None:
    result, backend = simulate(bundles, "v2-good", limits=LIMITS, requests_per_window=300, seed=1)
    assert result.outcome is Outcome.PROMOTED
    assert [w.step_percent for w in result.windows if w.verdict.action is Action.PASS] == [
        10,
        50,
        100,
    ]
    assert backend.stable_version == "v2-good"
    assert backend.canary_replicas == 0


def test_bad_release_is_rolled_back_at_10_percent(bundles: dict[str, ModelBundle]) -> None:
    result, backend = simulate(bundles, "v2-bad", limits=LIMITS, requests_per_window=300, seed=1)
    assert result.outcome is Outcome.ROLLED_BACK
    assert {w.step_percent for w in result.windows} == {10}
    assert backend.stable_version == "v1" and backend.canary_replicas == 0
    assert "error rate" in result.reason and "prediction PSI" in result.reason
    # the blast radius: only about 10% of requests ever reached the broken version
    assert backend.sent["canary"] / sum(backend.sent.values()) < 0.15


def test_report_writes_labelled_results_and_charts(
    bundles: dict[str, ModelBundle], tmp_path: Path
) -> None:
    report = build(seed=1, requests_per_window=300, bundles=bundles)
    md = results_markdown(report)
    assert "Synthetic data, local in-process simulation" in md
    assert "### v2-good: **promoted**" in md and "### v2-bad: **rolled back**" in md
    plot_timeline(report, tmp_path / "timeline.png")
    plot_checks(report, tmp_path / "checks.png")
    for name in ("timeline.png", "checks.png"):
        size = (tmp_path / name).stat().st_size
        assert 10_000 < size < 300_000
