"""Prometheus metrics: what the server exports, and how the controller reads it back.

The controller does not need a Prometheus server. It scrapes `/metrics` of every pod, parses the
text format into a `Snapshot`, and subtracts the previous scrape to get "what happened in this
window". Counters only go up, so a window is a difference of two snapshots.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeVar

from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_client.parser import text_string_to_metric_families

REQUESTS = "canary_requests"
LATENCY = "canary_request_latency_seconds"
SCORE = "canary_prediction_score"

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
SCORE_BUCKETS = tuple(round(0.1 * i, 1) for i in range(1, 11))  # 0.1 ... 1.0


@dataclass(frozen=True)
class ServerMetrics:
    """The three metrics of one server process, on their own registry (easy to test)."""

    registry: CollectorRegistry
    requests: Counter
    latency: Histogram
    score: Histogram


def server_metrics() -> ServerMetrics:
    registry = CollectorRegistry()
    return ServerMetrics(
        registry=registry,
        requests=Counter(
            REQUESTS,
            "Prediction requests by HTTP status and model version",
            ["model_version", "status"],
            registry=registry,
        ),
        latency=Histogram(
            LATENCY,
            "Prediction request latency",
            ["model_version"],
            buckets=LATENCY_BUCKETS,
            registry=registry,
        ),
        score=Histogram(
            SCORE,
            "Predicted dropout risk of successful requests",
            ["model_version"],
            buckets=SCORE_BUCKETS,
            registry=registry,
        ),
    )


Buckets = Mapping[float, float]  # upper bound "le" -> cumulative count
K = TypeVar("K", str, float)


def _combine(a: Mapping[K, float], b: Mapping[K, float], sign: int) -> dict[K, float]:
    keys = set(a) | set(b)
    # max(0, ...): a difference of counters is never negative traffic (resets: see window_delta).
    return {k: max(0.0, a.get(k, 0.0) + sign * b.get(k, 0.0)) for k in keys}


@dataclass(frozen=True)
class Snapshot:
    """Counter values of one pod (or a sum of pods) at one moment."""

    requests: Mapping[str, float] = field(default_factory=dict)  # status code -> count
    latency: Buckets = field(default_factory=dict)
    score: Buckets = field(default_factory=dict)

    def __add__(self, other: Snapshot) -> Snapshot:
        return Snapshot(
            _combine(self.requests, other.requests, 1),
            _combine(self.latency, other.latency, 1),
            _combine(self.score, other.score, 1),
        )

    def __sub__(self, other: Snapshot) -> Snapshot:
        return Snapshot(
            _combine(self.requests, other.requests, -1),
            _combine(self.latency, other.latency, -1),
            _combine(self.score, other.score, -1),
        )

    @property
    def total(self) -> float:
        return float(sum(self.requests.values()))

    @property
    def server_errors(self) -> float:
        return float(sum(v for k, v in self.requests.items() if k.startswith("5")))


def parse(text: str) -> Snapshot:
    """Parse Prometheus text format; sums over label sets so one pod = one snapshot."""
    requests: dict[str, float] = {}
    latency: dict[float, float] = {}
    score: dict[float, float] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name == f"{REQUESTS}_total":
                status = sample.labels["status"]
                requests[status] = requests.get(status, 0.0) + sample.value
            elif sample.name in (f"{LATENCY}_bucket", f"{SCORE}_bucket"):
                target = latency if sample.name.startswith(LATENCY) else score
                le = float(sample.labels["le"])
                target[le] = target.get(le, 0.0) + sample.value
    return Snapshot(requests, latency, score)


PodSnapshots = Mapping[str, Snapshot]  # pod name -> snapshot


def window_delta(before: PodSnapshots, after: PodSnapshots) -> Snapshot:
    """Traffic between two scrapes, summed over pods.

    - A pod only in `after` is new (scaled up): all its counts belong to this window.
    - A pod only in `before` was scaled down: its last counts were taken by the previous window.
    - A pod whose total went down restarted: counters began again at 0, so all of `after`
      belongs to this window (the same rule PromQL's rate() uses for counter resets).
    """
    total = Snapshot()
    for pod, snap in after.items():
        prev = before.get(pod, Snapshot())
        total = total + (snap if snap.total < prev.total else snap - prev)
    return total
