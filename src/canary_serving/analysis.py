"""Canary analysis: compare canary vs stable metric snapshots and decide. Pure functions only.

Three possible answers for one rollout step:
- WAIT     not enough requests yet to judge (a decision on 12 requests is a coin flip);
- PASS     the canary is no worse than stable on every check, move to the next step;
- FAIL     at least one check failed, roll back.

Every threshold is "fail only if strictly worse than the limit", so a value exactly on the limit
passes. Tests pin this down, because off-by-one at a boundary is a classic rollout bug.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from .drift import psi
from .metrics import Buckets, Snapshot


class Action(StrEnum):
    WAIT = "wait"
    PASS = "pass"
    FAIL = "fail"


@dataclass(frozen=True)
class Thresholds:
    """Limits for one step. Defaults are explained in docs/decisions.md."""

    min_requests: int = 200  # per side; same-model PSI noise is ~0.05 here, ~0.36 at 45
    error_rate_margin: float = 0.01  # canary 5xx rate may exceed stable by at most 1 pp
    latency_ratio: float = 1.5  # canary p95 may be at most 1.5x stable p95 ...
    latency_slack_s: float = 0.025  # ... unless the absolute gap is under 25 ms
    max_score_psi: float = 0.2  # prediction distribution shift, canary vs stable


@dataclass(frozen=True)
class Comparison:
    """The numbers behind a verdict, kept for logs, reports and charts."""

    stable_requests: float
    canary_requests: float
    stable_error_rate: float
    canary_error_rate: float
    stable_p95_s: float
    canary_p95_s: float
    score_psi: float


@dataclass(frozen=True)
class Verdict:
    action: Action
    comparison: Comparison
    reasons: tuple[str, ...] = field(default_factory=tuple)


def histogram_quantile(q: float, buckets: Buckets) -> float:
    """Quantile from cumulative buckets, linear inside a bucket (same rule as PromQL).

    Returns nan when there are no observations. If the quantile falls in the +Inf bucket, the
    largest finite bound is returned, because nothing more precise is known.
    """
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    bounds = sorted(buckets)
    if not bounds or buckets[bounds[-1]] <= 0:
        return math.nan
    rank = q * buckets[bounds[-1]]
    lower, below = 0.0, 0.0
    for le in bounds:
        count = buckets[le]
        if count >= rank:
            if math.isinf(le):
                return lower
            if count == below:
                return le
            return lower + (le - lower) * (rank - below) / (count - below)
        lower, below = le, count
    return lower


def per_bucket(buckets: Buckets) -> list[float]:
    """Cumulative counts -> counts per bucket, in bound order."""
    bounds = sorted(buckets)
    cumulative = [buckets[b] for b in bounds]
    return [c - p for c, p in zip(cumulative, [0.0, *cumulative[:-1]], strict=True)]


def error_rate(snap: Snapshot) -> float:
    return snap.server_errors / snap.total if snap.total else 0.0


def score_psi(stable: Snapshot, canary: Snapshot) -> float:
    """PSI of canary vs stable prediction histograms; 0 when either side has no successes."""
    bounds = sorted(set(stable.score) | set(canary.score))
    s = per_bucket({b: stable.score.get(b, 0.0) for b in bounds})
    c = per_bucket({b: canary.score.get(b, 0.0) for b in bounds})
    if not bounds or sum(s) <= 0 or sum(c) <= 0:
        return 0.0
    return psi(s, c)


def compare(stable: Snapshot, canary: Snapshot) -> Comparison:
    return Comparison(
        stable_requests=stable.total,
        canary_requests=canary.total,
        stable_error_rate=error_rate(stable),
        canary_error_rate=error_rate(canary),
        stable_p95_s=histogram_quantile(0.95, stable.latency),
        canary_p95_s=histogram_quantile(0.95, canary.latency),
        score_psi=score_psi(stable, canary),
    )


def analyse(stable: Snapshot, canary: Snapshot, limits: Thresholds | None = None) -> Verdict:
    """Decide one step from the stable baseline and the canary traffic of this step."""
    t = limits or Thresholds()
    c = compare(stable, canary)
    if c.canary_requests < t.min_requests or c.stable_requests < t.min_requests:
        return Verdict(
            Action.WAIT,
            c,
            (
                f"need {t.min_requests} requests per side, have stable={c.stable_requests:.0f} "
                f"canary={c.canary_requests:.0f}",
            ),
        )
    reasons = []
    if c.canary_error_rate > c.stable_error_rate + t.error_rate_margin:
        reasons.append(
            f"error rate {c.canary_error_rate:.2%} > stable {c.stable_error_rate:.2%} "
            f"+ {t.error_rate_margin:.0%}"
        )
    if latency_regressed(c.stable_p95_s, c.canary_p95_s, t):
        reasons.append(f"p95 latency {c.canary_p95_s * 1000:.0f} ms vs {c.stable_p95_s * 1000:.0f}")
    if c.score_psi > t.max_score_psi:
        reasons.append(f"prediction PSI {c.score_psi:.3f} > {t.max_score_psi}")
    return Verdict(Action.FAIL if reasons else Action.PASS, c, tuple(reasons))


def latency_regressed(stable_p95: float, canary_p95: float, t: Thresholds) -> bool:
    """Both a relative and an absolute gap are needed, so 1 ms vs 2 ms is not a regression."""
    if math.isnan(canary_p95) or math.isnan(stable_p95):
        return False
    return canary_p95 > stable_p95 * t.latency_ratio and canary_p95 - stable_p95 > t.latency_slack_s
