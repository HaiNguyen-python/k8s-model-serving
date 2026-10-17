"""Canary controller: 10% -> 50% -> 100%, check metrics at each step, promote or roll back.

The loop knows nothing about Kubernetes. It talks to a `Backend` (kubectl in `kube.py`, in-process
servers in `simulate.py`), so the same code runs in the kind e2e test and in the local simulation,
and the decision logic in `analysis.py` stays pure and unit-tested.

Baselines: the canary is judged on its traffic *in the current step*; stable is judged on all its
traffic *since the rollout started*. That keeps a big stable baseline even at the 100% step, when
stable has no pods left.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from .analysis import Action, Thresholds, Verdict, analyse
from .metrics import PodSnapshots, Snapshot, window_delta

DEFAULT_STEPS: tuple[int, ...] = (10, 50, 100)


class Backend(Protocol):
    def set_canary_weight(self, percent: int) -> None: ...
    def wait_window(self) -> None: ...
    def scrape(self) -> tuple[PodSnapshots, PodSnapshots]: ...  # (stable, canary)
    def promote(self) -> None: ...
    def rollback(self) -> None: ...


class Outcome(StrEnum):
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class WindowRecord:
    step_percent: int
    window: int  # 1-based index inside the step
    verdict: Verdict


@dataclass
class RolloutResult:
    outcome: Outcome = Outcome.PROMOTED
    reason: str = ""
    windows: list[WindowRecord] = field(default_factory=list)


def replica_split(total: int, canary_percent: int) -> tuple[int, int]:
    """(stable, canary) replicas for a traffic share, with at least one canary pod above 0%.

    Replica-ratio canaries can only express multiples of 1/total. With 10 replicas, 10% is one
    pod; with 4 replicas, "10%" becomes 25%. This rounding is the main limit of the approach.
    """
    if total < 1 or not 0 <= canary_percent <= 100:
        raise ValueError("need total >= 1 and 0 <= percent <= 100")
    canary = round(total * canary_percent / 100)
    if canary_percent > 0:
        canary = max(1, canary)
    if canary_percent < 100:
        canary = min(canary, total - 1) if total > 1 else canary
    return total - canary, canary


def run_rollout(
    backend: Backend,
    steps: tuple[int, ...] = DEFAULT_STEPS,
    limits: Thresholds | None = None,
    max_windows_per_step: int = 10,
) -> RolloutResult:
    """Drive the rollout. Every exit promotes or rolls back: never a half-deployed canary."""
    result = RolloutResult()
    stable_total = Snapshot()
    for percent in steps:
        backend.set_canary_weight(percent)
        prev_stable, prev_canary = backend.scrape()
        canary_total = Snapshot()
        verdict: Verdict | None = None
        for window in range(1, max_windows_per_step + 1):
            backend.wait_window()
            stable_now, canary_now = backend.scrape()
            stable_total = stable_total + window_delta(prev_stable, stable_now)
            canary_total = canary_total + window_delta(prev_canary, canary_now)
            prev_stable, prev_canary = stable_now, canary_now
            verdict = analyse(stable_total, canary_total, limits)
            result.windows.append(WindowRecord(percent, window, verdict))
            if verdict.action is not Action.WAIT:
                break
        if verdict is None or verdict.action is not Action.PASS:
            backend.rollback()
            result.outcome = Outcome.ROLLED_BACK
            why = "; ".join(verdict.reasons) if verdict else "no windows"
            if verdict is not None and verdict.action is Action.WAIT:
                why = f"not enough traffic after {max_windows_per_step} windows ({why})"
            result.reason = f"step {percent}%: {why}"
            return result
    backend.promote()
    result.reason = f"passed all steps {', '.join(f'{s}%' for s in steps)}"
    return result
