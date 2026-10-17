"""Population Stability Index (PSI): how far a distribution has moved from a reference.

PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%).
Rule of thumb used here (common in credit scoring): < 0.1 stable, 0.1-0.25 watch, > 0.25 act.

Used twice:
- the canary controller compares canary vs stable *prediction* histograms (from /metrics);
- `canary drift` compares current *inputs and outputs* with the training reference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .data import FEATURES

EPS = 1e-4  # floor for empty bins, so ln(0) never happens; small enough not to dominate
WARN, ALERT = 0.1, 0.25


def psi(expected: Sequence[float] | np.ndarray, actual: Sequence[float] | np.ndarray) -> float:
    """PSI between two vectors of bin counts (or proportions) over the same bins."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    if e.shape != a.shape or e.ndim != 1 or len(e) == 0:
        raise ValueError("expected and actual must be 1-D and the same length")
    if e.sum() <= 0 or a.sum() <= 0:
        raise ValueError("both distributions need at least one observation")
    e = np.clip(e / e.sum(), EPS, None)
    a = np.clip(a / a.sum(), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def quantile_edges(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Inner bin edges at observed reference deciles. Duplicates collapse (0/1 column: 2 bins)."""
    qs = np.quantile(reference, np.linspace(0, 1, n_bins + 1)[1:-1], method="lower")
    return np.unique(qs)


def bin_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Counts per bin; bins are (-inf, e0], (e0, e1], ..., (e_last, inf)."""
    idx = np.searchsorted(edges, values, side="left")
    return np.bincount(idx, minlength=len(edges) + 1).astype(float)


def psi_vs_reference(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    edges = quantile_edges(reference, n_bins)
    return psi(bin_counts(reference, edges), bin_counts(current, edges))


def status(value: float) -> str:
    return "alert" if value > ALERT else "warn" if value > WARN else "ok"


@dataclass(frozen=True)
class DriftRow:
    name: str
    psi: float

    @property
    def status(self) -> str:
        return status(self.psi)


def drift_report(
    reference_X: np.ndarray,
    reference_scores: np.ndarray,
    current_X: np.ndarray,
    current_scores: np.ndarray,
) -> list[DriftRow]:
    """PSI per input feature plus the model output, against the training reference."""
    rows = [
        DriftRow(name, psi_vs_reference(reference_X[:, i], current_X[:, i]))
        for i, name in enumerate(FEATURES)
    ]
    rows.append(DriftRow("prediction_score", psi_vs_reference(reference_scores, current_scores)))
    return rows
