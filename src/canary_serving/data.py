"""Student records: a seeded synthetic generator and a loader for the real UCI dataset.

The schema mirrors seven columns of the UCI dataset "Predict students' dropout and academic
success" (Realinho et al., 2022, CC BY 4.0), so the same code can train on the real file.

The synthetic generator plants three effects on purpose, so the model comparison is explainable:
- the share of first-semester units a student passed (approved / enrolled) drives dropout in a
  non-linear way, so a model that sees this ratio (v2) beats one that only sees raw counts (v1);
- about 6% of students enrolled in zero units. They are high risk, and they are also the inputs
  that crash the broken v2-bad release (division by zero);
- `--shift` in `generate` moves the population (lower ability, more debt) to demo input drift.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURES: tuple[str, ...] = (
    "age_at_enrollment",
    "scholarship_holder",
    "debtor",
    "tuition_up_to_date",
    "sem1_enrolled",
    "sem1_approved",
    "sem1_grade",
)

# Column names in the UCI CSV (semicolon separated) for each feature above.
UCI_COLUMNS: dict[str, str] = {
    "age_at_enrollment": "Age at enrollment",
    "scholarship_holder": "Scholarship holder",
    "debtor": "Debtor",
    "tuition_up_to_date": "Tuition fees up to date",
    "sem1_enrolled": "Curricular units 1st sem (enrolled)",
    "sem1_approved": "Curricular units 1st sem (approved)",
    "sem1_grade": "Curricular units 1st sem (grade)",
}


@dataclass(frozen=True)
class Dataset:
    """Feature matrix in `FEATURES` order plus the binary dropout label."""

    X: np.ndarray
    y: np.ndarray

    def rows(self) -> list[dict[str, float]]:
        """Records as dicts, the shape the API receives."""
        return [dict(zip(FEATURES, map(float, row), strict=True)) for row in self.X]


def generate(n: int, seed: int = 0, shift: float = 0.0) -> Dataset:
    """Synthetic students. `shift` > 0 lowers ability and raises debt (a cohort change)."""
    rng = np.random.default_rng(seed)
    ability = rng.normal(-shift, 1.0, n)
    age = np.clip(18 + rng.gamma(1.5, 2.5, n) + shift, 17, 50).round()
    scholarship = (rng.random(n) < 0.25 - 0.08 * shift).astype(float)
    debtor = (rng.random(n) < 0.11 + 0.10 * shift).astype(float)
    tuition_ok = (rng.random(n) < np.where(debtor == 1, 0.55, 0.93)).astype(float)

    enrolled = np.clip(rng.poisson(6, n), 1, 8).astype(float)
    enrolled[rng.random(n) < 0.06] = 0.0
    pass_prob = 1 / (1 + np.exp(-(0.9 + 1.4 * ability)))
    approved = rng.binomial(enrolled.astype(int), pass_prob).astype(float)
    grade = np.where(approved > 0, np.clip(12.5 + 1.6 * ability + rng.normal(0, 1, n), 10, 18.5), 0)
    grade = grade.round(1)

    ratio = np.divide(approved, enrolled, out=np.zeros(n), where=enrolled > 0)
    logit = (
        -2.2
        + 4.0 * (1 - ratio) ** 2
        + 1.3 * (1 - tuition_ok)
        + 0.6 * debtor
        - 0.7 * scholarship
        + 0.06 * (age - 20)
        - 0.12 * np.where(approved > 0, grade - 12.5, 0)
        + 1.5 * (enrolled == 0)
    )
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = np.column_stack([age, scholarship, debtor, tuition_ok, enrolled, approved, grade])
    return Dataset(X=X, y=y)


def load_uci(path: Path) -> Dataset:
    """Load the real UCI CSV. Label = 1 when Target is "Dropout" (Enrolled/Graduate = 0)."""
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        rows = [r for r in reader]
    missing = [col for col in [*UCI_COLUMNS.values(), "Target"] if rows and col not in rows[0]]
    if missing:
        raise ValueError(f"UCI file is missing columns: {missing}")
    X = np.array([[float(r[UCI_COLUMNS[f]]) for f in FEATURES] for r in rows], dtype=float)
    y = np.array([1 if r["Target"].strip() == "Dropout" else 0 for r in rows], dtype=int)
    return Dataset(X=X.reshape(-1, len(FEATURES)), y=y)
