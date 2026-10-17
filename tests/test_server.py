"""Model server contract: endpoints, status codes, and what /metrics exports."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from canary_serving.metrics import parse
from canary_serving.model import ModelBundle
from canary_serving.server import app_from_env, create_app

STUDENT = {
    "age_at_enrollment": 19,
    "scholarship_holder": 0,
    "debtor": 0,
    "tuition_up_to_date": 1,
    "sem1_enrolled": 6,
    "sem1_approved": 5,
    "sem1_grade": 13.2,
}


def client(bundles: dict[str, ModelBundle], version: str) -> TestClient:
    return TestClient(create_app(bundles[version], version), raise_server_exceptions=False)


def test_predict_contract(bundles: dict[str, ModelBundle]) -> None:
    c = client(bundles, "v2-good")
    body = c.post("/predict", json=STUDENT).json()
    assert set(body) == {"model_version", "dropout_risk", "high_risk"}
    assert body["model_version"] == "v2-good"
    assert 0 <= body["dropout_risk"] <= 1
    assert body["high_risk"] is (body["dropout_risk"] >= 0.5)


def test_struggling_student_scores_higher(bundles: dict[str, ModelBundle]) -> None:
    c = client(bundles, "v1")
    good = c.post("/predict", json=STUDENT).json()["dropout_risk"]
    risky = {**STUDENT, "sem1_approved": 1, "sem1_grade": 10.5, "tuition_up_to_date": 0}
    assert c.post("/predict", json=risky).json()["dropout_risk"] > good


def test_probes(bundles: dict[str, ModelBundle]) -> None:
    c = client(bundles, "v1")
    assert c.get("/healthz").status_code == 200
    assert c.get("/readyz").json() == {"status": "ready", "model_version": "v1"}


def test_not_ready_without_model_but_still_alive() -> None:
    c = TestClient(create_app(None, "v1"))
    assert c.get("/healthz").status_code == 200  # do not restart: a restart won't fix it
    assert c.get("/readyz").status_code == 503  # but send no traffic
    assert c.post("/predict", json=STUDENT).status_code == 503


def test_app_from_env_with_missing_model_dir_is_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "nothing-here"))
    monkeypatch.setenv("MODEL_VERSION", "v2-good")
    assert TestClient(app_from_env()).get("/readyz").status_code == 503


def test_metrics_export_status_version_latency_and_scores(bundles: dict[str, ModelBundle]) -> None:
    c = client(bundles, "v2-bad")
    c.post("/predict", json=STUDENT)
    c.post("/predict", json={**STUDENT, "sem1_enrolled": 0, "sem1_approved": 0, "sem1_grade": 0})
    c.post("/predict", json={**STUDENT, "debtor": 7})  # invalid -> 422
    text = c.get("/metrics").text
    assert 'canary_requests_total{model_version="v2-bad",status="200"} 1.0' in text
    assert 'canary_requests_total{model_version="v2-bad",status="500"} 1.0' in text
    assert 'canary_requests_total{model_version="v2-bad",status="422"} 1.0' in text
    assert "canary_request_latency_seconds_bucket" in text
    snap = parse(text)
    assert snap.latency[float("inf")] == 3  # latency counts every request
    assert snap.score[float("inf")] == 1  # scores only for successful predictions


def test_metrics_and_probes_are_not_counted_as_predictions(bundles: dict[str, ModelBundle]) -> None:
    c = client(bundles, "v1")
    c.get("/healthz")
    c.get("/metrics")
    assert parse(c.get("/metrics").text).total == 0
