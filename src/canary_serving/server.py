"""FastAPI model server: one process serves one model version (set by MODEL_VERSION).

Endpoints
- POST /predict  dropout risk for one student
- GET  /healthz  liveness: the process answers (Kubernetes restarts the pod if not)
- GET  /readyz   readiness: the model is loaded (Kubernetes sends no traffic until it is)
- GET  /metrics  Prometheus text format

Liveness and readiness are separate on purpose: a pod whose model failed to load should get no
traffic, but restarting it in a loop would not fix a missing file and would hide the logs.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from . import model as model_lib
from .metrics import ServerMetrics, server_metrics

log = logging.getLogger("canary_serving.server")


class Student(BaseModel):
    """First-semester record of one student (same fields as the UCI dataset subset)."""

    age_at_enrollment: float = Field(ge=15, le=80)
    scholarship_holder: int = Field(ge=0, le=1)
    debtor: int = Field(ge=0, le=1)
    tuition_up_to_date: int = Field(ge=0, le=1)
    sem1_enrolled: int = Field(ge=0, le=30)
    sem1_approved: int = Field(ge=0, le=30)
    sem1_grade: float = Field(ge=0, le=20)


class Prediction(BaseModel):
    model_version: str
    dropout_risk: float
    high_risk: bool


def create_app(
    bundle: model_lib.ModelBundle | None,
    version: str,
    risk_threshold: float = 0.5,
    metrics: ServerMetrics | None = None,
) -> FastAPI:
    """Build the app around an already loaded bundle (None = model failed to load)."""
    m = metrics or server_metrics()
    app = FastAPI(title="Student dropout risk", version=version)

    @app.middleware("http")
    async def record(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path != "/predict":
            return await call_next(request)
        start = time.perf_counter()
        response = await call_next(request)
        m.latency.labels(version).observe(time.perf_counter() - start)
        m.requests.labels(version, str(response.status_code)).inc()
        return response

    @app.post("/predict", response_model=Prediction)
    def predict(student: Student) -> Prediction:
        if bundle is None:
            raise HTTPException(503, "model not loaded")
        try:
            score = bundle.predict_one(student.model_dump())
        except Exception as exc:  # any model bug must become a counted 500, not a crash
            log.exception("prediction failed")
            raise HTTPException(500, f"prediction failed: {type(exc).__name__}") from exc
        m.score.labels(version).observe(score)
        return Prediction(
            model_version=version, dropout_risk=round(score, 4), high_risk=score >= risk_threshold
        )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/readyz")
    def readyz(response: Response) -> dict[str, str]:
        if bundle is None:
            response.status_code = 503
            return {"status": "model not loaded", "model_version": version}
        return {"status": "ready", "model_version": version}

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        return Response(generate_latest(m.registry), media_type=CONTENT_TYPE_LATEST)

    return app


def app_from_env() -> FastAPI:
    """Entry point for uvicorn in the container: settings come from env (ConfigMap)."""
    version = os.environ.get("MODEL_VERSION", "v1")
    model_dir = Path(os.environ.get("MODEL_DIR", "models"))
    threshold = float(os.environ.get("RISK_THRESHOLD", "0.5"))
    try:
        bundle: model_lib.ModelBundle | None = model_lib.load(model_dir, version)
    except (OSError, ValueError) as exc:
        log.error("could not load model %s from %s: %s", version, model_dir, exc)
        bundle = None
    return create_app(bundle, version, threshold)
