"""Local simulation of a rollout: no cluster, no network, same controller and analysis code.

Two in-process FastAPI servers (stable and canary) stand in for the two Deployments. A router
sends each synthetic request to the canary with probability canary_replicas / total, which is
what kube-proxy does with a replica-ratio split. Metrics are scraped from each server's real
/metrics endpoint and parsed exactly like in the cluster.

Latency numbers here are in-process and say nothing about a real cluster; the latency check is
exercised, but errors and prediction drift are what this simulation is meant to show.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from .analysis import Thresholds
from .metrics import PodSnapshots, parse
from .model import ModelBundle
from .rollout import DEFAULT_STEPS, RolloutResult, replica_split, run_rollout
from .server import create_app
from .traffic import students

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # starlette's httpx deprecation notice, not our code
    from fastapi.testclient import TestClient


class LocalBackend:
    def __init__(
        self,
        bundles: dict[str, ModelBundle],
        stable_version: str,
        canary_version: str,
        total_replicas: int = 10,
        requests_per_window: int = 500,
        seed: int = 0,
    ) -> None:
        self.bundles, self.total, self.per_window = bundles, total_replicas, requests_per_window
        self.stable_version, self.canary_version = stable_version, canary_version
        self.stable = self._client(stable_version)
        self.canary = self._client(canary_version)
        self.stable_replicas, self.canary_replicas = total_replicas, 0
        self.rng = np.random.default_rng(seed)
        self.traffic = students(seed=seed + 1000)
        self.sent = {"stable": 0, "canary": 0}

    def _client(self, version: str) -> TestClient:
        return TestClient(create_app(self.bundles[version], version), raise_server_exceptions=False)

    def set_canary_weight(self, percent: int) -> None:
        self.stable_replicas, self.canary_replicas = replica_split(self.total, percent)

    def wait_window(self) -> None:
        share = self.canary_replicas / self.total
        for _ in range(self.per_window):
            track = "canary" if self.rng.random() < share else "stable"
            client = self.canary if track == "canary" else self.stable
            client.post("/predict", json=next(self.traffic))
            self.sent[track] += 1

    def _scrape(self, client: TestClient, replicas: int, name: str) -> PodSnapshots:
        return {name: parse(client.get("/metrics").text)} if replicas > 0 else {}

    def scrape(self) -> tuple[PodSnapshots, PodSnapshots]:
        return (
            self._scrape(self.stable, self.stable_replicas, f"stable-{self.stable_version}"),
            self._scrape(self.canary, self.canary_replicas, f"canary-{self.canary_version}"),
        )

    def promote(self) -> None:
        self.stable_version = self.canary_version
        self.stable = self._client(self.canary_version)
        self.stable_replicas, self.canary_replicas = self.total, 0

    def rollback(self) -> None:
        self.stable_replicas, self.canary_replicas = self.total, 0


def simulate(
    bundles: dict[str, ModelBundle],
    canary_version: str,
    stable_version: str = "v1",
    steps: tuple[int, ...] = DEFAULT_STEPS,
    limits: Thresholds | None = None,
    requests_per_window: int = 500,
    seed: int = 0,
) -> tuple[RolloutResult, LocalBackend]:
    backend = LocalBackend(
        bundles, stable_version, canary_version, requests_per_window=requests_per_window, seed=seed
    )
    server_log = logging.getLogger("canary_serving.server")
    level = server_log.level
    server_log.setLevel(logging.CRITICAL)  # v2-bad would log hundreds of expected tracebacks
    try:
        return run_rollout(backend, steps, limits), backend
    finally:
        server_log.setLevel(level)
