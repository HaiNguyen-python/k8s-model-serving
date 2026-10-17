"""Kubernetes backend for the controller, using plain `kubectl` (no client library, no mesh).

- Traffic split: replica ratio. Stable and canary Deployments share the label `app=dropout-model`,
  the Service selects only that label, so traffic splits roughly by ready pod count.
- Metrics: each pod's /metrics is read through the API server pod proxy
  (`kubectl get --raw /api/v1/namespaces/<ns>/pods/<pod>:8000/proxy/metrics`). No Prometheus
  server and no port-forward are needed, and RBAC decides who may read them.
- Promote: copy the canary's MODEL_VERSION into the stable Deployment, scale stable back up,
  then scale the canary to 0. Rollback: scale stable up first, then the canary to 0, so capacity
  never drops.

All kubectl calls go through one `runner` function, so tests replace it with a fake.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Sequence

from .metrics import PodSnapshots, parse
from .rollout import replica_split

Runner = Callable[[Sequence[str]], str]


def run_kubectl(args: Sequence[str]) -> str:
    done = subprocess.run(["kubectl", *args], check=True, capture_output=True, text=True)
    return done.stdout


class KubectlBackend:
    def __init__(
        self,
        namespace: str = "canary-demo",
        app: str = "dropout-model",
        total_replicas: int = 10,
        window_seconds: float = 15.0,
        port: int = 8000,
        runner: Runner = run_kubectl,
        sleep: Callable[[float], None] = time.sleep,
        rollout_timeout: str = "180s",
    ) -> None:
        self.ns, self.app, self.total = namespace, app, total_replicas
        self.window_seconds, self.port = window_seconds, port
        self.kubectl, self.sleep, self.timeout = runner, sleep, rollout_timeout

    def deployment(self, track: str) -> str:
        return f"{self.app}-{track}"

    def _scale(self, track: str, replicas: int) -> None:
        self.kubectl(
            [
                "-n",
                self.ns,
                "scale",
                f"deployment/{self.deployment(track)}",
                f"--replicas={replicas}",
            ]
        )
        self.kubectl(
            [
                "-n",
                self.ns,
                "rollout",
                "status",
                f"deployment/{self.deployment(track)}",
                f"--timeout={self.timeout}",
            ]
        )

    def model_version(self, track: str) -> str:
        jsonpath = '{.spec.template.spec.containers[0].env[?(@.name=="MODEL_VERSION")].value}'
        out = self.kubectl(
            [
                "-n",
                self.ns,
                "get",
                f"deployment/{self.deployment(track)}",
                "-o",
                f"jsonpath={jsonpath}",
            ]
        )
        return out.strip()

    def set_canary_weight(self, percent: int) -> None:
        stable, canary = replica_split(self.total, percent)
        # Scale up first, then down: total ready capacity never goes below `total`.
        self._scale("canary", canary)
        self._scale("stable", stable)

    def wait_window(self) -> None:
        self.sleep(self.window_seconds)

    def _pods(self, track: str) -> list[str]:
        out = self.kubectl(
            [
                "-n",
                self.ns,
                "get",
                "pods",
                "-l",
                f"app={self.app},track={track}",
                "--field-selector=status.phase=Running",
                "-o",
                "jsonpath={.items[*].metadata.name}",
            ]
        )
        return out.split()

    def _scrape_track(self, track: str) -> PodSnapshots:
        snaps = {}
        for pod in self._pods(track):
            path = f"/api/v1/namespaces/{self.ns}/pods/{pod}:{self.port}/proxy/metrics"
            try:
                snaps[pod] = parse(self.kubectl(["get", "--raw", path]))
            except subprocess.CalledProcessError:
                continue  # pod is terminating or not ready yet; it reappears next window
        return snaps

    def scrape(self) -> tuple[PodSnapshots, PodSnapshots]:
        return self._scrape_track("stable"), self._scrape_track("canary")

    def promote(self) -> None:
        version = self.model_version("canary")
        stable = f"deployment/{self.deployment('stable')}"
        self.kubectl(["-n", self.ns, "set", "env", stable, f"MODEL_VERSION={version}"])
        self._scale("stable", self.total)
        self._scale("canary", 0)

    def rollback(self) -> None:
        self._scale("stable", self.total)
        self._scale("canary", 0)
