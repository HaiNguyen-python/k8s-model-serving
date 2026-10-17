"""CLI wiring and the traffic generator (real HTTP against a tiny local server)."""

from __future__ import annotations

import argparse
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from canary_serving import cli, model, traffic
from canary_serving.model import ModelBundle, save_all


@pytest.fixture(scope="module")
def model_dir(bundles: dict[str, ModelBundle], tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("models")
    save_all(bundles, path)
    return path


def test_drift_command_exit_code_follows_alerts(
    model_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["drift", "--model-dir", str(model_dir), "--n", "800", "--fail-on-alert"]
    assert cli.main(base) == 0
    assert cli.main([*base, "--shift", "1.0"]) == 1
    out = capsys.readouterr().out
    assert "prediction_score" in out and "alert" in out


def test_local_rollout_exit_codes(
    bundles: dict[str, ModelBundle], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model, "train_all", lambda: bundles)
    args = ["rollout", "--min-requests", "100"]
    assert cli.main([*args, "--local", "v2-good", "--steps", "50,100"]) == 0
    assert cli.main([*args, "--local", "v2-bad"]) == cli.ROLLED_BACK_EXIT


def test_steps_must_increase_and_end_at_100() -> None:
    assert cli._steps("10,50,100") == (10, 50, 100)
    for bad in ("10,50", "50,10,100", "0,100"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli._steps(bad)


def test_serve_passes_settings_to_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: seen.update(app=app, **kw))
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    assert cli.main(["serve", "--version", "v2-good", "--port", "9001"]) == 0
    assert seen["app"] == "canary_serving.server:app_from_env" and seen["factory"] is True
    assert seen["port"] == 9001
    assert os.environ["MODEL_VERSION"] == "v2-good"


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        length = int(self.headers["content-length"])
        body = self.rfile.read(length)
        self.send_response(500 if b'"sem1_enrolled": 0.0' in body else 200)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture
def server_url() -> Iterator[str]:
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


def test_traffic_counts_statuses_over_real_http(server_url: str) -> None:
    counts = traffic.run(server_url, rate=0, total=200, seed=3)
    assert sum(counts.values()) == 200
    assert 0 < counts[500] < 30  # the zero-enrolment students, about 6%


def test_unreachable_server_is_status_zero_not_a_crash() -> None:
    assert traffic.post_json("http://127.0.0.1:9/predict", {"x": 1.0}, timeout=0.5) == 0


def test_traffic_paces_requests() -> None:
    sleeps: list[float] = []
    traffic.run("http://unused", rate=10, total=5, send=lambda url, p: 200, sleep=sleeps.append)
    assert len(sleeps) == 5 and all(0.09 < s <= 0.1 for s in sleeps)
