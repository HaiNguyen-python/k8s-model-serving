"""Traffic generator: sends synthetic students to the model Service.

It opens a new HTTP connection for every request on purpose. kube-proxy picks a backend pod per
*connection*, so a client with keep-alive would stick to one pod and break the replica-ratio
traffic split. Standard-library urllib is enough and keeps the image small.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator

from .data import generate

log = logging.getLogger("canary_serving.traffic")


def students(seed: int = 0, batch: int = 1000) -> Iterator[dict[str, float]]:
    """Endless stream of synthetic students (new seeded batch every `batch` rows)."""
    while True:
        yield from generate(batch, seed=seed).rows()
        seed += 1


def post_json(url: str, payload: dict[str, float], timeout: float = 5.0) -> int:
    """POST one request and return the HTTP status (0 when the server is unreachable)."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", "connection": "close"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return 0


def run(
    base_url: str,
    rate: float,
    total: int | None = None,
    seed: int = 0,
    send: Callable[[str, dict[str, float]], int] = post_json,
    sleep: Callable[[float], None] = time.sleep,
) -> Counter[int]:
    """Send `total` requests (forever if None) at about `rate` per second; count statuses."""
    url = base_url.rstrip("/") + "/predict"
    counts: Counter[int] = Counter()
    interval = 1.0 / rate if rate > 0 else 0.0
    for i, student in enumerate(students(seed)):
        if total is not None and i >= total:
            break
        started = time.monotonic()
        counts[send(url, student)] += 1
        if (i + 1) % 500 == 0:
            log.info("sent %d requests: %s", i + 1, dict(counts))
        sleep(max(0.0, interval - (time.monotonic() - started)))
    return counts
