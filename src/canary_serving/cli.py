"""Command line: canary train | serve | traffic | rollout | drift | report.

Exit codes of `canary rollout`: 0 promoted, 3 rolled back (so CI and scripts can assert on it).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

from .analysis import Thresholds
from .rollout import Outcome, RolloutResult

ROLLED_BACK_EXIT = 3


def _steps(text: str) -> tuple[int, ...]:
    steps = tuple(int(s) for s in text.split(","))
    if not steps or steps[-1] != 100 or list(steps) != sorted(steps) or steps[0] <= 0:
        raise argparse.ArgumentTypeError("steps must increase and end at 100, e.g. 10,50,100")
    return steps


def cmd_train(args: argparse.Namespace) -> int:
    from .model import save_all, train_all

    bundles = train_all(args.seed)
    for path in save_all(bundles, Path(args.out)):
        print(f"wrote {path}")
    for version, bundle in bundles.items():
        print(f"{version}: {json.dumps({k: round(v, 4) for k, v in bundle.metrics.items()})}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    if args.version:
        os.environ["MODEL_VERSION"] = args.version
    os.environ.setdefault("MODEL_DIR", args.model_dir)
    uvicorn.run(
        "canary_serving.server:app_from_env",
        factory=True,
        host=args.host,
        port=args.port,
        access_log=False,
    )
    return 0


def cmd_traffic(args: argparse.Namespace) -> int:
    from .traffic import run

    counts = run(args.url, args.rate, args.total, args.seed)
    print(json.dumps({str(k): v for k, v in sorted(counts.items())}))
    return 0


def _print_result(result: RolloutResult) -> int:
    for w in result.windows:
        c = w.verdict.comparison
        print(
            json.dumps(
                {
                    "step": w.step_percent,
                    "window": w.window,
                    "decision": w.verdict.action.value,
                    "stable_requests": c.stable_requests,
                    "canary_requests": c.canary_requests,
                    "canary_error_rate": round(c.canary_error_rate, 4),
                    "stable_error_rate": round(c.stable_error_rate, 4),
                    "score_psi": round(c.score_psi, 4),
                    "reasons": list(w.verdict.reasons),
                }
            )
        )
    print(f"outcome: {result.outcome.value} ({result.reason})")
    return 0 if result.outcome is Outcome.PROMOTED else ROLLED_BACK_EXIT


def cmd_rollout(args: argparse.Namespace) -> int:
    from .rollout import run_rollout

    limits = Thresholds(min_requests=args.min_requests, max_score_psi=args.max_psi)
    if args.local:
        from .model import train_all
        from .simulate import simulate

        result, _ = simulate(train_all(), args.local, steps=args.steps, limits=limits)
        return _print_result(result)

    from .kube import KubectlBackend

    backend = KubectlBackend(
        namespace=args.namespace,
        total_replicas=args.replicas,
        window_seconds=args.window_seconds,
    )
    return _print_result(run_rollout(backend, args.steps, limits, args.max_windows))


def cmd_drift(args: argparse.Namespace) -> int:
    from .data import generate, load_uci
    from .drift import drift_report
    from .model import load

    bundle = load(Path(args.model_dir), args.version)
    current = load_uci(Path(args.input)) if args.input else generate(args.n, args.seed, args.shift)
    scores = np.array([bundle.predict_one(row) for row in current.rows()])
    rows = drift_report(bundle.reference_X, bundle.reference_scores, current.X, scores)
    print(f"{'feature':<22}{'psi':>8}  status")
    for row in rows:
        print(f"{row.name:<22}{row.psi:>8.3f}  {row.status}")
    alerts = [r.name for r in rows if r.status == "alert"]
    return 1 if alerts and args.fail_on_alert else 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import write

    r = write(Path(args.out), args.seed)
    for version, res in r.rollouts.items():
        print(f"{version}: {res.outcome.value} ({res.reason})")
    print(f"wrote {Path(args.out) / 'results.md'} and charts in {Path(args.out) / 'img'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="canary", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train v1, v2-good, v2-bad on synthetic data")
    t.add_argument("--out", default="models")
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("serve", help="run the model server")
    s.add_argument("--version", default=None, help="overrides MODEL_VERSION")
    s.add_argument("--model-dir", default="models")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    g = sub.add_parser("traffic", help="send synthetic requests to a server or Service")
    g.add_argument("--url", required=True)
    g.add_argument("--rate", type=float, default=40.0, help="requests per second")
    g.add_argument("--total", type=int, default=None, help="stop after N requests")
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_traffic)

    r = sub.add_parser("rollout", help="canary rollout with automatic promote/rollback")
    r.add_argument("--local", choices=["v2-good", "v2-bad"], help="in-process simulation")
    r.add_argument("--namespace", default="canary-demo")
    r.add_argument("--replicas", type=int, default=10)
    r.add_argument("--window-seconds", type=float, default=15.0)
    r.add_argument("--max-windows", type=int, default=10)
    r.add_argument("--steps", type=_steps, default=(10, 50, 100))
    r.add_argument("--min-requests", type=int, default=Thresholds.min_requests)
    r.add_argument("--max-psi", type=float, default=Thresholds.max_score_psi)
    r.set_defaults(func=cmd_rollout)

    d = sub.add_parser("drift", help="input/output PSI vs the training reference")
    d.add_argument("--version", default="v1", choices=["v1", "v2-good"])
    d.add_argument("--model-dir", default="models")
    d.add_argument("--input", help="UCI CSV file; default: a synthetic batch")
    d.add_argument("--n", type=int, default=2000)
    d.add_argument("--shift", type=float, default=0.0, help="synthetic cohort shift")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--fail-on-alert", action="store_true")
    d.set_defaults(func=cmd_drift)

    rep = sub.add_parser("report", help="local simulation -> docs/results.md + charts")
    rep.add_argument("--out", default="docs")
    rep.add_argument("--seed", type=int, default=42)
    rep.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("canary_serving").setLevel(logging.INFO)
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
