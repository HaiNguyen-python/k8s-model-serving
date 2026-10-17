# How it works

This page follows one release from training to "promoted" or "rolled back", and explains the
important code on the way. File references point to `src/canary_serving/`.

## 1. Data: synthetic students with the UCI schema

`data.py → generate` creates seeded students with seven first-semester fields from the UCI
dropout dataset: age, scholarship, debtor, tuition paid, units enrolled, units approved, grade.
The dropout label comes from a logistic formula with three effects planted on purpose:

- the pass ratio `approved / enrolled` drives risk through `4 * (1 - ratio)^2`, a curve that a
  linear model on raw counts cannot fully copy;
- about 6% of students enrolled in zero units (high risk, and the input that breaks v2-bad);
- `shift > 0` lowers ability and raises debt, to demo input drift.

`load_uci` reads the real semicolon-separated UCI file and maps `Target == "Dropout"` to 1.

## 2. Three model versions

`model.py → train_all` trains two logistic regressions on the same seeded data:

| Version | Features | What it stands for |
| --- | --- | --- |
| v1 | the 7 raw columns | current production model |
| v2-good | 7 columns + pass ratio | the improvement (AUC 0.838 → 0.842, synthetic) |
| v2-bad | v2-good's weights, broken serving code | a bad release |

The important class is `ModelBundle`. It holds the estimator **and the serving-side feature
assembly** (`vector`). v2-bad is built with `dataclasses.replace` from v2-good: same estimator,
but `feature_order` has `sem1_grade` and `sem1_approved` swapped and `guard_zero_enrolled=False`.
That is how real serving bugs look: training was fine, the code around the model was not.

`evaluate` scores the test set *through `predict_one`*, the same path the API uses. That is why
the offline table shows v2-bad at AUC 0.798 with 6.1% failing requests. In a real team the
offline evaluation often skips the serving code, and then v2-bad would look exactly like v2-good.

The bundle also stores 2,000 training rows and their scores as the drift reference.

## 3. Model server

`server.py → create_app(bundle, version)` builds the FastAPI app around a loaded bundle.

- `POST /predict` validates the input with pydantic (`Student`), calls `predict_one`, and turns
  *any* model exception into a counted HTTP 500. A crash inside the model must become a metric,
  not a dead process.
- `GET /healthz` always answers 200 (liveness). `GET /readyz` answers 503 when the model failed
  to load (readiness). Kubernetes then keeps the pod out of the Service but does not restart it
  in a loop, which would not fix a missing file.
- An HTTP middleware records latency and `status` for `/predict` only, so probe and scrape
  traffic do not pollute the numbers.
- `GET /metrics` returns the app's own `CollectorRegistry` (one per app keeps tests isolated).

`app_from_env` reads `MODEL_VERSION`, `MODEL_DIR` and `RISK_THRESHOLD` (from the Deployment env and
the ConfigMap). `canary serve` starts it with uvicorn.

## 4. Metrics: from text to a window of traffic

`metrics.py` defines three metrics, all labelled with `model_version`:

| Metric | Type | Used for |
| --- | --- | --- |
| `canary_requests_total{status}` | counter | error rate (5xx / all) |
| `canary_request_latency_seconds` | histogram | p95 latency |
| `canary_prediction_score` | histogram, 10 buckets of 0.1 | prediction distribution (PSI) |

`parse` turns Prometheus text into a `Snapshot` (status counts plus cumulative buckets).
Counters only grow, so "what happened in this window" is `after - before`. `window_delta` does
that per pod and sums:

- a new pod (just scaled up) counts in full;
- a pod that disappeared (scaled down) adds nothing;
- a pod whose total went *down* restarted, so its counters began again at 0 and the new values
  count in full (the same rule PromQL's `rate()` uses).

## 5. Analysis: wait, pass or fail

`analysis.py → analyse(stable, canary, limits)` is a pure function.

1. **Wait** if either side has fewer than `min_requests` (200). PSI from a small sample is
   biased upwards, because empty buckets get a tiny floor and a large log term. A quick check
   (v1 scored against itself, 300 random draws, 2,000 baseline requests) gave an average PSI
   of 0.36 at 45 canary requests and 0.05 at 200 (95th percentile 0.10), well under the 0.2
   limit. The
   good release in `docs/results.md` shows exactly this: PSI 0.371 after 45 requests, 0.070
   after 244.
2. **Error rate:** fail if canary 5xx rate > stable 5xx rate + 1 percentage point. 4xx are the
   client's fault and are ignored.
3. **Latency:** `histogram_quantile` estimates p95 from buckets the way PromQL does (linear inside
   a bucket). Fail only if canary p95 > 1.5 × stable **and** more than 25 ms slower. The double
   rule stops "1 ms vs 2 ms" from failing a release.
4. **Prediction drift:** PSI between the canary and stable score histograms. Fail above 0.2.

All comparisons are strict (`>`), so a value exactly on the limit passes. The tests pin each
boundary.

## 6. The controller loop

`rollout.py → run_rollout(backend, steps=(10, 50, 100))`:

```
for each step:
    backend.set_canary_weight(step)          # scale Deployments
    baseline = backend.scrape()
    repeat up to max_windows:
        backend.wait_window()                # let traffic flow
        add window_delta to stable_total and canary_step_total
        verdict = analyse(stable_total, canary_step_total)
        wait -> next window | pass -> next step | fail -> rollback, stop
    only waits -> rollback ("not enough traffic")
backend.promote()
```

Two details matter:

- **Different baselines.** The canary is judged on this step's traffic only (after a scale-up it
  is a new set of pods). Stable is judged on everything since the rollout started. At the 100%
  step stable has no pods left, and this running total is still there to compare against.
- **Every exit is promote or rollback.** The function never returns with the canary at 50%.

`replica_split(total, percent)` maps a percentage to pods: 10 replicas → 9/1, 5/5, 0/10. It keeps
at least one canary pod above 0% and one stable pod below 100%.

## 7. Backends: kubectl and in-process

`kube.py → KubectlBackend` implements the backend with plain `kubectl` calls:

- `set_canary_weight`: scale the canary up first, then stable down, waiting for
  `kubectl rollout status` each time, so capacity never drops below 10 pods;
- `scrape`: list running pods per track, then read each one through the API server proxy
  `/api/v1/namespaces/canary-demo/pods/<pod>:8000/proxy/metrics`. No Prometheus, no port-forward;
- `promote`: read `MODEL_VERSION` from the canary Deployment, `kubectl set env` it on stable,
  scale stable to 10, canary to 0;
- `rollback`: stable to 10, then canary to 0.

Every call goes through one `runner` function, so tests use a fake that records commands.

`simulate.py → LocalBackend` runs two FastAPI apps in-process (TestClient) and routes each request
to the canary with probability `canary_pods / total`, like kube-proxy does. It scrapes their real
`/metrics`. This is what `canary rollout --local` and `canary report` use.

## 8. Kubernetes manifests

```
k8s/base               Deployment (probes, resources, securityContext) + ConfigMap
k8s/overlays/stable    nameSuffix -stable, label track=stable, 10 replicas, MODEL_VERSION=v1
k8s/overlays/canary    nameSuffix -canary, label track=canary, 0 replicas, MODEL_VERSION=v2-good
k8s/shared             Namespace (Pod Security "restricted") + the one Service
k8s/e2e                traffic generator (test only)
```

The Service selects only `app: dropout-model`, so it routes to pods of both Deployments. Each
Deployment selects `app` *and* `track`, so they never fight over pods. The Service lives in
`k8s/shared` because `nameSuffix` in an overlay would rename it.

The traffic generator opens a new connection per request. kube-proxy picks a pod per
connection, so a keep-alive client would stick to one pod and break the split.

## 9. Drift monitoring

`drift.py` has the PSI maths. `psi_vs_reference` puts bin edges at the reference deciles
(observed values, so a 0/1 column collapses to two bins) and compares bin shares. `canary drift`
scores a batch (synthetic, or a UCI CSV with `--input`) and prints PSI per feature and for the
prediction score; `--fail-on-alert` exits 1 if anything is above 0.25, for use in a cron job.

## 10. The end-to-end test

`scripts/e2e.sh` (CI job `e2e`):

1. create or reuse a kind cluster, build the image, `kind load docker-image`;
2. apply `shared`, `stable`, `canary`, wait for `rollout status`, start the traffic generator;
3. set the canary to v2-good, run `canary rollout` → expect exit 0; assert stable now runs v2-good
   with 10 ready pods and the canary has 0;
4. set the canary to v2-bad, run `canary rollout` → expect exit 3; assert stable still runs
   v2-good with 10 ready pods;
5. on any failure, print Deployments, pods, events and the last log lines of every pod.
