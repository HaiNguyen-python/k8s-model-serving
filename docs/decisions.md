# Design decisions

Short records of the main choices: what was decided, why, and what it costs. Format inspired by
Architecture Decision Records (ADRs).

## 1. Replica-ratio canary instead of a service mesh or ingress weights

- **Decision:** two Deployments (stable, canary) behind one Service that selects only
  `app: dropout-model`. Traffic share = canary ready pods / all ready pods.
- **Why:** it needs nothing beyond core Kubernetes, works on any cluster including kind, and is
  easy to explain and test. The controller only has to scale Deployments.
- **Cost (the limits):**
  - *Coarse steps.* With 10 replicas the smallest step is 10%; with 4 replicas "10%" is 25%.
    Exact 1% or 5% needs more pods or weighted routing.
  - *Approximate split.* kube-proxy picks a pod per connection at random. Over a few hundred
    requests the share is close to the pod ratio, but not exact, and clients with long-lived
    keep-alive connections can skew it (the traffic generator opens a new connection per
    request for this reason).
  - *No stickiness.* The same student can get v1 and v2 answers on consecutive requests.
  - *No header routing.* You cannot send only internal testers to the canary.
  - *Capacity is tied to the split.* Scaling for load and scaling for the canary are the same knob.
- **Alternatives:** ingress-nginx canary annotations (`canary-weight`), Gateway API
  `HTTPRoute` weights, Argo Rollouts or Flagger (which automate this loop), Istio/Linkerd.
  All give exact weights; all add a component to install and operate.

## 2. Scrape pods through the API server proxy, not a Prometheus server

- **Decision:** the controller lists pods per track and reads
  `/api/v1/namespaces/<ns>/pods/<pod>:8000/proxy/metrics` with `kubectl get --raw`.
- **Why:** the e2e test needs no Prometheus install, access is controlled by RBAC (`pods/proxy`),
  and there is no port-forward to keep alive. Parsing the standard text format means switching
  to Prometheus later only changes the backend, not the analysis.
- **Cost:** counts from a pod that is scaled down between two scrapes are lost (a few seconds of
  traffic). It also adds load on the API server, fine for 10 pods, not for 1,000.

## 3. Pure decision function with three answers

- **Decision:** `analyse(stable, canary, limits) -> wait | pass | fail`, no I/O inside.
- **Why:** the risky part of a canary is the decision. As a pure function it can be tested at
  every boundary in milliseconds, and the same code runs in the simulation and the cluster.
- **Cost:** the controller loop around it still needs integration testing (fake backend tests
  plus the kind e2e job).

## 4. Thresholds

- **Decision:** at least 200 requests per side; fail on canary error rate > stable + 1 pp,
  p95 latency > 1.5x stable *and* > 25 ms slower, prediction PSI > 0.2. Values on the limit pass.
- **Why:**
  - 200 requests: a quick check measured an average PSI of 0.05 (95th percentile 0.10) for the
    same model against itself at 200 requests, versus 0.36 at 45 (see how-it-works, step 5).
  - Relative error margin: the canary is judged against what stable does right now, not against
    an absolute number, so a downstream outage that hits both does not blame the canary.
  - Two-part latency rule: tiny absolute differences are noise even when the ratio is large.
  - PSI 0.2: a common "significant shift" level. A better model may move predictions a bit
    (v2-good: about 0.03-0.07), so the limit is meant to catch large moves like v2-bad (1.1).
- **Cost:** 200 requests at 10% means 2,000 total requests per step; low-traffic services wait
  longer. A PSI limit can block a model that is better *because* it predicts differently; then
  a human has to review and raise the limit for that release on purpose.

## 5. Stable baseline over the whole rollout, canary per step

- **Decision:** stable totals keep growing from the first window; the canary total resets at
  each step.
- **Why:** at the 100% step stable has no pods, so a per-step stable baseline would be empty.
  The canary resets because after a scale-up it is a different set of pods under more load.
- **Cost:** if stable itself changes behaviour during a long rollout (time of day, traffic mix),
  the old baseline is a little stale.

## 6. Not enough traffic means rollback

- **Decision:** if a step only produces `wait` for `max_windows` windows, roll back.
- **Why:** a controller must never hang with a half-deployed canary. Rolling back is the safe
  default; the release can be retried when there is traffic.
- **Cost:** a quiet night can roll back a perfectly good release.

## 7. Promote by copying the version into the stable Deployment

- **Decision:** on promote, `kubectl set env deployment/dropout-model-stable MODEL_VERSION=<canary>`,
  scale stable to 10, canary to 0.
- **Why:** the end state is always the same shape ("stable has everything, canary has 0"), so
  the next rollout starts from a known place and the e2e test can assert it.
- **Cost:** the live cluster now differs from `k8s/overlays/stable` in Git. Re-applying the
  overlay would bring v1 back. In a real setup the controller (or a person) commits the new
  version to Git; see the runbook.

## 8. Models trained into the image at build time

- **Decision:** `canary train` runs in the Docker build stage with a fixed seed; all three
  versions are in one image and `MODEL_VERSION` picks one.
- **Why:** one image to build and `kind load`, no model registry or volume needed for the demo.
  The seed makes builds reproducible.
- **Cost:** a model change needs an image rebuild, and every pod carries all versions (small
  here: three logistic regressions). With real models, pull from a registry (e.g. MLflow) at
  start-up and gate readiness on that download.

## 9. Liveness and readiness are different questions

- **Decision:** `/healthz` always 200 while the process runs; `/readyz` 503 until the model loads.
- **Why:** if the model file is missing, restarting will not help, but sending traffic would
  return errors. Not-ready keeps the pod out of the Service and keeps its logs readable.
- **Cost:** a pod that is alive but never ready needs an alert (see runbook), otherwise it
  silently reduces capacity.

## 10. Container hardening

- **Decision:** non-root user 10001, read-only root filesystem, all capabilities dropped,
  no privilege escalation, seccomp `RuntimeDefault`, no service account token, namespace with
  Pod Security "restricted". Two-stage build on `python:3.12-slim-bookworm`.
- **Why:** a model server needs none of those permissions, and the restricted profile turns
  mistakes into admission errors instead of silent risk. The Debian codename is pinned because
  plain `-slim` tags move to new Debian releases.
- **Cost:** anything that writes files must use the `/tmp` emptyDir.

## 11. Synthetic data with the UCI schema

- **Decision:** a seeded generator mirrors seven UCI columns; `load_uci` reads the real file.
- **Why:** CI and anyone cloning the repo can run everything offline, and the planted effects
  (non-linear pass ratio, zero-enrolment students) make the v1 → v2 gain and the v2-bad crash
  explainable.
- **Cost:** results are about the release process, not about predicting real dropout.
