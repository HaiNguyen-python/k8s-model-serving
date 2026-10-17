# Runbook: dropout-model serving and canary rollouts

For whoever is on call for the `dropout-model` service in namespace `canary-demo`.
The alert expressions below are written in PromQL for a Prometheus setup; this repo itself does
not deploy Prometheus (the controller scrapes pods directly), so treat them as the rules to add.
They assume the scrape config copies the pod label `track` onto every series (a standard
`kubernetes_sd_configs` relabel rule); the server itself only adds `model_version`.

## Quick reference

```bash
NS=canary-demo
kubectl -n $NS get deploy,pods -l app=dropout-model -L track     # who runs what
kubectl -n $NS get deploy dropout-model-stable dropout-model-canary \
  -o custom-columns=NAME:.metadata.name,REPLICAS:.spec.replicas,READY:.status.readyReplicas,\
VERSION:'.spec.template.spec.containers[0].env[?(@.name=="MODEL_VERSION")].value'
kubectl -n $NS logs -l app=dropout-model,track=canary --tail=50   # canary errors
kubectl get --raw /api/v1/namespaces/$NS/pods/<pod>:8000/proxy/metrics | grep canary_
```

## Alerts

### CanaryErrorRateHigh (page during a rollout)

```promql
sum(rate(canary_requests_total{status=~"5..", track="canary"}[5m]))
  / sum(rate(canary_requests_total{track="canary"}[5m]))
> on() (sum(rate(canary_requests_total{status=~"5..", track="stable"}[5m]))
  / sum(rate(canary_requests_total{track="stable"}[5m])) + 0.01)
```

- **Means:** the canary returns more server errors than stable (more than 1 pp above). The
  controller rolls back on the same rule; this alert is the safety net if the controller is not
  running or is stuck.
- **Check:** is a rollout running (`ps`/CI job)? Canary logs: `prediction failed` with a traceback
  points at the model code (v2-bad shows `ZeroDivisionError`).
- **Act:** if the controller is not acting within one window, roll back manually (below).

### PredictionDistributionShift (ticket, page if during a rollout)

- **Means:** PSI of the canary's (or the current model's) prediction histogram vs stable / the
  training reference is above 0.2. The model gives different answers to the same kind of input.
- **Check:** `canary drift --version <version>` for input drift. If inputs are stable but
  predictions moved, suspect the serving code (feature order, preprocessing). If inputs moved,
  it is a population change; see InputDrift.
- **Act:** during a rollout, roll back. Outside a rollout, open a ticket for the model owner.

### LatencyP95High (ticket)

```promql
histogram_quantile(0.95, sum by (le, track) (rate(canary_request_latency_seconds_bucket[5m])))
  > 0.25
```

- **Means:** 5% of predictions take more than 250 ms.
- **Check:** CPU throttling (`kubectl top pods`, limits are 500m), pod restarts, a new model that
  is heavier.
- **Act:** scale stable (`kubectl scale ... --replicas=N`, and update the overlay in Git), or
  raise the CPU limit. Roll back if a new version caused it.

### PodsNotReady (ticket, page if stable is below 7 ready pods)

```promql
kube_deployment_status_replicas_ready{deployment="dropout-model-stable"}
  < kube_deployment_spec_replicas{deployment="dropout-model-stable"}
```

- **Means:** pods run but `/readyz` is 503, usually "model not loaded".
- **Check:** `kubectl logs <pod>` for `could not load model`; a wrong `MODEL_VERSION` value
  (only `v1`, `v2-good`, `v2-bad` exist in the image) or a wrong image tag.
- **Act:** fix the env value or image; `kubectl rollout undo deployment/dropout-model-stable`
  returns to the previous ReplicaSet.

### InputDrift (ticket, daily)

- **Means:** `canary drift --fail-on-alert` exited 1: at least one feature or the score has
  PSI > 0.25 against the training reference.
- **Act:** not an outage. Tell the model owner; retraining or re-checking thresholds may be
  needed. The model can be right and the population different (a new intake, a new course).

## Manual rollback

Use when the controller is down, stuck, or a problem appears after promotion.

**During a rollout (canary has pods):**

```bash
NS=canary-demo
kubectl -n $NS scale deployment/dropout-model-stable --replicas=10
kubectl -n $NS rollout status deployment/dropout-model-stable --timeout=180s
kubectl -n $NS scale deployment/dropout-model-canary --replicas=0
```

Stable first, then the canary: capacity never drops.

**After a bad promotion (stable already runs the new version):**

```bash
kubectl -n $NS set env deployment/dropout-model-stable MODEL_VERSION=v1   # last good version
kubectl -n $NS rollout status deployment/dropout-model-stable --timeout=180s
# or, to return to the previous ReplicaSet exactly:
kubectl -n $NS rollout undo deployment/dropout-model-stable
```

**Then:**

1. Check: every stable pod ready, canary at 0, `canary_requests_total{status="500"}` flat.
2. Make Git match the cluster: `MODEL_VERSION` in `k8s/overlays/stable/kustomization.yaml`.
   Otherwise the next `kubectl apply -k` undoes your fix (or your promotion).
3. Write down what happened (version, time, which check fired, how many requests hit the
   canary) for the post-incident review.

## Running a rollout

```bash
kubectl -n canary-demo set env deployment/dropout-model-canary MODEL_VERSION=v2-good
canary rollout --namespace canary-demo --replicas 10 --window-seconds 15 --max-windows 12
echo $?   # 0 promoted, 3 rolled back, anything else: the controller failed, check the state
```

After a promotion, update `MODEL_VERSION` in the stable overlay in Git.
