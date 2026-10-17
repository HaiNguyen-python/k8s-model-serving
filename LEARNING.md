# Learning guide (for me, Hai)

This file is my study guide for this repo. Goal: explain every part in an interview, change it
with confidence, and answer "why?" three levels deep. It connects the University of Helsinki
MOOCs *DevOps with Docker* and *DevOps with Kubernetes* to an ML release problem.

Start with [docs/how-it-works.md](docs/how-it-works.md) (code walkthrough) and
[docs/decisions.md](docs/decisions.md) (reasons). This file is the practice plan.

## 1. Run it and watch each step (1-2 hours)

```bash
make install
canary train --out models                 # three versions; read the printed metrics
canary serve --version v2-bad             # in another terminal:
canary traffic --url http://localhost:8000 --rate 0 --total 300
curl localhost:8000/metrics | grep canary_requests_total    # count the 500s
canary rollout --local v2-good; echo $?   # 0
canary rollout --local v2-bad; echo $?    # 3
canary drift --shift 1.0
canary report
```

Then, with Docker and kind on my own laptop: `make e2e`. Watch it in a second terminal:
`kubectl -n canary-demo get pods -L track -w`. See the canary pod appear, then 5 pods, then the
stable pods restart with v2-good. Run it again and watch v2-bad never get past one pod.

## 2. Read the code in this order

| File | What to understand | Question to answer myself |
| --- | --- | --- |
| `data.py` | the planted effects | Why does v2 beat v1 at all? |
| `model.py` | `ModelBundle.vector`, how v2-bad is made | Why would offline AUC on the *estimator* miss the bug? |
| `server.py` | middleware, 500 handling, `/readyz` vs `/healthz` | What happens to a pod whose model file is missing? |
| `metrics.py` | counters vs histograms, `window_delta` | What does a pod restart do to a counter? |
| `analysis.py` | `histogram_quantile`, the three checks | Why wait for 200 requests? Why `>` and not `>=`? |
| `drift.py` | PSI formula, decile bins | Why the `EPS` floor? What does PSI 0.2 mean in words? |
| `rollout.py` | the loop, the two baselines | What is the stable baseline at the 100% step? |
| `kube.py` | scale order, pod proxy, promote | Why scale the canary up *before* stable down? |
| `k8s/` | base, overlays, shared Service | Why is the Service not in an overlay? |
| `scripts/e2e.sh` | waits, asserts, debug dump | What does the script print when it fails? |

## 3. Key concepts in my own words (fill these in)

- **Canary release vs blue-green vs rolling update:** ...
- **Replica-ratio traffic split and its limits:** ...
- **Liveness vs readiness probe:** ...
- **Prometheus counter vs histogram; why `rate()` and deltas:** ...
- **p95 from histogram buckets:** ...
- **PSI:** ...
- **Blast radius:** ...
- **Why the decision logic is a pure function:** ...

## 4. Interview questions this repo answers

1. *How do you release a new model safely?* Canary 10/50/100, compare with stable on errors,
   latency and prediction distribution, automatic rollback. Show the timeline chart.
2. *Offline metrics were fine but production broke. How?* Training/serving skew: v2-bad has the
   same weights, but the serving code swaps two features and crashes on one input type.
3. *How do you split traffic without a service mesh, and what are the limits?* Two Deployments
   behind one Service; coarse steps, random per connection, no stickiness, no header routing.
4. *What if there is not enough traffic?* Wait for a minimum sample; after N windows, roll back.
5. *How do you avoid false rollbacks?* Minimum sample size, relative error margin, latency needs
   both a ratio and an absolute gap, boundaries tested. Mention PSI 0.37 on 45 requests.
6. *What is in your runbook?* Alerts, what each means, manual rollback, keeping Git in sync.
7. *How would you do this at scale?* Prometheus + Argo Rollouts/Flagger with analysis templates,
   ingress or Gateway API weights, GitOps.

## 5. Make it mine (do at least three)

- [ ] Run `make e2e` on my laptop, record a short screen capture of `kubectl get pods -w`.
- [ ] Add a fourth check: canary mean predicted risk within ±0.05 of stable. Write its tests first.
- [ ] Make a "slow" release (sleep 100 ms in `predict_one`) and show the latency check failing.
- [ ] Replace the replica split with ingress-nginx `canary-weight` annotations in a new backend.
- [ ] Download the UCI file, run `canary drift --input data.csv`, and describe what I see.
- [ ] Deploy Prometheus with the kube-prometheus-stack chart and turn the runbook alerts into rules.

## 6. Quality checks I should be able to explain

- `make lint`, `mypy src`, `pytest` (58 test functions, 96% coverage), `pre-commit install`.
- The manifest tests render the overlays with a small Python renderer when `kustomize` is not
  installed; CI also runs `kubectl kustomize`.

## 7. Honesty checklist before pushing

- [ ] I can explain every file without looking at it.
- [ ] Results say "synthetic data, local simulation" wherever they appear.
- [ ] I ran the e2e myself or I say clearly that only CI ran it.
- [ ] Anything I haven't built is under "What I'd do next".

Note: Docker builds and the kind e2e test were not run in the environment where this repo was
first built (no Docker daemon there). CI runs them; check the first CI run and fix what breaks.
