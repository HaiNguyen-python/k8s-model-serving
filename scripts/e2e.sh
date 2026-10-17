#!/usr/bin/env bash
# End-to-end test on a kind cluster: deploy v1, roll out v2-good (expect promote), then roll out
# v2-bad (expect rollback), and check the final cluster state with kubectl.
#
# Needs: docker, kind, kubectl, and this package installed (`pip install -e .`) for the controller.
# CI creates the cluster with helm/kind-action; locally the script creates it if it is missing.
#   CLUSTER_NAME=canary scripts/e2e.sh
set -Eeuo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-canary}"
NS=canary-demo
IMAGE=canary-serving:local
REPLICAS=10
WINDOW_SECONDS="${WINDOW_SECONDS:-15}"
MAX_WINDOWS="${MAX_WINDOWS:-12}"
TIMEOUT=240s
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log() { printf '\n=== %s ===\n' "$*"; }

dump_state() {
  log "FAILED - cluster state for debugging"
  kubectl -n "$NS" get deploy,rs,pods,svc,endpoints -o wide || true
  kubectl -n "$NS" get events --sort-by=.lastTimestamp | tail -n 40 || true
  for pod in $(kubectl -n "$NS" get pods -o name 2>/dev/null); do
    echo "--- logs $pod"
    kubectl -n "$NS" logs "$pod" --tail=40 || true
  done
}
trap 'dump_state' ERR

for tool in docker kind kubectl canary; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool"; exit 1; }
done

if ! kind get clusters | grep -qx "$CLUSTER_NAME"; then
  log "Creating kind cluster $CLUSTER_NAME"
  kind create cluster --name "$CLUSTER_NAME" --wait 120s
fi
kubectl config use-context "kind-$CLUSTER_NAME"

log "Build and load image"
docker build -t "$IMAGE" .
kind load docker-image "$IMAGE" --name "$CLUSTER_NAME"

log "Deploy stable v1, canary at 0 replicas, Service, traffic generator"
kubectl apply -k k8s/shared
kubectl apply -k k8s/overlays/stable
kubectl apply -k k8s/overlays/canary
kubectl -n "$NS" rollout status deployment/dropout-model-stable --timeout="$TIMEOUT"
kubectl apply -f k8s/e2e/traffic-generator.yaml
kubectl -n "$NS" rollout status deployment/traffic-generator --timeout="$TIMEOUT"

model_version() {
  kubectl -n "$NS" get "deployment/$1" \
    -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="MODEL_VERSION")].value}'
}

assert_state() {  # expected stable version, expected ready stable replicas
  local version replicas ready canary
  version="$(model_version dropout-model-stable)"
  replicas="$(kubectl -n "$NS" get deployment/dropout-model-stable -o jsonpath='{.spec.replicas}')"
  ready="$(kubectl -n "$NS" get deployment/dropout-model-stable -o jsonpath='{.status.readyReplicas}')"
  canary="$(kubectl -n "$NS" get deployment/dropout-model-canary -o jsonpath='{.spec.replicas}')"
  echo "stable version=$version replicas=$replicas ready=$ready; canary replicas=$canary"
  [[ "$version" == "$1" ]] || { echo "expected stable version $1, got $version"; return 1; }
  [[ "$replicas" == "$2" && "$ready" == "$2" ]] || { echo "expected $2 ready stable pods"; return 1; }
  [[ "$canary" == "0" ]] || { echo "expected canary scaled to 0, got $canary"; return 1; }
}

run_rollout() {  # candidate version, expected exit code (0 promote, 3 rollback)
  log "Rollout $1 (expect exit $2)"
  kubectl -n "$NS" set env deployment/dropout-model-canary "MODEL_VERSION=$1"
  local code=0
  canary rollout --namespace "$NS" --replicas "$REPLICAS" \
    --window-seconds "$WINDOW_SECONDS" --max-windows "$MAX_WINDOWS" || code=$?
  [[ "$code" == "$2" ]] || { echo "rollout of $1 exited $code, expected $2"; return 1; }
}

run_rollout v2-good 0
assert_state v2-good "$REPLICAS"

run_rollout v2-bad 3
assert_state v2-good "$REPLICAS"

log "E2E passed: v2-good promoted, v2-bad rolled back, stable serves v2-good"
