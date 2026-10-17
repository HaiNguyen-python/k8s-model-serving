.PHONY: install test lint typecheck format train serve report simulate e2e manifests

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src

format:
	ruff format .
	ruff check --fix .

train:
	canary train --out models

serve: train
	canary serve --version v1

# Local in-process rollouts (no cluster): one promote, one rollback
simulate:
	canary rollout --local v2-good
	canary rollout --local v2-bad || test $$? -eq 3

report:
	canary report --out docs

# Render the Kustomize overlays (needs kubectl or kustomize)
manifests:
	kubectl kustomize k8s/overlays/stable
	kubectl kustomize k8s/overlays/canary

# Full kind test: needs docker, kind and kubectl
e2e:
	bash scripts/e2e.sh
