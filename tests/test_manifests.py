"""Kubernetes manifests: render the overlays and check what production needs.

With a `kustomize` binary on PATH the real `kustomize build` output is checked. Without it (this
sandbox), `render` applies the few Kustomize features the repo uses (namespace, nameSuffix,
labels, JSON patches) so the same checks still run. CI also renders with `kubectl kustomize`.
"""

from __future__ import annotations

import copy
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from canary_serving.model import VERSIONS

K8S = Path(__file__).resolve().parents[1] / "k8s"
Doc = dict[str, Any]


def _load_dir(path: Path) -> tuple[Doc, list[Doc]]:
    kust = yaml.safe_load((path / "kustomization.yaml").read_text())
    docs: list[Doc] = []
    for res in kust.get("resources", []):
        target = (path / res).resolve()
        if target.is_dir():
            docs += _render_python(target)
        else:
            docs += [d for d in yaml.safe_load_all(target.read_text()) if d]
    return kust, docs


def _json_patch(doc: Doc, ops: list[Doc]) -> None:
    for op in ops:
        assert op["op"] == "replace", "only replace ops are used in this repo"
        *parents, last = [int(p) if p.isdigit() else p for p in op["path"].strip("/").split("/")]
        node: Any = doc
        for key in parents:
            node = node[key]
        node[last] = op["value"]


def _render_python(path: Path) -> list[Doc]:
    kust, docs = _load_dir(path)
    docs = copy.deepcopy(docs)
    for patch in kust.get("patches", []):
        target = patch["target"]
        for doc in docs:
            if doc["kind"] == target["kind"] and doc["metadata"]["name"] == target["name"]:
                _json_patch(doc, yaml.safe_load(patch["patch"]))
    renames: dict[str, str] = {}
    for doc in docs:
        if ns := kust.get("namespace"):
            doc["metadata"]["namespace"] = ns
        if suffix := kust.get("nameSuffix"):
            old = doc["metadata"]["name"]
            renames[old] = doc["metadata"]["name"] = old + suffix
        for group in kust.get("labels", []):
            doc["metadata"].setdefault("labels", {}).update(group["pairs"])
            if doc["kind"] == "Deployment" and group.get("includeSelectors"):
                doc["spec"]["selector"]["matchLabels"].update(group["pairs"])
                doc["spec"]["template"]["metadata"]["labels"].update(group["pairs"])
    for doc in docs:  # name references, the one kustomize does for envFrom
        if doc["kind"] == "Deployment":
            for container in doc["spec"]["template"]["spec"]["containers"]:
                for ref in container.get("envFrom", []):
                    name = ref["configMapRef"]["name"]
                    ref["configMapRef"]["name"] = renames.get(name, name)
    return docs


def render(rel: str) -> list[Doc]:
    path = K8S / rel
    binary = shutil.which("kustomize")
    if binary:
        out = subprocess.run([binary, "build", str(path)], check=True, capture_output=True)
        return [d for d in yaml.safe_load_all(out.stdout) if d]
    return _render_python(path)


def _deployment(docs: list[Doc]) -> Doc:
    (dep,) = [d for d in docs if d["kind"] == "Deployment"]
    return dep


def _env(dep: Doc) -> dict[str, str]:
    container = dep["spec"]["template"]["spec"]["containers"][0]
    return {e["name"]: e["value"] for e in container["env"]}


@pytest.mark.parametrize(
    ("overlay", "replicas", "version"), [("stable", 10, "v1"), ("canary", 0, "v2-good")]
)
def test_overlay_names_labels_replicas_and_version(
    overlay: str, replicas: int, version: str
) -> None:
    docs = render(f"overlays/{overlay}")
    dep = _deployment(docs)
    # the controller addresses deployments as <app>-<track>
    assert dep["metadata"]["name"] == f"dropout-model-{overlay}"
    assert dep["metadata"]["namespace"] == "canary-demo"
    assert dep["spec"]["replicas"] == replicas
    assert _env(dep)["MODEL_VERSION"] == version and version in VERSIONS
    labels = {"app": "dropout-model", "track": overlay}
    assert dep["spec"]["selector"]["matchLabels"] == labels
    assert dep["spec"]["template"]["metadata"]["labels"] == labels
    configmaps = {d["metadata"]["name"] for d in docs if d["kind"] == "ConfigMap"}
    env_from = dep["spec"]["template"]["spec"]["containers"][0]["envFrom"][0]
    assert env_from["configMapRef"]["name"] in configmaps


@pytest.mark.parametrize("overlay", ["stable", "canary"])
def test_deployment_has_probes_resources_and_hardening(overlay: str) -> None:
    pod = _deployment(render(f"overlays/{overlay}"))["spec"]["template"]["spec"]
    (container,) = pod["containers"]
    assert container["readinessProbe"]["httpGet"]["path"] == "/readyz"
    assert container["livenessProbe"]["httpGet"]["path"] == "/healthz"
    for kind in ("requests", "limits"):
        assert {"cpu", "memory"} <= set(container["resources"][kind])
    assert pod["securityContext"]["runAsNonRoot"] is True
    sc = container["securityContext"]
    assert sc["allowPrivilegeEscalation"] is False and sc["readOnlyRootFilesystem"] is True
    assert sc["capabilities"]["drop"] == ["ALL"]


def test_service_selects_both_tracks() -> None:
    (svc,) = [d for d in render("shared") if d["kind"] == "Service"]
    selector = svc["spec"]["selector"]
    assert "track" not in selector  # replica-ratio split needs both tracks behind it
    for overlay in ("stable", "canary"):
        pod_labels = _deployment(render(f"overlays/{overlay}"))["spec"]["template"]["metadata"]
        assert selector.items() <= pod_labels["labels"].items()


def test_traffic_generator_targets_the_service() -> None:
    (dep,) = yaml.safe_load_all((K8S / "e2e" / "traffic-generator.yaml").read_text())
    command = dep["spec"]["template"]["spec"]["containers"][0]["command"]
    assert "http://dropout-model.canary-demo.svc" in command
