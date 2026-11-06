"""Static validation of authored platform artefacts.

Docker and Kubernetes could not run in this environment. These tests validate
the artefact files themselves (YAML parses, required fields present, secrets
referenced and never embedded). Passing here does not mean deployed.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load_all(path):
    return [doc for doc in yaml.safe_load_all(path.read_text()) if doc]


def test_compose_yaml_valid():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    assert set(compose["services"]) == {"app", "opensearch"}
    assert compose["services"]["app"]["ports"] == ["8501:8501"]


def test_k8s_manifests_have_required_fields():
    deployment = _load_all(ROOT / "k8s" / "deployment.yaml")[0]
    assert deployment["kind"] == "Deployment"
    assert deployment["spec"]["replicas"] == 2
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["ports"][0]["containerPort"] == 8501
    env_from = container["envFrom"]
    kinds = {list(entry)[0] for entry in env_from}
    assert kinds == {"configMapRef", "secretRef"}

    service = _load_all(ROOT / "k8s" / "service.yaml")[0]
    assert service["kind"] == "Service"
    assert service["spec"]["ports"][0]["targetPort"] == 8501

    docs = _load_all(ROOT / "k8s" / "configmap.yaml")
    kinds = {d["kind"] for d in docs}
    assert kinds == {"ConfigMap", "Secret"}
    secret = [d for d in docs if d["kind"] == "Secret"][0]
    assert secret.get("stringData", {}) == {}  # referenced only, no values committed


def test_dockerfile_present_and_sane():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert dockerfile.startswith("# Authored artefact")
    assert "FROM python:3.12" in dockerfile
    assert "8501" in dockerfile


def test_adrs_exist():
    adr_dir = ROOT / "docs" / "adr"
    names = sorted(p.name for p in adr_dir.glob("ADR-*.md"))
    assert len(names) == 3
    for name in names:
        assert "Status: Accepted" in (adr_dir / name).read_text()
