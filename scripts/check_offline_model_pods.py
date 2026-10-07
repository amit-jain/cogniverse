"""Check that deployed inference pods start without network access.

Reads the live Deployments and their pods with ``kubectl get`` (read-only) and
reports, per Deployment:

- a container or init container that installs packages, reaches a host outside
  the cluster, or downloads from the Hugging Face Hub outside a cache-first
  model-warm init container;
- a model server that starts through a shell script or runs without
  ``HF_HUB_OFFLINE=1``;
- a pod that is not Ready or whose containers have restarted.

The same detector backs ``tests/charts/test_model_pod_offline_startup.py``, so
the rendered chart and the running cluster are held to one rule.

Usage::

    uv run python scripts/check_offline_model_pods.py [deployment ...]
        [--context k3d-cogniverse-e2e] [--namespace cogniverse]

With no deployment named, every inference Deployment in the namespace is
checked. Exits 1 when any check fails.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from urllib.parse import urlparse

_PACKAGE_INSTALL = re.compile(
    r"\b(?:pip3?|uv\s+pip|-m\s+pip)\s+install\b"
    r"|\bapt(?:-get)?\s+(?:-\S+\s+)*install\b"
    r"|\bapk\s+add\b"
    r"|\b(?:conda|mamba|micromamba)\s+install\b"
    r"|\bnpm\s+(?:install|ci)\b"
)
_URL = re.compile(r"https?://[^\s\"'`]+")
_HUB_DOWNLOAD = re.compile(
    r"\b(?:snapshot_download|hf_hub_download)\s*\("
    r"|\bhuggingface-cli\s+download\b"
    r"|\bhf\s+download\b"
)
_SHELLS = {"sh", "bash", "/bin/sh", "/bin/bash", "ash", "dash"}
_INFERENCE_COMPONENT = "app.kubernetes.io/component"


def _start_text(container: dict) -> str:
    return "\n".join(
        str(part)
        for part in [*container.get("command", []), *container.get("args", [])]
    )


def _is_cluster_host(url: str) -> bool:
    host = urlparse(url.replace("$(", "").replace("${", "")).hostname or ""
    return (
        host in {"localhost", "127.0.0.1"}
        or "." not in host
        or host.endswith((".svc", ".cluster.local"))
    )


def offline_startup_violations(pod: dict) -> list[str]:
    """Return why a pod spec could not start without network access."""
    found: list[str] = []
    init = pod.get("initContainers", [])
    for kind, containers in (("init", init), ("container", pod["containers"])):
        for container in containers:
            name = f"{kind} {container['name']}"
            text = _start_text(container)
            if _PACKAGE_INSTALL.search(text):
                found.append(f"{name} installs packages at start")
            env_values = [str(e.get("value", "")) for e in container.get("env", [])]
            for url in _URL.findall("\n".join([text, *env_values])):
                if not _is_cluster_host(url):
                    found.append(f"{name} reaches outside host {url}")
            if _HUB_DOWNLOAD.search(text):
                if kind != "init":
                    found.append(f"{name} downloads from the Hub")
                elif "local_files_only=True" not in text:
                    found.append(f"{name} downloads from the Hub without a cache check")
    for container in pod["containers"]:
        name = f"container {container['name']}"
        command = container.get("command", [])
        if command and command[0] in _SHELLS:
            found.append(f"{name} starts through a shell script")
        env = {e["name"]: e.get("value") for e in container.get("env", [])}
        if env.get("HF_HUB_OFFLINE") != "1":
            found.append(f"{name} does not set HF_HUB_OFFLINE=1")
    return found


def pod_start_violations(pods: Sequence[dict]) -> list[str]:
    """Return why the Deployment's pods did not start cleanly."""
    if not pods:
        return ["no pods"]
    found: list[str] = []
    for pod in pods:
        name = pod["metadata"]["name"]
        status = pod.get("status", {})
        ready = {c["type"]: c["status"] for c in status.get("conditions", [])}.get(
            "Ready"
        )
        if ready != "True":
            found.append(f"pod {name} is not Ready")
        for kind, key in (
            ("init", "initContainerStatuses"),
            ("container", "containerStatuses"),
        ):
            for container in status.get(key, []):
                restarts = container.get("restartCount", 0)
                if restarts:
                    found.append(
                        f"pod {name} {kind} {container['name']} restarted {restarts} times"
                    )
    return found


def deployment_violations(deployment: dict, pods: Sequence[dict]) -> list[str]:
    return offline_startup_violations(
        deployment["spec"]["template"]["spec"]
    ) + pod_start_violations(pods)


Runner = Callable[..., subprocess.CompletedProcess]


def _kubectl_json(run: Runner, context: str, namespace: str, *args: str) -> dict:
    result = run(
        ["kubectl", "--context", context, "-n", namespace, "get", *args, "-o", "json"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"kubectl get {' '.join(args)} failed: {result.stderr}")
    return json.loads(result.stdout)


def check(
    deployments: Sequence[str],
    *,
    context: str,
    namespace: str,
    run: Runner = subprocess.run,
) -> dict[str, list[str]]:
    """Return the violations of each checked Deployment, keyed by name."""
    if deployments:
        items = [
            _kubectl_json(run, context, namespace, "deployment", name)
            for name in deployments
        ]
    else:
        items = [
            item
            for item in _kubectl_json(run, context, namespace, "deployments")["items"]
            if item["spec"]["template"]["metadata"]["labels"]
            .get(_INFERENCE_COMPONENT, "")
            .startswith("inference-")
        ]
        if not items:
            raise RuntimeError(f"no inference Deployments in namespace {namespace}")
    results: dict[str, list[str]] = {}
    for deployment in items:
        selector = ",".join(
            f"{key}={value}"
            for key, value in sorted(
                deployment["spec"]["selector"]["matchLabels"].items()
            )
        )
        pods = _kubectl_json(run, context, namespace, "pods", "-l", selector)["items"]
        results[deployment["metadata"]["name"]] = deployment_violations(
            deployment, pods
        )
    return results


def main(argv: Sequence[str] | None = None, run: Runner = subprocess.run) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("deployments", nargs="*")
    parser.add_argument("--context", default="k3d-cogniverse-e2e")
    parser.add_argument("--namespace", default="cogniverse")
    args = parser.parse_args(argv)
    results = check(
        args.deployments, context=args.context, namespace=args.namespace, run=run
    )
    for name, problems in sorted(results.items()):
        print(f"{name}: {'ok' if not problems else 'FAIL'}")
        for problem in problems:
            print(f"  - {problem}")
    return 1 if any(results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
