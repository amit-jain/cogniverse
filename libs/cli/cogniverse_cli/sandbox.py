"""OpenShell sandbox gateway setup for the coding agent.

Handles three concerns:

1. Ensuring the ``openshell`` host CLI is installed and the gateway is running.
2. Syncing the gateway's mTLS certs and metadata into k8s secrets/configmaps
   so the runtime pod can reach the host gateway via host.docker.internal.
3. Re-syncing certs on demand (after rotation) without rebuilding anything.

The runtime pod picks up the certs via volume mounts defined in the Helm
chart (``runtime.sandbox.enabled`` = true).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

NAMESPACE = "cogniverse"
MTLS_SECRET = "openshell-mtls"
METADATA_CONFIGMAP = "openshell-metadata"
ACTIVE_CONFIGMAP = "openshell-active"
GATEWAY_NAME = "cogniverse"
POD_GATEWAY_ENDPOINT = "https://host.docker.internal:19091"


def openshell_installed() -> bool:
    """Return True if the openshell CLI is available on the host."""
    return shutil.which("openshell") is not None


def get_active_gateway_dir() -> Optional[Path]:
    """Return the config directory of the active openshell gateway, or None."""
    config_root = Path.home() / ".config" / "openshell"
    active_file = config_root / "active_gateway"
    if not active_file.exists():
        return None
    name = active_file.read_text().strip()
    if not name:
        return None
    gateway_dir = config_root / "gateways" / name
    if not gateway_dir.exists():
        return None
    return gateway_dir


def gateway_running() -> bool:
    """Return True if the openshell gateway is healthy."""
    if not openshell_installed():
        return False
    try:
        result = subprocess.run(
            ["openshell", "gateway", "info"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0 and "Gateway endpoint" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def start_gateway() -> bool:
    """Start the openshell gateway. Returns True on success."""
    if not openshell_installed():
        console.print(
            "[yellow]openshell CLI not found. Install with: "
            "`curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh`"
            "[/yellow]"
        )
        return False

    console.print("Starting OpenShell gateway...")
    try:
        result = subprocess.run(
            ["openshell", "gateway", "start"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        console.print("[red]Gateway start timed out after 120s[/red]")
        return False

    if result.returncode != 0:
        console.print(f"[red]Gateway start failed: {result.stderr[:300]}[/red]")
        return False

    return gateway_running()


def _kubectl(args: list[str], *, input_data: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["kubectl", *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        input=input_data,
    )


def sync_gateway_certs_to_cluster() -> bool:
    """Sync openshell gateway mTLS certs + metadata into k8s resources.

    Idempotent — re-creates the secret and configmaps from the current
    gateway state each time. Call this:
    - During ``cogniverse up`` after the cluster is ready.
    - After the gateway is restarted (cert rotation).
    - Before running coding agent e2e tests.

    Returns True on success.
    """
    gateway_dir = get_active_gateway_dir()
    if gateway_dir is None:
        console.print("[red]No active openshell gateway found[/red]")
        return False

    metadata_file = gateway_dir / "metadata.json"
    if not metadata_file.exists():
        console.print(f"[red]Gateway metadata missing: {metadata_file}[/red]")
        return False

    mtls_dir = gateway_dir / "mtls"
    ca_crt = mtls_dir / "ca.crt"
    tls_crt = mtls_dir / "tls.crt"
    tls_key = mtls_dir / "tls.key"
    for cert in (ca_crt, tls_crt, tls_key):
        if not cert.exists():
            console.print(f"[red]Gateway cert missing: {cert}[/red]")
            return False

    ns_check = _kubectl(["get", "namespace", NAMESPACE])
    if ns_check.returncode != 0:
        _kubectl(["create", "namespace", NAMESPACE])

    secret_yaml = _kubectl([
        "create", "secret", "generic", MTLS_SECRET,
        "-n", NAMESPACE,
        f"--from-file=ca.crt={ca_crt}",
        f"--from-file=tls.crt={tls_crt}",
        f"--from-file=tls.key={tls_key}",
        "--dry-run=client", "-o", "yaml",
    ])
    if secret_yaml.returncode != 0:
        console.print(f"[red]Failed to build mTLS secret: {secret_yaml.stderr}[/red]")
        return False
    apply_secret = _kubectl(["apply", "-f", "-"], input_data=secret_yaml.stdout)
    if apply_secret.returncode != 0:
        console.print(f"[red]Failed to apply mTLS secret: {apply_secret.stderr}[/red]")
        return False

    metadata = json.loads(metadata_file.read_text())
    metadata["gateway_endpoint"] = POD_GATEWAY_ENDPOINT
    metadata_for_pod = json.dumps(metadata)

    proc = subprocess.run(
        ["kubectl", "create", "configmap", METADATA_CONFIGMAP,
         "-n", NAMESPACE,
         f"--from-literal=metadata.json={metadata_for_pod}",
         "--dry-run=client", "-o", "yaml"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    if proc.returncode != 0:
        console.print(f"[red]Failed to build metadata configmap: {proc.stderr}[/red]")
        return False
    apply_meta = _kubectl(["apply", "-f", "-"], input_data=proc.stdout)
    if apply_meta.returncode != 0:
        console.print(f"[red]Failed to apply metadata configmap: {apply_meta.stderr}[/red]")
        return False

    active_yaml = _kubectl([
        "create", "configmap", ACTIVE_CONFIGMAP,
        "-n", NAMESPACE,
        f"--from-literal=active_gateway={GATEWAY_NAME}",
        "--dry-run=client", "-o", "yaml",
    ])
    apply_active = _kubectl(["apply", "-f", "-"], input_data=active_yaml.stdout)
    if apply_active.returncode != 0:
        console.print(f"[red]Failed to apply active configmap: {apply_active.stderr}[/red]")
        return False

    console.print(f"[green]OpenShell gateway certs synced to {NAMESPACE} namespace[/green]")
    return True


def ensure_sandbox_ready() -> bool:
    """End-to-end setup: CLI installed → gateway running → certs synced.

    Returns True if the sandbox is ready to be used by the runtime pod.
    """
    if not openshell_installed():
        console.print(
            "[yellow]Coding agent sandbox disabled — openshell CLI not installed.[/yellow]"
        )
        return False

    if not gateway_running():
        if not start_gateway():
            return False

    return sync_gateway_certs_to_cluster()
