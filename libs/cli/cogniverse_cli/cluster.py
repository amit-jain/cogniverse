"""k3d cluster lifecycle and prerequisite checks."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys

PREREQUISITES = ["docker", "kubectl", "helm"]
CLUSTER_NAME = "cogniverse"
NAMESPACE = "cogniverse"
DEFAULT_PORTS = [8080, 19071, 8000, 8501, 6006, 4317, 11434, 2746]

# Install commands per tool per platform
_INSTALL_COMMANDS: dict[str, dict[str, list[str]]] = {
    "k3d": {
        "darwin": ["brew", "install", "k3d"],
        "linux": ["bash", "-c", "curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash"],
    },
    "helm": {
        "darwin": ["brew", "install", "helm"],
        "linux": ["bash", "-c", "curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash"],
    },
    "kubectl": {
        "darwin": ["brew", "install", "kubectl"],
        "linux": ["bash", "-c", "curl -LO 'https://dl.k8s.io/release/$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl' && install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl"],
    },
}


def check_prerequisites(*, require_k3d: bool = True) -> list[str]:
    """Return list of missing prerequisites."""
    tools = list(PREREQUISITES)
    if require_k3d:
        tools.append("k3d")
    return [tool for tool in tools if shutil.which(tool) is None]


def install_prerequisite(tool: str) -> bool:
    """Attempt to install a missing prerequisite. Returns True on success."""
    os_name = platform.system().lower()
    commands = _INSTALL_COMMANDS.get(tool, {})
    cmd = commands.get(os_name)
    if not cmd:
        return False
    try:
        result = subprocess.run(cmd, timeout=120, check=False)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def install_missing_prerequisites(missing: list[str]) -> list[str]:
    """Try to install missing tools. Returns list of tools that still can't be found."""
    still_missing = []
    for tool in missing:
        if tool == "docker":
            # Docker must be installed manually
            still_missing.append(tool)
            continue
        if install_prerequisite(tool):
            # Verify it's now in PATH
            if shutil.which(tool):
                continue
        still_missing.append(tool)
    return still_missing


def has_existing_k8s() -> bool:
    """Check if kubectl can reach an existing K8s cluster."""
    try:
        result = subprocess.run(
            ["kubectl", "cluster-info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def cluster_exists(name: str = CLUSTER_NAME) -> bool:
    """Check if a k3d cluster with the given name exists."""
    result = subprocess.run(
        ["k3d", "cluster", "list", name],
        capture_output=True,
        timeout=10,
    )
    return result.returncode == 0


def create_cluster(
    name: str = CLUSTER_NAME,
    ports: list[int] | None = None,
) -> None:
    """Create a k3d cluster with port mappings.

    Each port gets a ``-p`` flag mapping it through the load balancer.
    """
    if ports is None:
        ports = DEFAULT_PORTS
    cmd = ["k3d", "cluster", "create", name]
    for port in ports:
        cmd.extend(["-p", f"{port}:{port}@loadbalancer"])
    subprocess.run(cmd, check=True, timeout=120)


def delete_cluster(name: str = CLUSTER_NAME) -> None:
    """Delete a k3d cluster."""
    subprocess.run(
        ["k3d", "cluster", "delete", name],
        check=True,
        timeout=60,
    )


# Port-forward specs: (service_name, namespace, local_port, service_port)
PORT_FORWARD_SPECS: list[tuple[str, str, int, int]] = [
    ("cogniverse-vespa", NAMESPACE, 8080, 8080),
    ("cogniverse-vespa", NAMESPACE, 19071, 19071),
    ("cogniverse-runtime", NAMESPACE, 8000, 8000),
    ("cogniverse-dashboard", NAMESPACE, 8501, 8501),
    ("cogniverse-phoenix", NAMESPACE, 6006, 6006),
    ("cogniverse-phoenix", NAMESPACE, 4317, 4317),
    ("cogniverse-llm", NAMESPACE, 11434, 11434),
    ("argo-server", "argo", 2746, 2746),
]

PID_FILE = "/tmp/cogniverse-port-forwards.pids"

_port_forward_procs: list[subprocess.Popen] = []


def _start_single_port_forward(
    svc_name: str, ns: str, local_port: int, svc_port: int,
) -> subprocess.Popen:
    """Start a self-restarting port-forward as a detached daemon.

    Wraps kubectl port-forward in a shell loop that auto-restarts
    when the connection drops (pod restart, idle timeout, etc.).
    """
    # Shell loop: restart port-forward on exit, with 2s backoff
    shell_cmd = (
        f"while true; do "
        f"kubectl port-forward svc/{svc_name} {local_port}:{svc_port} -n {ns} "
        f">/dev/null 2>&1; "
        f"sleep 2; "
        f"done"
    )
    return subprocess.Popen(
        ["sh", "-c", shell_cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def start_port_forwards(*, skip_llm: bool = False) -> None:
    """Start kubectl port-forward for all services as detached daemons.

    Uses ``start_new_session=True`` so processes survive after the CLI exits.
    PIDs are written to ``PID_FILE`` for cross-process cleanup.
    """
    import os

    pids: list[int] = []

    for svc_name, ns, local_port, svc_port in PORT_FORWARD_SPECS:
        if skip_llm and svc_name == "cogniverse-llm":
            continue
        proc = _start_single_port_forward(svc_name, ns, local_port, svc_port)
        pids.append(proc.pid)
        _port_forward_procs.append(proc)

    with open(PID_FILE, "w") as f:
        f.write("\n".join(str(p) for p in pids))


def restart_dead_port_forwards() -> None:
    """Check for dead port-forward processes and restart them."""
    import os

    new_pids: list[int] = []
    alive_procs: list[subprocess.Popen] = []

    for proc in _port_forward_procs:
        if proc.poll() is not None:
            # Process is dead — find its spec and restart
            cmd_str = " ".join(proc.args) if isinstance(proc.args, list) else str(proc.args)
            for svc_name, ns, local_port, svc_port in PORT_FORWARD_SPECS:
                if f"{local_port}:{svc_port}" in cmd_str:
                    new_proc = _start_single_port_forward(svc_name, ns, local_port, svc_port)
                    alive_procs.append(new_proc)
                    new_pids.append(new_proc.pid)
                    break
        else:
            alive_procs.append(proc)
            new_pids.append(proc.pid)

    _port_forward_procs.clear()
    _port_forward_procs.extend(alive_procs)

    with open(PID_FILE, "w") as f:
        f.write("\n".join(str(p) for p in new_pids))


def stop_port_forwards() -> None:
    """Stop all background port-forward processes.

    Kills the entire process group (shell wrapper + kubectl children).
    """
    import os
    import signal

    # Kill process groups from in-process list
    for proc in _port_forward_procs:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    _port_forward_procs.clear()

    # Also kill from PID file (for cross-process cleanup)
    if os.path.exists(PID_FILE):
        with open(PID_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        pgid = os.getpgid(int(line))
                        os.killpg(pgid, signal.SIGTERM)
                    except (ProcessLookupError, ValueError, PermissionError):
                        pass
        os.unlink(PID_FILE)
