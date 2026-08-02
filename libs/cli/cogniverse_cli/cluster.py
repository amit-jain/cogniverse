"""k3d cluster lifecycle and prerequisite checks."""

from __future__ import annotations

import errno
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import dataclass

PREREQUISITES = ["docker", "kubectl", "helm"]
CLUSTER_NAME = "cogniverse"

# Host ports k3d's loadbalancer publishes — each must match the chart's
# service.nodePort. 28xxx/26xxx avoid collisions with common dev ports,
# 29xxx are inference sidecars that e2e tests probe directly.
DEFAULT_PORTS = [
    8080,
    19071,
    28000,
    28501,
    26006,
    4317,
    11434,
    2746,
    29001,
    29002,
    29004,
    29005,
    29006,
    29010,
    29011,
]

_SERVICE_BY_NODE_PORT = {
    4317: "OTLP",
    6443: "Kubernetes API",
    8080: "Vespa",
    11434: "LLM (Ollama)",
    19071: "Vespa",
    26006: "Phoenix",
    2746: "Argo",
    28000: "Runtime",
    28501: "Dashboard",
    29001: "inference sidecar",
    29002: "inference sidecar",
    29004: "inference sidecar",
    29005: "inference sidecar",
    29006: "inference sidecar",
    29010: "inference sidecar",
    29011: "inference sidecar",
}


class ClusterStartError(RuntimeError):
    """A cluster start failure safe to show directly to an operator."""


@dataclass(frozen=True)
class _PortBinding:
    host_ip: str
    host_port: int
    node_port: int
    protocol: str

    @property
    def service(self) -> str:
        return _SERVICE_BY_NODE_PORT.get(
            self.node_port, f"cluster port {self.node_port}"
        )


def _get_arch() -> str:
    """Return architecture string for download URLs (amd64/arm64)."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "amd64"
    if machine in ("aarch64", "arm64"):
        return "arm64"
    return machine


def _get_install_instructions(tool: str) -> dict[str, str]:
    """Return human-readable install instructions per platform."""
    arch = _get_arch()
    return {
        "k3d": {
            "darwin": "brew install k3d",
            "linux": "curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash",
        },
        "helm": {
            "darwin": "brew install helm",
            "linux": "curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash",
        },
        "kubectl": {
            "darwin": "brew install kubectl",
            "linux": (
                f"curl -LO 'https://dl.k8s.io/release/"
                f"$(curl -sL https://dl.k8s.io/release/stable.txt)"
                f"/bin/linux/{arch}/kubectl' && "
                f"chmod +x kubectl && sudo mv kubectl /usr/local/bin/"
            ),
        },
        "docker": {
            "darwin": "Install Docker Desktop: https://docs.docker.com/desktop/install/mac-install/",
            "linux": "curl -fsSL https://get.docker.com | sh",
        },
    }.get(tool, {})


def check_prerequisites(*, require_k3d: bool = True) -> list[str]:
    """Return list of missing prerequisites."""
    tools = list(PREREQUISITES)
    if require_k3d:
        tools.append("k3d")
    return [tool for tool in tools if shutil.which(tool) is None]


def get_install_commands(missing: list[str]) -> list[tuple[str, str]]:
    """Return (tool, install_command) pairs for missing prerequisites.

    Does NOT run anything — just returns the commands for the user to review.
    """
    os_name = platform.system().lower()
    commands = []
    for tool in missing:
        instructions = _get_install_instructions(tool)
        cmd = instructions.get(os_name)
        if cmd:
            commands.append((tool, cmd))
        else:
            commands.append((tool, f"Install {tool} manually for {os_name}"))
    return commands


def install_prerequisite(tool: str) -> bool:
    """Attempt to install a single prerequisite. Returns True on success."""
    os_name = platform.system().lower()
    instructions = _get_install_instructions(tool)
    cmd_str = instructions.get(os_name)
    if not cmd_str:
        return False

    # Use brew on macOS, shell on Linux
    if os_name == "darwin" and shutil.which("brew"):
        cmd = cmd_str.split()
    else:
        cmd = ["bash", "-c", cmd_str]

    try:
        result = subprocess.run(cmd, timeout=300, check=False)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def install_missing_prerequisites(missing: list[str]) -> list[str]:
    """Install missing tools after showing what will be installed.

    Returns list of tools that still can't be found after install attempts.
    """
    still_missing = []
    for tool in missing:
        if tool == "docker":
            still_missing.append(tool)
            continue
        if install_prerequisite(tool):
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
    """Check if a k3d cluster with the given name exists.

    A missing or hung k3d binary reads as "no cluster" — the same guard
    has_existing_k8s uses — so `up` reaches its install-prerequisites
    prompt instead of dying on a FileNotFoundError traceback.
    """
    try:
        result = subprocess.run(
            ["k3d", "cluster", "list", name],
            capture_output=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _parse_port_csv(value: str | None) -> list[int]:
    """Parse a comma-separated list of integer ports, ignoring blanks.

    A non-numeric entry aborts with a one-line config error instead of a
    bare ValueError traceback mid-`cogniverse up`.
    """
    if not value:
        return []
    out: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(int(raw))
        except ValueError:
            raise SystemExit(
                f"Invalid port {raw!r} in {value!r} — "
                f"COGNIVERSE_K3D_*PORTS entries must be integers"
            ) from None
    return out


def create_cluster(
    name: str = CLUSTER_NAME,
    ports: list[int | str] | None = None,
    *,
    exclude_ports: list[int] | None = None,
    workspace_path: str | None = None,
    share_hf_cache: bool = True,
    share_host_storage: bool = True,
) -> None:
    """Create a k3d cluster with port mappings and workspace volume.

    Each port gets a ``-p`` flag mapping it through the load balancer.
    Use *exclude_ports* to skip ports that are already in use on the host
    (e.g., 11434 when host LM is running).
    *workspace_path* mounts the project root into the k3d node at
    ``/cogniverse-src`` for devMode volume access.

    The dev-convenience flags default ON for ``cogniverse up``. The
    deployment-lifecycle test fixture flips them OFF so the test
    cluster mirrors the production deployment shape (built images,
    fresh data, no host bind-mounts):

    - ``share_hf_cache``: mount ``~/.cache/huggingface`` so inference
      pods reuse on-host model downloads instead of re-pulling.
    - ``share_host_storage``: mount ``~/.local/share/cogniverse`` so
      Vespa + Phoenix data survives cluster recreation.

    Port set resolution (highest precedence first):

    - ``ports`` argument (explicit) — full override; ``[]`` disables
      loadbalancer mappings entirely (e.g. tests that use
      ``kubectl port-forward`` instead).
    - ``COGNIVERSE_K3D_PORTS`` env (comma-separated ints) — full override.
    - ``DEFAULT_PORTS`` plus the env ``COGNIVERSE_K3D_EXTRA_PORTS``
      (additive).

    Then ``exclude_ports`` and the env ``COGNIVERSE_K3D_EXCLUDE_PORTS``
    are subtracted.

    On AMD hosts (/dev/kfd present) bind-mounts /dev/kfd and /dev/dri
    into the k3d server so in-cluster pods can enumerate the GPU.
    """
    if ports is None:
        env_override = _parse_port_csv(os.environ.get("COGNIVERSE_K3D_PORTS"))
        if env_override:
            ports = env_override
        else:
            ports = list(DEFAULT_PORTS) + _parse_port_csv(
                os.environ.get("COGNIVERSE_K3D_EXTRA_PORTS")
            )
    env_exclude = _parse_port_csv(os.environ.get("COGNIVERSE_K3D_EXCLUDE_PORTS"))
    drop = set(env_exclude)
    if exclude_ports:
        drop.update(exclude_ports)
    if drop:
        ports = [p for p in ports if p not in drop]
    cmd = [
        "k3d",
        "cluster",
        "create",
        name,
        # Allow any port as NodePort (default range is 30000-32767)
        "--k3s-arg",
        "--service-node-port-range=1-65535@server:0",
    ]
    if workspace_path:
        cmd.extend(["--volume", f"{workspace_path}:/cogniverse-src@server:0"])
    if os.path.exists("/dev/kfd"):
        cmd.extend(["--volume", "/dev/kfd:/dev/kfd@server:0"])
    if os.path.isdir("/dev/dri"):
        cmd.extend(["--volume", "/dev/dri:/dev/dri@server:0"])
    if share_hf_cache:
        host_hf_cache = os.path.expanduser("~/.cache/huggingface")
        if os.path.isdir(host_hf_cache):
            cmd.extend(["--volume", f"{host_hf_cache}:/host-hf-cache@server:0"])
    if share_host_storage:
        host_state = os.path.expanduser("~/.local/share/cogniverse")
        os.makedirs(host_state, exist_ok=True)
        cmd.extend(["--volume", f"{host_state}:/host-data@server:0"])
    for port in ports:
        # An int maps host:node 1:1; a "host:node" string maps a different
        # host port onto a chart NodePort (e.g. "33000:28000" — used by the
        # e2e stack so its host ports never collide with a dev cluster's).
        mapping = str(port)
        if ":" not in mapping:
            mapping = f"{mapping}:{mapping}"
        cmd.extend(["-p", f"{mapping}@loadbalancer"])
    subprocess.run(cmd, check=True, timeout=120)
    if not pin_coredns_upstreams(name):
        raise ClusterStartError(
            f"Could not pin CoreDNS upstreams for k3d cluster {name!r}"
        )


_COREDNS_HOST_FORWARD = "forward . /etc/resolv.conf"
_COREDNS_PINNED_FORWARD = "forward . 1.1.1.1 8.8.8.8"


def pinned_corefile(configmap_yaml: str) -> str | None:
    """The coredns ConfigMap YAML with the host-resolver forward replaced by
    pinned public upstreams, or ``None`` when no rewrite is needed."""
    if _COREDNS_HOST_FORWARD not in configmap_yaml:
        return None
    return configmap_yaml.replace(_COREDNS_HOST_FORWARD, _COREDNS_PINNED_FORWARD)


def pin_coredns_upstreams(name: str = CLUSTER_NAME, *, timeout_s: float = 60.0) -> bool:
    """Point the cluster's CoreDNS at pinned public resolvers.

    k3d's CoreDNS forwards to the host's ``/etc/resolv.conf``; on hosts whose
    resolver is a dead/localhost stub every pod's external DNS fails — the
    vLLM pods crashloop and their NodePorts flap (the serverlb accepts TCP
    while the upstream is dead). Idempotent: returns True once the Corefile
    is pinned (already or by this call), False when the coredns configmap
    never appeared within ``timeout_s``.
    """
    ctx = f"k3d-{name}"
    get_cmd = [
        "kubectl",
        "--context",
        ctx,
        "-n",
        "kube-system",
        "get",
        "configmap",
        "coredns",
        "-o",
        "yaml",
    ]
    deadline = time.time() + timeout_s
    while True:
        cm = subprocess.run(
            get_cmd, capture_output=True, text=True, timeout=30, check=False
        )
        if cm.returncode == 0:
            break
        if time.time() >= deadline:
            return False
        time.sleep(2)

    patched = pinned_corefile(cm.stdout or "")
    if patched is None:
        return True
    subprocess.run(
        ["kubectl", "--context", ctx, "apply", "-f", "-"],
        input=patched,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    subprocess.run(
        [
            "kubectl",
            "--context",
            ctx,
            "-n",
            "kube-system",
            "rollout",
            "restart",
            "deployment/coredns",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return True


def stop_cluster(name: str = CLUSTER_NAME) -> None:
    """Stop a k3d cluster's containers without deleting anything.

    Frees the RAM/GPU its pods hold while volumes stay intact;
    ``start_cluster`` resumes it. This is the supported way to pause one
    cluster on a host that cannot run the dev and e2e stacks together.
    """
    subprocess.run(["k3d", "cluster", "stop", name], check=True, timeout=300)


def _cluster_description(name: str, subject: str) -> dict:
    """Return one cluster's k3d JSON description or a display-safe error."""
    try:
        result = subprocess.run(
            ["k3d", "cluster", "list", name, "-o", "json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        suffix = f": {detail}" if detail else ""
        raise ClusterStartError(
            f"Could not inspect {subject} for k3d cluster {name!r}{suffix}"
        ) from None
    except FileNotFoundError:
        raise ClusterStartError("k3d not found on PATH") from None
    except subprocess.TimeoutExpired:
        raise ClusterStartError(
            f"Timed out inspecting {subject} for k3d cluster {name!r}"
        ) from None

    try:
        clusters = json.loads(result.stdout or "[]")
        return next(item for item in clusters if item.get("name") == name)
    except (json.JSONDecodeError, StopIteration, TypeError, AttributeError):
        raise ClusterStartError(
            f"Could not inspect {subject} for k3d cluster {name!r}: "
            "k3d returned an invalid cluster description"
        ) from None


def _stopped_loadbalancer_bindings(name: str) -> list[_PortBinding]:
    """Read host bindings that Docker must claim when k3d starts the cluster."""
    cluster = _cluster_description(name, "port mappings")
    bindings: list[_PortBinding] = []
    for node in cluster.get("nodes") or []:
        if node.get("role") != "loadbalancer":
            continue
        if (node.get("State") or {}).get("Running", False):
            continue
        for target, published in (node.get("portMappings") or {}).items():
            raw_node_port, _, protocol = target.partition("/")
            try:
                node_port = int(raw_node_port)
            except ValueError:
                raise ClusterStartError(
                    f"Could not inspect port mappings for k3d cluster {name!r}: "
                    f"invalid load balancer target {target!r}"
                ) from None
            for mapping in published or []:
                try:
                    host_port = int(mapping["HostPort"])
                except (KeyError, TypeError, ValueError):
                    raise ClusterStartError(
                        f"Could not inspect port mappings for k3d cluster {name!r}: "
                        f"invalid host binding for {target!r}"
                    ) from None
                bindings.append(
                    _PortBinding(
                        host_ip=mapping.get("HostIp", ""),
                        host_port=host_port,
                        node_port=node_port,
                        protocol=protocol or "tcp",
                    )
                )
    return bindings


def _bind_addresses(host_ip: str) -> list[tuple[socket.AddressFamily, str]]:
    if host_ip == "":
        addresses = [(socket.AF_INET, "0.0.0.0")]
        if socket.has_ipv6:
            addresses.append((socket.AF_INET6, "::"))
        return addresses
    if ":" in host_ip:
        return [(socket.AF_INET6, host_ip)]
    return [(socket.AF_INET, host_ip)]


def _binding_is_available(binding: _PortBinding) -> bool:
    socket_type = socket.SOCK_DGRAM if binding.protocol == "udp" else socket.SOCK_STREAM
    for family, host_ip in _bind_addresses(binding.host_ip):
        try:
            with socket.socket(family, socket_type) as probe:
                if family == socket.AF_INET6:
                    probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                probe.bind((host_ip, binding.host_port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                return False
            if family == socket.AF_INET6 and exc.errno in {
                errno.EAFNOSUPPORT,
                errno.EADDRNOTAVAIL,
            }:
                continue
            raise ClusterStartError(
                f"Could not check host port {binding.host_port}: {exc}"
            ) from None
    return True


def _cluster_start_conflicts(name: str) -> list[_PortBinding]:
    return [
        binding
        for binding in _stopped_loadbalancer_bindings(name)
        if not _binding_is_available(binding)
    ]


def _port_conflict_message(name: str, conflicts: list[_PortBinding]) -> str:
    conflict_lines = [
        f"Host port {binding.host_port} required by {binding.service} is in use."
        for binding in conflicts
    ]
    ports = ", ".join(str(binding.host_port) for binding in conflicts)
    return "\n".join(
        [
            f"Cannot start k3d cluster {name!r}:",
            *conflict_lines,
            "k3d cannot remove or remap published ports on an existing cluster.",
            "Free or reconfigure the host listener, then retry:",
            f"  cogniverse start --name {name}",
            f"Or recreate the cluster with host port {ports} excluded or remapped.",
        ]
    )


def _verify_loadbalancer_network(name: str) -> None:
    """Ensure a started load balancer can resolve and reach cluster nodes."""
    cluster = _cluster_description(name, "load balancer network")
    network_name = (cluster.get("network") or {}).get("name")
    if not network_name:
        raise ClusterStartError(
            f"Could not inspect load balancer network for k3d cluster {name!r}: "
            "k3d returned an invalid cluster description"
        )

    load_balancers = [
        node
        for node in cluster.get("nodes") or []
        if node.get("role") == "loadbalancer"
    ]
    if not load_balancers and cluster.get("hasLoadbalancer", False):
        raise ClusterStartError(
            f"Could not inspect load balancer network for k3d cluster {name!r}: "
            "k3d did not describe its load balancer"
        )

    for load_balancer in load_balancers:
        container = load_balancer.get("name", "")
        if not (load_balancer.get("State") or {}).get("Running", False):
            raise ClusterStartError(
                f"k3d cluster {name!r} started, but load balancer "
                f"{container!r} is not running."
            )
        if network_name in (load_balancer.get("Networks") or []):
            continue
        raise ClusterStartError(
            "\n".join(
                [
                    f"k3d cluster {name!r} started, but load balancer "
                    f"{container!r} is not attached to network {network_name!r}.",
                    "The cluster API and published services are unavailable.",
                    "Repair the existing cluster without recreating it:",
                    f"  docker network connect {network_name} {container}",
                    f"  docker restart {container}",
                ]
            )
        )


def start_cluster(name: str = CLUSTER_NAME) -> None:
    """Start a previously stopped k3d cluster (volumes intact).

    Checks the stopped load balancer's stored host bindings before k3d starts
    any nodes, preventing a bind conflict from leaving a partially running
    cluster.

    Re-pins CoreDNS upstreams on every start so clusters created before the
    pin existed converge on the working DNS configuration.
    """
    conflicts = _cluster_start_conflicts(name)
    if conflicts:
        raise ClusterStartError(_port_conflict_message(name, conflicts))
    try:
        subprocess.run(
            ["k3d", "cluster", "start", name],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise ClusterStartError(
            f"Could not start k3d cluster {name!r}{suffix}"
        ) from None
    except FileNotFoundError:
        raise ClusterStartError("k3d not found on PATH") from None
    except subprocess.TimeoutExpired:
        raise ClusterStartError(f"Timed out starting k3d cluster {name!r}") from None
    _verify_loadbalancer_network(name)
    if not pin_coredns_upstreams(name):
        raise ClusterStartError(
            f"Could not pin CoreDNS upstreams for k3d cluster {name!r}"
        )


def list_cluster_states() -> list[dict]:
    """Name and running-state of every k3d cluster on the host."""
    result = subprocess.run(
        ["k3d", "cluster", "list", "-o", "json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return [
        {
            "name": cluster.get("name", ""),
            "servers_running": cluster.get("serversRunning", 0),
            "servers_count": cluster.get("serversCount", 0),
        }
        for cluster in json.loads(result.stdout or "[]")
    ]


def delete_cluster(name: str = CLUSTER_NAME) -> None:
    """Delete a k3d cluster."""
    subprocess.run(
        ["k3d", "cluster", "delete", name],
        check=True,
        timeout=60,
    )


# Port-forward specs: (service, namespace, local_port, service_port).
# Cogniverse services use the k3d loadbalancer; only argo (different
# namespace, no NodePort) needs an explicit port-forward.
PORT_FORWARD_SPECS: list[tuple[str, str, int, int]] = [
    ("argo-server", "argo", 2746, 2746),
]

PID_FILE = "/tmp/cogniverse-port-forwards.pids"

_port_forward_procs: list[subprocess.Popen] = []


def _start_single_port_forward(
    svc_name: str,
    ns: str,
    local_port: int,
    svc_port: int,
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


# Grace given to a SIGTERM'd daemon group before it is SIGKILLed. The daemon
# is a shell restart-loop that stays blocked in its child and does not exit on
# SIGTERM, so the group must be force-killed to actually reap it.
_REAP_GRACE_SECONDS = 3.0


def _pgroup_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_recorded_daemons() -> None:
    """Terminate the port-forward daemons recorded in ``PID_FILE``.

    Each daemon is spawned as its own session leader (``start_new_session``),
    so its PID is also its process-group id. The daemon's shell restart-loop
    survives ``SIGTERM`` while blocked on its child, so each group is sent
    ``SIGTERM``, given ``_REAP_GRACE_SECONDS`` to exit, then ``SIGKILL``ed. The
    file is removed once the recorded daemons are gone.
    """
    if not os.path.exists(PID_FILE):
        return

    with open(PID_FILE) as f:
        recorded = [line.strip() for line in f if line.strip()]

    pgids: list[int] = []
    for entry in recorded:
        try:
            pid = int(entry)
        except ValueError:
            continue
        try:
            pgids.append(os.getpgid(pid))
        except (ProcessLookupError, PermissionError):
            continue

    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    deadline = time.monotonic() + _REAP_GRACE_SECONDS
    while time.monotonic() < deadline:
        pgids = [pgid for pgid in pgids if _pgroup_alive(pgid)]
        if not pgids:
            break
        time.sleep(0.05)

    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    os.unlink(PID_FILE)


def start_port_forwards() -> None:
    """Start kubectl port-forward for all services as detached daemons.

    Uses ``start_new_session=True`` so processes survive after the CLI exits.
    Daemons recorded from a prior start are reaped first so a repeated start
    never orphans an earlier restart-loop still retrying its bind. PIDs are
    written to ``PID_FILE`` for cross-process cleanup.
    """

    _terminate_recorded_daemons()
    _port_forward_procs.clear()

    pids: list[int] = []

    for svc_name, ns, local_port, svc_port in PORT_FORWARD_SPECS:
        proc = _start_single_port_forward(svc_name, ns, local_port, svc_port)
        pids.append(proc.pid)
        _port_forward_procs.append(proc)

    with open(PID_FILE, "w") as f:
        f.write("\n".join(str(p) for p in pids))


def restart_dead_port_forwards() -> None:
    """Check for dead port-forward processes and restart them."""

    new_pids: list[int] = []
    alive_procs: list[subprocess.Popen] = []

    for proc in _port_forward_procs:
        if proc.poll() is not None:
            # Process is dead — find its spec and restart
            cmd_str = (
                " ".join(proc.args) if isinstance(proc.args, list) else str(proc.args)
            )
            for svc_name, ns, local_port, svc_port in PORT_FORWARD_SPECS:
                if f"{local_port}:{svc_port}" in cmd_str:
                    new_proc = _start_single_port_forward(
                        svc_name, ns, local_port, svc_port
                    )
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

    Kills the entire process group (shell wrapper + kubectl children) for the
    in-process daemons, then reaps any recorded in ``PID_FILE`` so a fresh CLI
    process (empty in-process list) still tears down daemons a prior run left.
    """
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

    _terminate_recorded_daemons()
