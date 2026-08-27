"""Single-run lock for the e2e cluster.

One e2e run may touch the cluster at a time. Concurrent runs multiply LM and
ingestion load on a serving stack whose memory scales with concurrency; on a
unified-memory host the GPU pool is pinned and unswappable, so the kernel
reclaims from every other process until it OOMs the desktop.

``flock`` is the mutex, so a killed run never strands the lock. The holder pid
is recorded in the file as well, which is how a lock taken by the batched
runner (plain bash, no flock) is honoured, and how a conflict names its owner.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

from tests.utils.vllm_sidecar import (
    _EXACT_MODEL_LEASE_DIR,
    EXACT_MODEL_LABEL,
    reap_dead_owner_containers,
)

DEFAULT_LOCK_PATH = "/tmp/cogniverse_e2e_run.lock"
_GTT_USED_LIMIT_BYTES = 2 * 1024**3
_TEST_CONTAINER_PREFIX = "cogniverse-test-"

_HELD: dict[str, int] = {}


class E2ERunLockError(RuntimeError):
    """Another e2e run owns the cluster, or the lock could not be taken."""


def default_lock_path() -> Path:
    return Path(os.environ.get("E2E_LOCK_FILE", DEFAULT_LOCK_PATH))


def _parent_pid(pid: int) -> int | None:
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    return None


def _ancestor_pids() -> set[int]:
    seen: set[int] = set()
    pid = os.getppid()
    while pid and pid > 1 and pid not in seen:
        seen.add(pid)
        pid = _parent_pid(pid) or 0
    return seen


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _recorded_pid(fd: int) -> int | None:
    os.lseek(fd, 0, os.SEEK_SET)
    raw = os.read(fd, 64).decode(errors="replace").strip()
    return int(raw) if raw.isdigit() else None


def _conflict(holder: int | None, path: Path) -> E2ERunLockError:
    who = f"pid {holder}" if holder else "an unidentified process"
    cmd = ""
    if holder:
        try:
            cmd = (
                Path(f"/proc/{holder}/cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode()
            )
        except OSError:
            cmd = ""
    return E2ERunLockError(
        f"an e2e run is already in flight ({who}); refusing to start a second one "
        f"against the same cluster. lock={path}"
        + (f" cmd={cmd.strip()}" if cmd.strip() else "")
        + ". Concurrent e2e runs have OOMed this host. Wait for it, or kill it."
    )


def _docker_detail(result: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(part for part in (result.stdout, result.stderr) if part).strip()


def _run_docker(
    command: list[str], *, timeout: int = 30
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"docker command failed while preparing the e2e stack: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _exact_model_rows() -> list[tuple[str, str]]:
    result = _run_docker(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"label={EXACT_MODEL_LABEL}",
            "--format",
            "{{.ID}}\t{{.Names}}",
        ]
    )
    if result.returncode != 0:
        detail = _docker_detail(result)
        raise RuntimeError(
            f"docker could not list exact-model containers: "
            f"{detail or f'exit {result.returncode}'}"
        )
    rows: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        rows.append((parts[0], parts[1]))
    return rows


def _live_exact_model_leases() -> dict[str, int]:
    try:
        entries = sorted(_EXACT_MODEL_LEASE_DIR.iterdir())
    except FileNotFoundError:
        return {}
    leases: dict[str, int] = {}
    for entry in entries:
        container, _, holder_pid = entry.name.rpartition(".")
        if not container or not holder_pid.isdigit():
            continue
        if os.path.exists(f"/proc/{holder_pid}"):
            leases[container] = int(holder_pid)
        else:
            entry.unlink(missing_ok=True)
    return leases


def classify_exact_model_containers(
    rows: Iterable[tuple[str, str]],
    live_leases: set[str],
) -> tuple[set[str], set[str]]:
    leased: set[str] = set()
    unleased: set[str] = set()
    for _container_id, name in rows:
        if name in live_leases:
            leased.add(name)
        else:
            unleased.add(name)
    return leased, unleased


def _remove_exact_model_container(container: str) -> None:
    result = _run_docker(["docker", "rm", "-f", container])
    detail = _docker_detail(result)
    if result.returncode == 0 or "No such container" in detail:
        return
    raise RuntimeError(
        f"docker could not remove exact-model container {container!r}: "
        f"{detail or f'exit {result.returncode}'}"
    )


def _running_test_container_names() -> list[str]:
    result = _run_docker(["docker", "ps", "--format", "{{.Names}}"])
    if result.returncode != 0:
        detail = _docker_detail(result)
        raise RuntimeError(
            f"docker could not list running test containers: "
            f"{detail or f'exit {result.returncode}'}"
        )
    return sorted(
        name
        for name in result.stdout.splitlines()
        if name.startswith(_TEST_CONTAINER_PREFIX)
    )


def _read_gtt_used_bytes() -> int:
    try:
        return int(Path("/sys/class/drm/card1/device/mem_info_gtt_used").read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "cannot read /sys/class/drm/card1/device/mem_info_gtt_used: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def ensure_e2e_gpu_residency(
    *,
    gtt_reader: Callable[[], int] = _read_gtt_used_bytes,
) -> None:
    """Drop stale test-owned GPU residents before the cluster starts."""
    reap_dead_owner_containers()

    exact_rows = _exact_model_rows()
    live_leases = _live_exact_model_leases()
    leased, unleased = classify_exact_model_containers(exact_rows, set(live_leases))

    for container in sorted(unleased):
        _remove_exact_model_container(container)

    if leased:
        container = sorted(leased)[0]
        pytest.fail(
            f"exact-model sidecar {container!r} is leased by live pytest pid "
            f"{live_leases[container]}; refusing to start the e2e stack",
            pytrace=False,
        )

    gtt_used_bytes = gtt_reader()
    if gtt_used_bytes > _GTT_USED_LIMIT_BYTES:
        gtt_gib = gtt_used_bytes / 1024**3
        running = _running_test_container_names()
        pytest.fail(
            f"GTT remains at {gtt_gib:.2f} GiB after reclaim; running "
            f"cogniverse-test-* containers: {', '.join(running) if running else '<none>'}",
            pytrace=False,
        )


def acquire(lock_path: str | os.PathLike[str] | None = None) -> bool:
    """Take the run lock.

    Returns True when this process now owns it and must ``release`` it, False
    when an ancestor (the batched runner) already owns it. Raises
    ``E2ERunLockError`` when another run holds it or the lock is unwritable.
    """
    path = Path(lock_path) if lock_path is not None else default_lock_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    except OSError as exc:
        raise E2ERunLockError(f"cannot take the e2e run lock at {path}: {exc}") from exc

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        holder = _recorded_pid(fd)
        os.close(fd)
        if holder is not None and holder in _ancestor_pids():
            return False
        raise _conflict(holder, path)

    holder = _recorded_pid(fd)
    if holder is not None and holder != os.getpid() and _pid_alive(holder):
        inherited = holder in _ancestor_pids()
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        if inherited:
            return False
        raise _conflict(holder, path)

    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, f"{os.getpid()}\n".encode())
    os.fsync(fd)
    _HELD[str(path)] = fd
    return True


def release(lock_path: str | os.PathLike[str] | None = None) -> None:
    """Drop the run lock if this process owns it; leave any other lock intact."""
    path = Path(lock_path) if lock_path is not None else default_lock_path()
    fd = _HELD.pop(str(path), None)
    try:
        owner = int(path.read_text().split()[0])
    except (OSError, IndexError, ValueError):
        owner = None
    if owner == os.getpid():
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    if fd is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
