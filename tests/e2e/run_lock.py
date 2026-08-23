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
from pathlib import Path

DEFAULT_LOCK_PATH = "/tmp/cogniverse_e2e_run.lock"

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
