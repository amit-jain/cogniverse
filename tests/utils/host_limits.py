"""Host kernel limits the K3s-backed integration fixtures depend on.

A K3s watches its containerd image store, manifests and config, so each one
holds roughly ``K3S_INOTIFY_INSTANCES`` inotify instances out of a per-user
budget. Exhausting that budget surfaces inside K3s as ``failed to create image
import watcher ... too many open files``, which reads as a file-descriptor
problem and is not one.
"""

from pathlib import Path

K3S_INOTIFY_INSTANCES = 35
RECOMMENDED_MAX_USER_INSTANCES = 512

_MAX_USER_INSTANCES = Path("/proc/sys/fs/inotify/max_user_instances")


def inotify_shortfall(
    *, cap: int, in_use: int, needed: int = K3S_INOTIFY_INSTANCES
) -> int:
    """Instances still required beyond the free headroom; 0 when there is room."""
    return max(0, needed - (cap - in_use))


def inotify_cap() -> int:
    return int(_MAX_USER_INSTANCES.read_text().strip())


def inotify_in_use() -> int:
    return sum(1 for fd in Path("/proc").glob("[0-9]*/fd/*") if _points_at_inotify(fd))


def _points_at_inotify(fd: Path) -> bool:
    try:
        return fd.readlink().name == "anon_inode:inotify"
    except OSError:
        return False


def inotify_preflight_message(
    *, cap: int, in_use: int, needed: int = K3S_INOTIFY_INSTANCES
) -> str | None:
    """The actionable message when a new K3s cannot fit, else ``None``."""
    if inotify_shortfall(cap=cap, in_use=in_use, needed=needed) == 0:
        return None
    return (
        f"fs.inotify.max_user_instances is {cap} with {in_use} in use, "
        f"leaving {cap - in_use} for a K3s that needs {needed}. Raise it: "
        f"sudo sysctl -w fs.inotify.max_user_instances="
        f"{RECOMMENDED_MAX_USER_INSTANCES}"
    )
