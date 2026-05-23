"""Helm install/upgrade/uninstall."""

from __future__ import annotations

import subprocess
from pathlib import Path

RELEASE_NAME = "cogniverse"
from cogniverse_cli.constants import NAMESPACE  # noqa: F401


def release_exists(name: str = RELEASE_NAME, namespace: str = NAMESPACE) -> bool:
    """Check if a Helm release exists."""
    result = subprocess.run(
        ["helm", "status", name, "-n", namespace],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def helm_install(
    chart_path: Path,
    values_file: Path | list[Path],
    *,
    set_values: dict[str, str] | None = None,
    name: str = RELEASE_NAME,
    namespace: str = NAMESPACE,
    timeout: str = "10m",
) -> None:
    """Install or upgrade Helm release.

    ``values_file`` accepts a single Path or a list — multiple files are
    applied in order so later overlays override earlier ones. ``timeout``
    is passed through to ``helm --timeout`` (cold-start clusters with
    GB-scale model pulls need ``20m`` headroom).
    """
    action = "upgrade" if release_exists(name, namespace) else "install"
    if isinstance(values_file, Path):
        values_files = [values_file]
    else:
        values_files = list(values_file)
    cmd: list[str] = [
        "helm",
        action,
        name,
        str(chart_path),
        "--namespace",
        namespace,
        "--create-namespace",
        "--timeout",
        timeout,
    ]
    for vf in values_files:
        cmd.extend(["-f", str(vf)])
    if set_values:
        for key, value in set_values.items():
            cmd.extend(["--set", f"{key}={value}"])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Print stderr but don't crash — let the CLI continue to health checks
        import sys

        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"helm {action} failed (exit {result.returncode})")


def helm_uninstall(name: str = RELEASE_NAME, namespace: str = NAMESPACE) -> None:
    """Uninstall Helm release if it exists."""
    if not release_exists(name, namespace):
        return
    subprocess.run(
        ["helm", "uninstall", name, "--namespace", namespace],
        check=True,
    )
