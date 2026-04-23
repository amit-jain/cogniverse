"""Helm install/upgrade/uninstall."""

from __future__ import annotations

import subprocess
from pathlib import Path

RELEASE_NAME = "cogniverse"
NAMESPACE = "cogniverse"


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
    values_file: Path,
    *,
    set_values: dict[str, str] | None = None,
    name: str = RELEASE_NAME,
    namespace: str = NAMESPACE,
) -> None:
    """Install or upgrade Helm release.

    Uses ``install`` if the release does not exist yet, ``upgrade`` if it
    does.
    """
    action = "upgrade" if release_exists(name, namespace) else "install"
    cmd: list[str] = [
        "helm",
        action,
        name,
        str(chart_path),
        "--namespace",
        namespace,
        "--create-namespace",
        "-f",
        str(values_file),
        "--timeout",
        "10m",
    ]
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
