"""Helm install/upgrade/uninstall."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
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


def semver_chart_version(version: str) -> str:
    """Translate a PEP 440 git version into a valid SemVer2 chart version.

    ``0.1.dev2395+g1998ce1b5`` becomes ``0.1.0-dev.2395+g1998ce1b5`` —
    Helm requires SemVer for the chart ``version`` field, while
    ``appVersion`` is free-form and keeps the exact git version.
    """
    match = re.fullmatch(
        r"(\d+)\.(\d+)(?:\.(\d+))?(?:\.?dev(\d+))?(?:\+(.+))?", version
    )
    if not match:
        return version
    major, minor, patch, dev, local = match.groups()
    out = f"{major}.{minor}.{patch or 0}"
    if dev is not None:
        out += f"-dev.{dev}"
    if local:
        out += "+" + local.replace("_", "-")
    return out


def helm_install(
    chart_path: Path,
    values_file: Path | list[Path],
    *,
    set_values: dict[str, str] | None = None,
    name: str = RELEASE_NAME,
    namespace: str = NAMESPACE,
    timeout: str = "10m",
    chart_version: str | None = None,
) -> None:
    """Install or upgrade Helm release.

    ``values_file`` accepts a single Path or a list — multiple files are
    applied in order so later overlays override earlier ones. ``timeout``
    is passed through to ``helm --timeout`` (cold-start clusters with
    GB-scale model pulls need ``20m`` headroom). ``chart_version`` (a git
    version) repackages the chart so the release records dev provenance
    instead of the static ``Chart.yaml`` line.
    """
    action = "upgrade" if release_exists(name, namespace) else "install"
    if isinstance(values_file, Path):
        values_files = [values_file]
    else:
        values_files = list(values_file)
    chart_ref = str(chart_path)
    if chart_version:
        package_cmd = [
            "helm",
            "package",
            str(chart_path),
            "--version",
            semver_chart_version(chart_version),
            "--app-version",
            chart_version,
            "-d",
            tempfile.mkdtemp(prefix="cogniverse-chart-"),
        ]
        packaged = subprocess.run(package_cmd, capture_output=True, text=True)
        if packaged.returncode != 0:
            print(packaged.stderr, file=sys.stderr)
            raise RuntimeError(f"helm package failed (exit {packaged.returncode})")
        chart_ref = packaged.stdout.strip().rsplit(": ", 1)[-1]
    cmd: list[str] = [
        "helm",
        action,
        name,
        chart_ref,
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
