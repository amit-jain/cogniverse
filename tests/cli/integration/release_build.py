"""Build the release set with ``scripts/build_packages.sh`` in a disposable Git
checkout.

Shared by the release-script and clean-install tests.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

CHECKOUT_PATHS = (
    ".gitignore",
    "pyproject.toml",
    "uv.lock",
    "LICENSE",
    "Readme.md",
    "libs",
    "scripts",
)

BUILD_TIMEOUT = 900

EXPECTED_RELEASE = {
    "cogniverse-sdk",
    "cogniverse-foundation",
    "cogniverse-core",
    "cogniverse-evaluation",
    "cogniverse-synthetic",
    "cogniverse-vespa",
    "cogniverse-agents",
    "cogniverse-telemetry-phoenix",
    "cogniverse-runtime",
    "cogniverse-dashboard",
}

GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "Release Test",
    "GIT_AUTHOR_EMAIL": "release-test@example.invalid",
    "GIT_COMMITTER_NAME": "Release Test",
    "GIT_COMMITTER_EMAIL": "release-test@example.invalid",
}


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        env={**os.environ, **GIT_ENV},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def make_checkout(root: Path, tag: str | None) -> Path:
    listed = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--",
            *CHECKOUT_PATHS,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode()
    for relative in filter(None, listed.split("\0")):
        source = REPO_ROOT / relative
        if not source.is_file():
            continue
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "release inputs")
    if tag is not None:
        git(root, "tag", "-a", tag, "-m", tag)
    return root


def build_command(repo: Path, *args: str) -> tuple[list[str], dict[str, str]]:
    env = {**os.environ, **GIT_ENV, "UV_PROJECT_ENVIRONMENT": sys.prefix}
    return [str(repo / "scripts" / "build_packages.sh"), *args], env


def run_build(repo: Path, *args: str) -> subprocess.CompletedProcess:
    command, env = build_command(repo, *args)
    return subprocess.run(
        command,
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT,
    )


def describe_run(result: subprocess.CompletedProcess) -> str:
    return (
        f"exit={result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
