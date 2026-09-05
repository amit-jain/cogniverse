"""Load ``.env/<NAME>.env`` secrets into the process environment.

``ensure_llm`` decides between a configured remote endpoint and building a
model container on this host. That decision reads credentials from the
environment, so a run without them provisions a local sidecar instead of
using the remote service. Loading here makes every pytest invocation see
them, from any working directory and from a worktree.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def env_secret_dirs(root: Path = REPO_ROOT) -> list[Path]:
    """``.env`` for this checkout, then the main checkout's.

    A git worktree has no ``.env`` of its own -- it is untracked and lives in
    the checkout that owns the repository. ``--git-common-dir`` resolves that
    checkout from inside any worktree, so the secrets are found rather than
    silently absent.
    """
    candidates = [root / ".env"]
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return candidates
    if common:
        main_checkout = (root / common).resolve().parent
        candidates.append(main_checkout / ".env")
    return [d for i, d in enumerate(candidates) if d not in candidates[:i]]


def load_env_secrets(root: Path = REPO_ROOT) -> dict[str, str]:
    """Set each unset ``.env/<NAME>.env`` value; return only what was set.

    An already-set variable always wins, so an explicit export overrides the
    file rather than the other way round.
    """
    applied: dict[str, str] = {}
    for directory in env_secret_dirs(root):
        if not directory.is_dir():
            continue
        for secret_file in sorted(directory.glob("*.env")):
            name = secret_file.stem
            if name in os.environ or name in applied:
                continue
            for line in secret_file.read_text(errors="replace").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                value = stripped
                if value.startswith(f"{name}="):
                    value = value[len(name) + 1 :]
                os.environ[name] = value
                applied[name] = value
                break
    return applied
