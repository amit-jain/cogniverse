"""Deno availability probe for DSPy RLM REPL execution.

DSPy's RLM module spawns a Deno subprocess to execute LLM-generated code in a
sandboxed JavaScript runtime. If Deno is missing, RLM fails on the first call
with a confusing trace deep inside dspy. This module fails fast, at instance
construction, with an actionable error message naming the install URL.

The probe checks PATH first, then ``~/.deno/bin/deno`` (the default install
location). When found via the home-dir fallback, PATH is amended in-process so
subprocess spawns also locate the binary.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_DENO_INSTALL_DOC = (
    "Install Deno: https://docs.deno.com/runtime/getting_started/installation/ "
    "or in the runtime container build (libs/runtime/Dockerfile). "
    "Set COGNIVERSE_RLM_SKIP_DENO_CHECK=1 to bypass this probe (not recommended)."
)


class DenoNotInstalledError(RuntimeError):
    """Raised at RLMInference construction when Deno is required but missing."""


def is_deno_available() -> bool:
    """Return True iff a Deno binary is reachable for subprocess execution.

    Side effect: when Deno is found via ``~/.deno/bin``, prepend it to ``PATH``
    so subsequent subprocess calls (DSPy's RLM REPL) can locate it.
    """
    if shutil.which("deno"):
        return True

    home_bin = Path.home() / ".deno" / "bin"
    if (home_bin / "deno").exists():
        os.environ["PATH"] = f"{home_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        return True

    return False


def assert_deno_available() -> None:
    """Raise DenoNotInstalledError unless Deno is reachable.

    Honours ``COGNIVERSE_RLM_SKIP_DENO_CHECK=1`` for environments that need to
    boot without Deno (e.g. test collection, agents that never invoke RLM).
    """
    if os.environ.get("COGNIVERSE_RLM_SKIP_DENO_CHECK", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return

    if is_deno_available():
        return

    raise DenoNotInstalledError(
        "Deno is required for DSPy RLM REPL execution but was not found on "
        f"PATH or in ~/.deno/bin. {_DENO_INSTALL_DOC}"
    )
