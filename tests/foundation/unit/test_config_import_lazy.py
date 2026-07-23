"""Importing cogniverse_foundation.config must not pull in dspy/litellm.

config is imported by nearly every process (every CLI, the worker pod). A
module-level ``import dspy`` in llm_factory cost ~2.2s of import time on every
one of them, even on paths that never build a dspy.LM. dspy is imported lazily
inside create_dspy_lm; this guards against a module-level import creeping back.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def test_config_import_does_not_pull_in_dspy():
    # A fresh interpreter so sys.modules isn't polluted by other imports.
    code = (
        "import sys\n"
        "import cogniverse_foundation.config\n"
        "heavy = [m for m in ('dspy', 'litellm') if m in sys.modules]\n"
        "sys.exit('LOADED:' + ','.join(heavy) if heavy else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        "importing cogniverse_foundation.config eagerly loaded heavy modules: "
        f"{result.stderr or result.stdout}"
    )
