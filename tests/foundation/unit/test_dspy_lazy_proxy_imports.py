"""A module that imports dspy must still import in a fresh interpreter.

dspy 3.4 puts lazy proxies for anyio, jiter, numpy and openai into
``sys.modules`` when it is imported before them. A later import of one of
their submodules through a proxy fails with a circular ImportError
(stanfordnlp/dspy#10516), so the failure depends only on import order.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.parametrize(
    "module",
    [
        # fastapi imports anyio.abc after dspy installed the anyio proxy.
        "cogniverse_agents.search_agent",
        # numpy.typing is imported after dspy installed the numpy proxy.
        "cogniverse_agents.entity_extraction_agent",
        # litellm imports openai._models after dspy installed the openai proxy.
        "cogniverse_agents.search.learned_reranker",
        "cogniverse_synthetic.dspy_modules",
    ],
)
def test_dspy_importing_module_imports_in_a_fresh_interpreter(module):
    result = _run(f"import {module}")
    assert result.returncode == 0, f"import {module} failed: {result.stderr[-2000:]}"


@pytest.mark.parametrize(
    "package",
    [
        "cogniverse_agents",
        "cogniverse_core",
        "cogniverse_foundation",
        "cogniverse_runtime",
        "cogniverse_synthetic",
    ],
)
def test_submodules_import_after_package_and_dspy(package):
    code = (
        f"import {package}\n"
        "import dspy\n"
        "import anyio.abc\n"
        "import openai._models\n"
        "from numpy.typing import NDArray\n"
        "import litellm\n"
    )
    result = _run(code)
    assert result.returncode == 0, (
        f"a submodule import after {package} and dspy failed: {result.stderr[-2000:]}"
    )


def test_runtime_reaches_litellm_after_startup_imports():
    """The first LiteLLM-backed LM call imports litellm after the app is up."""
    result = _run("import cogniverse_runtime.main\nimport litellm\n")
    assert result.returncode == 0, (
        f"litellm failed to import after cogniverse_runtime.main: "
        f"{result.stderr[-2000:]}"
    )
