"""Guard the core -> runtime layer boundary.

``cogniverse_core.registries.registry`` previously imported
``cogniverse_runtime.ingestion.strategy`` at module load, inverting the layer
(core depends on runtime). It happened to resolve because runtime is
co-installed, but any context where core loads without runtime breaks.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.unit
def test_core_registry_module_does_not_load_runtime():
    """Importing the core registry must NOT pull in cogniverse_runtime modules."""
    script = (
        "import sys\n"
        "import cogniverse_core.registries.registry  # noqa: F401\n"
        "assert 'cogniverse_runtime.ingestion.strategy' not in sys.modules, (\n"
        "    'core.registries.registry imported runtime at module load — "
        "the layer inversion is back'\n"
        ")\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr or result.stdout
