"""Memory integration test fixtures.

Re-exports the ``dspy_lm`` and ``_dspy_lm_instance`` fixtures from
``tests/agents/integration/conftest.py`` so memory-side integration
tests can exercise the real DSPy LM (no stubs). Pytest only walks UP
from a test file's directory; without this re-export, tests under
``tests/memory/integration/`` would not see the LM fixture defined
under ``tests/agents/integration/``.
"""

from __future__ import annotations

# Re-export both fixtures by name. Pytest discovers them via the symbol
# table of the conftest module, so a plain import is enough.
from tests.agents.integration.conftest import (  # noqa: F401
    _dspy_lm_instance,
    dspy_lm,
)
