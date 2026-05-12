"""Per-test tenant_id derivation for the consolidated shared_vespa.

When every test shares one Vespa container (see ``tests/conftest.py::shared_vespa``),
isolation comes from a unique tenant_id per test rather than a unique container.
This module is the single source of truth for how that tenant_id is computed
from the running test's ``request.node`` — keep all the rules here so a test
that uses ``vespa_tenant`` in package A gets the same shape as one in package B.
"""

from __future__ import annotations

import re

import pytest

# Vespa schema names accept ``[a-zA-Z][a-zA-Z0-9_]*``. Module paths and test
# function names contain ``.``, ``-``, ``[``, ``]`` (parametrize IDs), so we
# normalize every non-allowed char to ``_``. Schema-name length isn't bounded
# in practice but very long names slow log scans, so we cap at 60 chars.
_VESPA_SAFE = re.compile(r"[^A-Za-z0-9_]")
_MAX_TENANT_LEN = 60


def _normalize(part: str) -> str:
    """Replace any char Vespa schema names disallow with ``_`` and collapse runs."""
    cleaned = _VESPA_SAFE.sub("_", part)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def tenant_id_for_module(request: pytest.FixtureRequest) -> str:
    """Module-scoped tenant_id. All tests in the same module share it.

    Use when the consumer fixture is module-scoped (e.g. a Mem0MemoryManager
    instantiated once per module — every test in the module talks to the
    same agent_memories_<this-tenant> schema).
    """
    module_name = request.node.module.__name__
    # Strip the ``tests.`` prefix so the tenant_id stays short.
    if module_name.startswith("tests."):
        module_name = module_name[len("tests.") :]
    raw = _normalize(module_name)
    return raw[:_MAX_TENANT_LEN]


def tenant_id_for_test(request: pytest.FixtureRequest) -> str:
    """Per-test tenant_id. Different in every test, including parametrize cases.

    Use when the consumer fixture is function-scoped and each test needs a
    blank-slate tenant — typical for schema-deploy/lifecycle tests where one
    test's deploy must not be visible to the next.
    """
    module_part = tenant_id_for_module(request)
    func_part = _normalize(request.node.name)
    combined = f"{module_part}__{func_part}"
    return combined[:_MAX_TENANT_LEN]
