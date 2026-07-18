"""Foundation must not import upward into cogniverse_core.

Foundation is a lower layer than core (core depends on foundation). Foundation
code that reaches back into ``cogniverse_core`` — e.g. config/manager.py's
top-level ``from cogniverse_core.common.tenant_utils import ...`` — forms a
dependency cycle: importing ``cogniverse_foundation.config.manager`` pulled in
core at import time. The shared pure helpers (tenant identity, DSPy registries)
now live in ``cogniverse_foundation.common``; core re-exports them. This test
scans every foundation source file and fails if any reintroduces a core import.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FOUNDATION_SRC = (
    Path(__file__).resolve().parents[3]
    / "libs"
    / "foundation"
    / "cogniverse_foundation"
)

pytestmark = [pytest.mark.unit]


def _foundation_files():
    return sorted(FOUNDATION_SRC.rglob("*.py"))


def test_no_foundation_module_imports_cogniverse_core():
    offenders = []
    for path in _foundation_files():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith(("import cogniverse_core", "from cogniverse_core")):
                rel = path.relative_to(FOUNDATION_SRC.parents[1])
                offenders.append(f"{rel}:{lineno}: {stripped}")
    assert not offenders, "foundation must not import cogniverse_core:\n" + "\n".join(
        offenders
    )


def test_core_reexports_foundation_tenant_utils():
    import cogniverse_core.common.tenant_utils as ct
    import cogniverse_foundation.common.tenant_utils as ft

    for name in (
        "require_tenant_id",
        "canonical_tenant_id",
        "parse_tenant_id",
        "validate_tenant_id",
        "get_tenant_storage_path",
        "sanitize_k8s_label_value",
    ):
        assert getattr(ct, name) is getattr(ft, name)
    assert ct.SYSTEM_TENANT_ID == ft.SYSTEM_TENANT_ID == "__system__"
    # Runtime-coupled helpers stay in core.
    assert hasattr(ct, "assert_tenant_exists")
    assert hasattr(ct, "invalidate_tenant_exists")


def test_core_reexports_foundation_dspy_registry():
    import cogniverse_core.common.dspy_module_registry as cdr
    import cogniverse_foundation.common.dspy_module_registry as fdr

    assert cdr.DSPyModuleRegistry is fdr.DSPyModuleRegistry
    assert cdr.DSPyOptimizerRegistry is fdr.DSPyOptimizerRegistry
