"""Production code reads only attributes ``SystemConfig`` defines.

``getattr(sys_cfg, "gliner_model", None)`` read a field ``SystemConfig`` never
had, so a configured GLiNER model was silently ignored. The scan finds every
value bound from ``get_system_config()`` in ``libs/`` and ``scripts/`` and
fails on an attribute read, ``getattr`` or ``hasattr`` that names anything
``SystemConfig`` lacks.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from cogniverse_foundation.config.unified_config import SystemConfig

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
KNOWN = frozenset(field.name for field in dataclasses.fields(SystemConfig)) | frozenset(
    dir(SystemConfig)
)


def _is_system_config_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_system_config"
    )


def _bound_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return f"self.{node.attr}"
    return None


def unknown_reads(source: str) -> list[str]:
    """``line: attribute`` for each read of an attribute SystemConfig lacks."""
    tree = ast.parse(source)
    bound = {
        name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and _is_system_config_call(node.value)
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if (name := _bound_name(target)) is not None
    }

    def is_system_config(node: ast.AST) -> bool:
        return _is_system_config_call(node) or _bound_name(node) in bound

    reads = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and is_system_config(node.value):
            if node.attr not in KNOWN:
                reads.append(f"{node.lineno}: {node.attr}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "hasattr"}
            and len(node.args) >= 2
            and is_system_config(node.args[0])
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value not in KNOWN
        ):
            reads.append(f"{node.lineno}: {node.args[1].value}")
    return reads


def test_production_code_reads_only_system_config_fields():
    found = {}
    for root in ("libs", "scripts"):
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            if ".venv" in path.parts:
                continue
            reads = unknown_reads(path.read_text(encoding="utf-8"))
            if reads:
                found[str(path.relative_to(REPO_ROOT))] = reads
    assert found == {}


def test_the_detector_finds_each_unknown_read_form():
    source = """
def resolve(self):
    sys_cfg = self._config_manager.get_system_config()
    self.cfg = manager.get_system_config()
    model = getattr(sys_cfg, "gliner_model", None)
    other = sys_cfg.gliner_device
    third = self.cfg.no_such_field
    fourth = manager.get_system_config().also_missing
    fine = sys_cfg.inference_service_urls
    present = hasattr(sys_cfg, "redis_url")
    absent = hasattr(sys_cfg, "missing_flag")
    return model, other, third, fourth, fine, present, absent
"""
    assert unknown_reads(source) == [
        "5: gliner_model",
        "6: gliner_device",
        "7: no_such_field",
        "8: also_missing",
        "11: missing_flag",
    ]
