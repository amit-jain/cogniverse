"""Guard in-pod telemetry scripts against bare singleton bootstraps."""

from __future__ import annotations

import ast
from pathlib import Path

EXPECTED_TELEMETRY_SCRIPT_COUNT = 10
PRELUDE_MARKERS = (
    "IN_POD_TELEMETRY_PRELUDE",
    "resolve_library_env_defaults()['telemetry_otlp_endpoint']",
    'resolve_library_env_defaults()["telemetry_otlp_endpoint"]',
)


def _telemetry_script_sources(path: Path) -> list[tuple[int, str]]:
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    scripts: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        value = None
        if isinstance(node, ast.Assign):
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            value = node.value
        if value is None:
            continue
        expr = ast.get_source_segment(source, value)
        if expr is None or "get_telemetry_manager" not in expr:
            continue
        scripts.append((node.lineno, expr))
    return scripts


def _resolves_endpoint_first(script_source: str) -> bool:
    telemetry_index = script_source.find("get_telemetry_manager")
    if telemetry_index == -1:
        return False
    prefix_index = min(
        (
            script_source.find(marker)
            for marker in PRELUDE_MARKERS
            if script_source.find(marker) != -1
        ),
        default=-1,
    )
    return prefix_index != -1 and prefix_index < telemetry_index


def test_in_pod_telemetry_scripts_resolve_endpoint_first():
    scripts: list[tuple[Path, int, str]] = []
    guard_path = Path(__file__).resolve()
    e2e_dir = guard_path.parent
    for path in sorted(e2e_dir.glob("*.py")):
        if path.resolve() in {guard_path, e2e_dir / "conftest.py"}:
            continue
        for line, script_source in _telemetry_script_sources(path):
            scripts.append((path, line, script_source))

    assert len(scripts) == EXPECTED_TELEMETRY_SCRIPT_COUNT, (
        f"expected {EXPECTED_TELEMETRY_SCRIPT_COUNT} telemetry scripts, "
        f"found {len(scripts)}: "
        + "; ".join(f"{path.name}:{line}" for path, line, _ in scripts)
    )

    offenders = [
        f"{path.name}:{line}"
        for path, line, script_source in scripts
        if not _resolves_endpoint_first(script_source)
    ]
    assert offenders == [], (
        "in-pod telemetry scripts must configure the deployment endpoint "
        f"before the first get_telemetry_manager() call: {offenders}"
    )
