"""Every test file in a marker-gated directory must carry the gate marker.

pytest's ``--strict-markers`` rejects UNDEFINED markers but cannot detect a
MISSING one: a file with no ``unit``/``integration`` marker collects cleanly
yet is silently deselected by every ``-m``-filtered CI job for its directory,
so its coverage stops running without anyone noticing. This pins the
convention the CI selections rely on.

The map mirrors the ``-m`` filters in .github/workflows/*.yml. Directories
whose jobs select with ``"unit or not integration"`` or with no ``-m`` at all
tolerate unmarked files and are deliberately absent here. ``ci_fast`` remains
a per-file judgment (heavy container suites may legitimately omit it), so only
the base gate marker is enforced.
"""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

TESTS_ROOT = Path(__file__).resolve().parents[2]

GATED_DIRS = {
    "runtime/integration": "integration",
    "common/integration": "integration",
    "routing/integration": "integration",
    "agents/unit": "unit",
    "core/unit": "unit",
    "routing/unit": "unit",
}


def test_gated_directories_have_their_gate_marker():
    missing: list[str] = []
    for rel, marker in GATED_DIRS.items():
        directory = TESTS_ROOT / rel
        for test_file in sorted(directory.rglob("test_*.py")):
            if f"pytest.mark.{marker}" not in test_file.read_text():
                missing.append(f"{test_file.relative_to(TESTS_ROOT)} needs {marker}")
    assert not missing, (
        "these files are invisible to their directory's CI -m selection:\n"
        + "\n".join(missing)
    )
