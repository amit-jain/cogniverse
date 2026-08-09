"""Every test file must be visible to its directory's CI ``-m`` selection.

pytest's ``--strict-markers`` rejects UNDEFINED markers but cannot detect a
MISSING one: a file whose markers don't satisfy the ``-m`` expression its CI
job uses collects cleanly yet is silently deselected, so its coverage stops
running without anyone noticing.

The selections are parsed from ``.github/workflows/*.yml`` directly — a
hardcoded directory map goes stale the moment a workflow adds or changes a
``-m`` expression, which is exactly how whole directories fell out of CI.
For each ``pytest <dir> -m "<expr>"`` job line, every ``test_*.py`` under
``<dir>`` must carry markers satisfying at least one of the directory's
BASE expressions, or declare its exclusion explicitly with ``local_only``.

The base expression is the CI expression with ``ci_fast`` treated as
satisfied: ``ci_fast`` is the documented per-file speed judgment (CI runs a
fast subset of heavy integration suites; the full suite runs locally), so the
guard enforces the gate markers the subset selection relies on (``unit`` /
``integration``) without forcing every heavy file into the fast subset.

File-level granularity: a file's marker set is the union of every
``pytest.mark.<name>`` it mentions, so a file mixing gated and ungated tests
can pass this check while individual tests stay deselected — the guard pins
the per-file convention, not per-test selection.
"""

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

REPO_ROOT = Path(__file__).resolve().parents[3]
TESTS_ROOT = REPO_ROOT / "tests"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

_MARKER_RE = re.compile(r"pytest\.mark\.([A-Za-z_][A-Za-z0-9_]*)")
_EXPR_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ALLOWED_EXPR_WORDS = {"and", "or", "not"}

# Marker that declares a file deliberately excluded from CI selections.
LOCAL_ONLY_MARKER = "local_only"


def _joined_run_lines(workflow_text: str) -> list[str]:
    """Physical command lines with backslash continuations joined."""
    joined: list[str] = []
    pending = ""
    for raw in workflow_text.splitlines():
        line = pending + raw.strip()
        if line.endswith("\\"):
            pending = line[:-1] + " "
            continue
        pending = ""
        joined.append(line)
    if pending:
        joined.append(pending)
    return joined


def parse_ci_selections() -> list[tuple[str, str | None]]:
    """(test dir, -m expression or None) for every pytest job line.

    Skips ``echo``-ed hint lines — only real invocations gate CI.
    """
    selections: list[tuple[str, str | None]] = []
    for workflow in sorted(WORKFLOWS_DIR.glob("*.yml")):
        for line in _joined_run_lines(workflow.read_text()):
            if "python -m pytest" not in line:
                continue
            prefix = line[: line.index("python -m pytest")]
            if "echo" in prefix:
                continue
            # Arguments AFTER the pytest invocation — the leading
            # ``python -m pytest`` must be stripped first or its own ``-m``
            # is mistaken for the marker expression.
            args = line[line.index("python -m pytest") + len("python -m pytest") :]
            paths = [t for t in args.split() if t.startswith("tests/")]
            expr_match = re.search(r"-m\s+(?:\"([^\"]+)\"|'([^']+)'|(\S+))", args)
            expr: str | None = None
            if expr_match:
                expr = expr_match.group(1) or expr_match.group(2) or expr_match.group(3)
            for path in paths:
                selections.append((path.rstrip("/"), expr))
    return selections


def file_markers(test_file: Path) -> set[str]:
    """Explicit ``pytest.mark.*`` mentions plus the location-derived marker.

    Mirrors the ``pytest_collection_modifyitems`` hook in tests/conftest.py:
    tests under a ``unit/`` (``integration/``, ``e2e/``) directory carry that
    marker from their location unless the file declares ``local_only``.
    """
    markers = set(_MARKER_RE.findall(test_file.read_text()))
    if LOCAL_ONLY_MARKER not in markers:
        path = test_file.as_posix()
        if "/unit/" in path:
            markers.add("unit")
        elif "/integration/" in path:
            markers.add("integration")
        elif "/e2e/" in path:
            markers.add("e2e")
    return markers


def expr_matches(expr: str, markers: set[str]) -> bool:
    """Evaluate a pytest ``-m`` expression against a file's marker set.

    ``ci_fast`` evaluates as satisfied — see the module docstring.
    """

    def replace(match: re.Match) -> str:
        word = match.group(0)
        if word in _ALLOWED_EXPR_WORDS:
            return word
        if word == "ci_fast":
            return "True"
        return "True" if word in markers else "False"

    return bool(eval(_EXPR_TOKEN_RE.sub(replace, expr)))  # noqa: S307


def test_every_test_file_is_visible_to_its_ci_selection():
    selections = parse_ci_selections()
    assert selections, "no pytest selections parsed from .github/workflows"

    invisible: list[str] = []
    for rel in sorted({path for path, _ in selections}):
        if not (REPO_ROOT / rel).is_dir():
            invisible.append(f"{rel}: selected in CI but does not exist")

    # A file is visible if ANY selection covering it (its own dir OR a parent
    # dir — e.g. ``tests/finetuning/`` covers ``tests/finetuning/integration``)
    # picks it up; an unfiltered selection picks up everything under its dir.
    checked: set[Path] = set()
    for rel, _ in selections:
        directory = REPO_ROOT / rel
        if not directory.is_dir():
            continue
        for test_file in sorted(directory.rglob("test_*.py")):
            if test_file in checked:
                continue
            checked.add(test_file)
            file_rel = test_file.relative_to(REPO_ROOT).as_posix()
            exprs = [
                expr for path, expr in selections if file_rel.startswith(path + "/")
            ]
            if any(expr is None for expr in exprs):
                continue
            markers = file_markers(test_file)
            if LOCAL_ONLY_MARKER in markers:
                continue  # declared exclusion, visible in the file
            if not any(expr_matches(expr, markers) for expr in exprs):
                invisible.append(
                    f"{file_rel}: markers {sorted(markers)} satisfy none of "
                    f"{exprs} — its tests never run in CI (mark it to match, "
                    f"or declare {LOCAL_ONLY_MARKER})"
                )
    assert not invisible, (
        "these files are invisible to their directory's CI -m selection:\n"
        + "\n".join(invisible)
    )


_NON_E2E_SWEEP = (
    "not e2e and not e2e_heavy and not benchmark and not "
    "requires_teacher_model and not requires_optimizer_data and not local_only"
)


def test_e2e_files_are_deselected_by_the_non_e2e_sweep():
    """Everything under tests/e2e needs the deployed cluster, so no file there
    may satisfy the sweep that excludes e2e — a file claiming ``integration``
    instead of ``e2e`` gets collected by that sweep and errors on the cluster
    it never provisioned."""
    selected = [
        test_file.relative_to(REPO_ROOT).as_posix()
        for test_file in sorted((TESTS_ROOT / "e2e").rglob("test_*.py"))
        if expr_matches(_NON_E2E_SWEEP, file_markers(test_file))
    ]
    assert selected == [], (
        "these tests/e2e files are collected by the non-e2e sweep:\n"
        + "\n".join(selected)
    )


def test_e2e_location_derives_the_e2e_marker():
    """The directory is the taxonomy: an item under tests/e2e that declares no
    e2e marker gets one at collection."""
    from tests.fixtures.markers import apply_location_markers

    class _Item:
        def __init__(self, path):
            self.path = path
            self.own_markers = []

        def get_closest_marker(self, name):
            return next((m for m in self.own_markers if m.name == name), None)

        def add_marker(self, marker):
            self.own_markers.append(marker.mark)

    item = _Item(TESTS_ROOT / "e2e" / "test_unmarked_e2e.py")
    apply_location_markers([item])

    assert [m.name for m in item.own_markers] == ["e2e"]
