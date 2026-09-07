"""A test that never runs reports as absence, which reads exactly like success.

Two independent ways CI reports green for a test it never executed:

1. **No selection reaches it.** CI never runs a bare directory — every job
   names paths AND a ``-m`` expression. A test whose file sits under no named
   path, or whose markers the expression deselects, silently never runs. Scope
   is derived, not asserted: ``ci_fast`` is the marker that declares a test
   belongs in CI (``pytest.ini``), and ``local_only`` is the marker that
   declares it must not. A test carrying neither is out of scope by its own
   markers, so an unmarked file is not a finding here.

2. **The workflow never fires.** A guard that reads a tree, or a single file,
   outside its own package only runs when the commit touches paths in its
   workflow's filter. A repo-wide scan inside a narrowly-filtered workflow
   misses every violation introduced outside that filter, and so does a filter
   that omits the one script a guard parses.

Both detectors are driven by synthetic input below, because a repo-wide "no
offenders remain" assertion cannot protect its own detector: once the last
offender is fixed, gutting the detector leaves it green.
"""

from __future__ import annotations

import ast
import functools
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

# pytest's own marker-expression evaluator: reimplementing the algebra gets
# ``not`` precedence and bare identifiers subtly wrong.
# ``test_marker_evaluation_agrees_with_a_real_pytest_run`` pins it against a
# real ``-m`` collection.
from _pytest.mark import MarkMatcher
from _pytest.mark.expression import Expression
from _pytest.mark.structures import Mark

from tests.fixtures.ci_workflows import (
    Selection,
    Workflow,
    load_workflows,
    workflows_running,
)
from tests.fixtures.marker_dump import DUMP_PATH_ENV

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
TESTS_ROOT = REPO_ROOT / "tests"

# The marker a test carries to declare it belongs in CI, and the one that
# declares it does not. Both are defined in pytest.ini.
CI_MARKER = "ci_fast"
OPT_OUT_MARKER = "local_only"

# Trees excluded by hand, each with the reason no CI job can host them.
# ``test_excluded_trees_still_exist`` fails when one is retired, so a stale
# exclusion cannot keep quietly widening.
EXCLUDED_TREES: tuple[tuple[str, str], ...] = (
    ("tests/e2e", "every test needs the deployed k3d cluster"),
)


# --------------------------------------------------------------------------
# Detector 1 — marker-aware reachability
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _expression(marker_expr: str) -> Expression:
    return Expression.compile(marker_expr)


def selects(selection: Selection, test_path: str, markers: Sequence[str]) -> bool:
    """Whether ``selection`` both names ``test_path`` and keeps ``markers``."""
    if not selection.names(test_path):
        return False
    if selection.marker_expr is None:
        return True
    matcher = MarkMatcher.from_markers(
        Mark(name=name, args=(), kwargs={}) for name in markers
    )
    return _expression(selection.marker_expr).evaluate(matcher)


def unreachable_ci_tests(
    marker_map: Mapping[str, Sequence[str]],
    selections: Sequence[Selection],
    excluded_trees: Sequence[str] = (),
) -> list[str]:
    """Node ids that declare themselves CI-runnable and that no job selects."""
    unreachable = []
    for nodeid, markers in sorted(marker_map.items()):
        if CI_MARKER not in markers or OPT_OUT_MARKER in markers:
            continue
        test_path = nodeid.split("::", 1)[0]
        if any(
            test_path == tree or test_path.startswith(tree + "/")
            for tree in excluded_trees
        ):
            continue
        if not any(selects(s, test_path, markers) for s in selections):
            unreachable.append(nodeid)
    return unreachable


# --------------------------------------------------------------------------
# Detector 2 — trees a test reads, and whether its workflow fires on them
# --------------------------------------------------------------------------

_SCAN_METHODS = frozenset({"rglob", "glob", "iterdir"})

# Calls that read one whole file rather than walking a tree.
_READ_METHODS = frozenset({"read_text", "read_bytes", "open"})

# Calls that take a path without reading the tree under it.
_PATH_PASSTHROUGH = frozenset({"Path", "len", "print", "repr", "str"})


def _ancestor(relative_path: str, levels: int) -> str:
    parts = relative_path.split("/")
    return "/".join(parts[: max(len(parts) - levels, 0)])


def _join(base: str, extra: str) -> str:
    return f"{base}/{extra}".strip("/") if base else extra.strip("/")


def _under(path: str, root: str) -> bool:
    return root == "" or path == root or path.startswith(root + "/")


def _resolve(
    node: ast.AST, env: Mapping[str, tuple[str, ...]], module_rel: str
) -> tuple[str, ...]:
    """Repo-relative directories an expression can denote.

    A tuple because one name reaches several: ``for root in (a, b)`` binds
    ``root`` to both, and a guard that walks either reads both trees.
    """
    if isinstance(node, ast.Name):
        return env.get(node.id, ())
    if isinstance(node, ast.List | ast.Tuple):
        return tuple(
            path for element in node.elts for path in _resolve(element, env, module_rel)
        )
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        return tuple(_ancestor(b, 1) for b in _resolve(node.value, env, module_rel))
    if isinstance(node, ast.Attribute) and node.attr in {"resolve", "absolute"}:
        return _resolve(node.value, env, module_rel)
    if isinstance(node, ast.Subscript):
        target = node.value
        if (
            isinstance(target, ast.Attribute)
            and target.attr == "parents"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
        ):
            levels = node.slice.value + 1
            return tuple(
                _ancestor(b, levels) for b in _resolve(target.value, env, module_rel)
            )
        return ()
    if isinstance(node, ast.Call):
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr in {
            "resolve",
            "absolute",
        }:
            return _resolve(function.value, env, module_rel)
        name = getattr(function, "id", None) or getattr(function, "attr", None)
        if name == "Path" and node.args:
            argument = node.args[0]
            if isinstance(argument, ast.Name) and argument.id == "__file__":
                return (module_rel,)
        return ()
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        base = _resolve(node.left, env, module_rel)
        if not isinstance(node.right, ast.Constant) or not isinstance(
            node.right.value, str
        ):
            # A runtime-computed segment can land anywhere under the base, so
            # the base itself is the tree the module reads.
            return base
        return tuple(_join(b, node.right.value) for b in base)
    return ()


def _bind_paths(tree: ast.AST, module_rel: str) -> dict[str, tuple[str, ...]]:
    """Names bound to repo-rooted directories, including loop variables."""
    env: dict[str, tuple[str, ...]] = {}

    def record(name: str, paths: tuple[str, ...]) -> None:
        if paths:
            env[name] = tuple(sorted(set(env.get(name, ())) | set(paths)))

    for _ in range(3):  # fixpoint: a name may be bound after its first use
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    record(target.id, _resolve(node.value, env, module_rel))
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                record(node.target.id, _resolve(node.iter, env, module_rel))
    return env


def scanned_trees(
    module_rel: str, source: str, repo_root: Path = REPO_ROOT
) -> tuple[str, ...]:
    """Repo-relative directories a test module reads file-by-file.

    Either directly (``root.rglob(...)``, ``os.walk(root)``) or by handing the
    directory to a helper that walks it: a guard reading the repo through
    ``missing_schema_references(root)`` depends on that tree just as much.

    A tree already covered by a wider scanned tree is dropped: reading
    ``tests/e2e`` adds nothing once the module reads all of ``tests``.
    """
    tree = ast.parse(source)
    env = _bind_paths(tree, module_rel)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr in _SCAN_METHODS:
            roots.update(_resolve(function.value, env, module_rel))
        elif (
            isinstance(function, ast.Attribute)
            and function.attr == "walk"
            and node.args
        ):
            roots.update(_resolve(node.args[0], env, module_rel))
        elif isinstance(function, ast.Name) and function.id not in _PATH_PASSTHROUGH:
            arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
            roots.update(
                candidate
                for argument in arguments
                for candidate in _resolve(argument, env, module_rel)
                if (repo_root / candidate).is_dir()
            )
    if "" in roots:
        return ("",)  # the repo root subsumes every other tree
    return tuple(
        sorted(
            root
            for root in roots
            if not any(
                other != root and root.startswith(other + "/") for other in roots
            )
        )
    )


def foreign_trees(
    module_rel: str, source: str, repo_root: Path = REPO_ROOT
) -> tuple[str, ...]:
    """Scanned trees outside the module's own test package."""
    own_package = "/".join(module_rel.split("/")[:2])
    return tuple(
        root
        for root in scanned_trees(module_rel, source, repo_root)
        if not (root == own_package or root.startswith(own_package + "/"))
    )


def read_files(
    module_rel: str, source: str, repo_root: Path = REPO_ROOT
) -> tuple[str, ...]:
    """Repo-relative single files a test module reads whole.

    Either directly (``(root / "run.sh").read_text()``) or by handing the file
    to a helper that opens it. Resolution is the same repo-rooted ``/``-join
    the tree scan uses, so both ``root / "scripts" / "run.sh"`` and
    ``root / "scripts/run.sh"`` land on the same path; a candidate counts when
    it names a file in the tree.
    """
    tree = ast.parse(source)
    env = _bind_paths(tree, module_rel)
    files: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr in _READ_METHODS:
            candidates = _resolve(function.value, env, module_rel)
        elif isinstance(function, ast.Name) and function.id not in _PATH_PASSTHROUGH:
            arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
            candidates = tuple(
                candidate
                for argument in arguments
                for candidate in _resolve(argument, env, module_rel)
            )
        else:
            continue
        files.update(c for c in candidates if (repo_root / c).is_file())
    return tuple(sorted(files))


def foreign_files(
    module_rel: str, source: str, repo_root: Path = REPO_ROOT
) -> tuple[str, ...]:
    """Read files outside the module's own package and outside every tree it scans.

    A file under a scanned tree is the tree finding's business, and a filter
    covering that tree already covers the file.
    """
    own_package = "/".join(module_rel.split("/")[:2])
    trees = scanned_trees(module_rel, source, repo_root)
    return tuple(
        path
        for path in read_files(module_rel, source, repo_root)
        if not _under(path, own_package)
        and not any(_under(path, root) for root in trees)
    )


def unwatched_scans(
    workflows: Sequence[Workflow], modules: Mapping[str, str]
) -> list[str]:
    """``module: root`` pairs whose scan no workflow running the module watches.

    A module no gating workflow runs at all is reachability's business, not
    this detector's.
    """
    findings = []
    for module_rel, source in sorted(modules.items()):
        running = workflows_running(workflows, module_rel)
        if not running:
            continue
        for root in foreign_trees(module_rel, source):
            if not any(workflow.watches(root) for workflow in running):
                findings.append(
                    f"{module_rel}: reads {root or '<repo root>'}/ but "
                    f"{[w.name for w in running]} does not fire on changes there"
                )
    return findings


def unwatched_file_reads(
    workflows: Sequence[Workflow], modules: Mapping[str, str]
) -> list[str]:
    """``module: file`` pairs whose read no workflow running the module watches."""
    findings = []
    for module_rel, source in sorted(modules.items()):
        running = workflows_running(workflows, module_rel)
        if not running:
            continue
        for path in foreign_files(module_rel, source):
            if not any(workflow.watches_file(path) for workflow in running):
                findings.append(
                    f"{module_rel}: reads {path} but "
                    f"{[w.name for w in running]} does not fire on changes there"
                )
    return findings


def unwatched_selections(workflows: Sequence[Workflow]) -> list[str]:
    """Selections whose own test paths their workflow's filter does not cover.

    Editing the test itself must run it; a filter that omits the directory it
    selects is the same blind spot one step earlier.
    """
    return list(
        dict.fromkeys(
            f"{selection.workflow} runs {path} but does not fire on changes there"
            for workflow in workflows
            if workflow.commit_gating
            for selection in workflow.selections
            for path in selection.paths
            if not workflow.watches(path)
        )
    )


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def workflows() -> tuple[Workflow, ...]:
    return load_workflows(WORKFLOWS_DIR)


def collect_marker_map(paths: Sequence[str], cwd: Path = REPO_ROOT) -> dict[str, list]:
    """``{node id: marker names}`` from a real pytest collection of ``paths``."""
    with tempfile.TemporaryDirectory() as scratch:
        dump = Path(scratch) / "markers.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *paths,
                "--collect-only",
                "-qq",
                "-p",
                "tests.fixtures.marker_dump",
                "-p",
                "no:cacheprovider",
                "--continue-on-collection-errors",
            ],
            cwd=cwd,
            env={**os.environ, DUMP_PATH_ENV: str(dump), "JAX_PLATFORM_NAME": "cpu"},
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.returncode != 0 or not dump.exists():
            raise AssertionError(
                f"collecting {list(paths)} failed (rc={result.returncode}); a test "
                "module that cannot be imported has no CI coverage either\n"
                f"{result.stdout[-4000:]}\n{result.stderr[-2000:]}"
            )
        return json.loads(dump.read_text())


@pytest.fixture(scope="module")
def repo_marker_map() -> dict[str, list]:
    return collect_marker_map(["tests"])


@pytest.fixture(scope="module")
def guard_modules() -> dict[str, str]:
    return {
        path.relative_to(REPO_ROOT).as_posix(): path.read_text(errors="replace")
        for path in sorted(TESTS_ROOT.rglob("test_*.py"))
    }


# --------------------------------------------------------------------------
# The workflow parser
# --------------------------------------------------------------------------


def _workflow(workflows: Sequence[Workflow], name: str) -> Workflow:
    return next(workflow for workflow in workflows if workflow.name == name)


def test_parser_joins_backslash_continuations(workflows) -> None:
    """runtime-tests.yml names four directories across five physical lines."""
    runtime = _workflow(workflows, "runtime-tests.yml")
    unit_jobs = [s for s in runtime.selections if s.job == "unit-tests"]
    assert [(s.paths, s.marker_expr) for s in unit_jobs] == [
        (
            (
                "tests/runtime/unit",
                "tests/admin/unit",
                "tests/foundation/unit",
                "tests/events/unit",
            ),
            None,
        )
    ]


def test_parser_ignores_echoed_command_hints(workflows) -> None:
    """agents-tests.yml echoes two pytest command lines that CI never runs."""
    agents = _workflow(workflows, "agents-tests.yml")
    assert [(s.job, s.paths, s.marker_expr) for s in agents.selections] == [
        ("unit-tests", ("tests/agents/unit",), "unit"),
        ("integration-tests", ("tests/agents/integration",), "ci_fast"),
    ]


def test_tag_only_workflow_gates_no_commit(workflows) -> None:
    """publish-packages.yml runs on ``v*`` tags, long after the regression."""
    publish = _workflow(workflows, "publish-packages.yml")
    assert (publish.commit_gating, [s.paths for s in publish.selections]) == (
        False,
        [("tests/common", "tests/routing/unit")],
    )


def test_every_selected_path_exists(workflows) -> None:
    assert [
        f"{s.workflow}:{s.job}:{path}"
        for w in workflows
        for s in w.selections
        for path in s.paths
        if not (REPO_ROOT / path).exists()
    ] == []


# --------------------------------------------------------------------------
# Detector 1 — self-tests on synthetic input
# --------------------------------------------------------------------------

_UNFILTERED = Selection("w.yml", "job", ("tests/covered",), None)
_CI_FAST_INTEGRATION = Selection(
    "w.yml", "job", ("tests/covered",), "integration and ci_fast and not requires_lm"
)


def test_detector_reports_a_ci_fast_test_no_selection_names() -> None:
    marker_map = {
        "tests/covered/test_a.py::test_a": ["ci_fast", "unit"],
        "tests/orphan/test_b.py::test_b": ["ci_fast", "unit"],
    }
    assert unreachable_ci_tests(marker_map, [_UNFILTERED]) == [
        "tests/orphan/test_b.py::test_b"
    ]


def test_detector_reports_a_test_its_selection_deselects() -> None:
    """Named by the path, dropped by the expression — the harder half."""
    marker_map = {
        "tests/covered/test_a.py::keeps": ["ci_fast", "integration"],
        "tests/covered/test_a.py::dropped": ["ci_fast", "integration", "requires_lm"],
    }
    assert unreachable_ci_tests(marker_map, [_CI_FAST_INTEGRATION]) == [
        "tests/covered/test_a.py::dropped"
    ]


def test_detector_respects_markers_and_ignores_an_unmarked_test() -> None:
    """An unmarked test declares nothing, so no CI job owes it a run.

    Adding its directory to a workflow would select zero tests while reading
    like a fix; the scope here is derived from the markers, not from the
    directory.
    """
    marker_map = {
        "tests/orphan/test_unmarked.py::test_x": [],
        "tests/orphan/test_slow.py::test_y": ["integration", "requires_gpu"],
        "tests/orphan/test_declared.py::test_z": ["ci_fast", "unit"],
    }
    assert unreachable_ci_tests(marker_map, [_UNFILTERED]) == [
        "tests/orphan/test_declared.py::test_z"
    ]


def test_detector_honours_the_opt_out_marker() -> None:
    marker_map = {
        "tests/orphan/test_a.py::opted_out": ["ci_fast", "unit", "local_only"],
        "tests/orphan/test_a.py::declared": ["ci_fast", "unit"],
    }
    assert unreachable_ci_tests(marker_map, [_UNFILTERED]) == [
        "tests/orphan/test_a.py::declared"
    ]


def test_excluded_tree_suppresses_only_tests_inside_it() -> None:
    marker_map = {
        "tests/e2e/test_cluster.py::test_x": ["ci_fast", "e2e"],
        "tests/e2etools/test_other.py::test_y": ["ci_fast", "unit"],
    }
    excluded = [tree for tree, _ in EXCLUDED_TREES]
    assert unreachable_ci_tests(marker_map, [_UNFILTERED], excluded) == [
        "tests/e2etools/test_other.py::test_y"
    ]
    assert unreachable_ci_tests(marker_map, [_UNFILTERED]) == [
        "tests/e2e/test_cluster.py::test_x",
        "tests/e2etools/test_other.py::test_y",
    ]


def test_excluded_trees_still_exist() -> None:
    """A retired exclusion is how a guard goes quietly blind."""
    assert [tree for tree, _ in EXCLUDED_TREES if not (REPO_ROOT / tree).is_dir()] == []


def test_marker_evaluation_agrees_with_a_real_pytest_run() -> None:
    """The in-process expression evaluator must match what ``-m`` collects."""
    paths = ["tests/utils", "tests/core/integration"]
    marker_expr = "integration and ci_fast and not requires_lm"
    marker_map = collect_marker_map(paths)
    computed = sorted(
        nodeid
        for nodeid, markers in marker_map.items()
        if selects(Selection("w", "j", ("tests",), marker_expr), "tests/x", markers)
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *paths,
            "-m",
            marker_expr,
            "--collect-only",
            "-qq",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    from_pytest = sorted(
        line for line in result.stdout.splitlines() if "::" in line and " " not in line
    )
    assert computed == from_pytest


# --------------------------------------------------------------------------
# Detector 2 — self-tests on synthetic input
# --------------------------------------------------------------------------

_NARROW_WORKFLOW = Workflow(
    "narrow.yml",
    commit_gating=True,
    trigger_filters=(("libs/vespa/**", "tests/backends/**"),),
    selections=(Selection("narrow.yml", "unit", ("tests/backends/unit",), None),),
)
_BROAD_WORKFLOW = Workflow(
    "broad.yml",
    commit_gating=True,
    trigger_filters=(("libs/**", "scripts/**", "tests/backends/**"),),
    selections=(Selection("broad.yml", "guards", ("tests/backends/unit",), None),),
)
_REPO_WIDE_SOURCE = (
    "from pathlib import Path\n"
    "_REPO_ROOT = Path(__file__).resolve().parents[3]\n"
    "def scan():\n"
    '    for root in (_REPO_ROOT / "libs", _REPO_ROOT / "scripts"):\n'
    '        for py in root.rglob("*.py"):\n'
    "            yield py\n"
)
_LOCAL_SOURCE = (
    "from pathlib import Path\n"
    "_HERE = Path(__file__).resolve().parent\n"
    "def scan():\n"
    '    return sorted(_HERE.glob("test_*.py"))\n'
)
_OFFENDER = "tests/backends/unit/test_synthetic_guard.py"


def test_scan_detector_reads_the_trees_a_module_walks() -> None:
    assert scanned_trees(_OFFENDER, _REPO_WIDE_SOURCE) == ("libs", "scripts")


def test_scan_detector_ignores_a_module_that_reads_only_its_own_package() -> None:
    assert foreign_trees(_OFFENDER, _LOCAL_SOURCE) == ()


def test_scan_detector_reports_a_narrow_filter() -> None:
    findings = unwatched_scans([_NARROW_WORKFLOW], {_OFFENDER: _REPO_WIDE_SOURCE})
    assert findings == [
        f"{_OFFENDER}: reads libs/ but ['narrow.yml'] does not fire on changes there",
        f"{_OFFENDER}: reads scripts/ but ['narrow.yml'] does not fire on "
        "changes there",
    ]


def test_scan_detector_accepts_a_filter_that_covers_every_scanned_tree() -> None:
    assert unwatched_scans([_BROAD_WORKFLOW], {_OFFENDER: _REPO_WIDE_SOURCE}) == []


def test_scan_detector_leaves_an_unrun_module_to_reachability() -> None:
    """No workflow runs it at all, so its scan is not what is broken."""
    orphan = "tests/orphan/test_guard.py"
    assert unwatched_scans([_BROAD_WORKFLOW], {orphan: _REPO_WIDE_SOURCE}) == []


def test_selection_detector_reports_a_workflow_blind_to_its_own_tests() -> None:
    blind = Workflow(
        "blind.yml",
        commit_gating=True,
        trigger_filters=(("libs/core/**",),),
        selections=(Selection("blind.yml", "unit", ("tests/core/unit",), "unit"),),
    )
    assert unwatched_selections([blind]) == [
        "blind.yml runs tests/core/unit but does not fire on changes there"
    ]


def test_selection_detector_accepts_a_workflow_that_watches_its_own_tests() -> None:
    seeing = Workflow(
        "seeing.yml",
        commit_gating=True,
        trigger_filters=(("libs/core/**", "tests/core/**"),),
        selections=(Selection("seeing.yml", "unit", ("tests/core/unit",), "unit"),),
    )
    assert unwatched_selections([seeing]) == []


def test_selection_detector_requires_every_trigger_to_watch() -> None:
    """A push filter that covers the tests while the PR filter does not means
    the job is blind on exactly the commits a reviewer sees."""
    half_blind = Workflow(
        "half.yml",
        commit_gating=True,
        trigger_filters=(("tests/core/**",), ("libs/core/**",)),
        selections=(Selection("half.yml", "unit", ("tests/core/unit",), "unit"),),
    )
    assert unwatched_selections([half_blind]) == [
        "half.yml runs tests/core/unit but does not fire on changes there"
    ]


def test_scan_detector_finds_the_shipped_repo_wide_guards(guard_modules) -> None:
    """Structural, not a hardcoded list: these are what the AST walk reports.

    ``test_marker_coverage`` reads ``REPO_ROOT / rel`` with ``rel`` computed
    from the workflows, so the tree it can reach is the repo.
    """
    docv1 = "tests/backends/unit/test_docv1_confined_to_vespa.py"
    spawners = "tests/common/unit/test_ci_fast_excludes_model_spawners.py"
    markers = "tests/runtime/unit/test_marker_coverage.py"
    assert (
        foreign_trees(docv1, guard_modules[docv1]),
        foreign_trees(spawners, guard_modules[spawners]),
        foreign_trees(markers, guard_modules[markers]),
    ) == (("libs", "scripts"), (".github/workflows",), ("",))


_SPLIT_JOIN_READ_SOURCE = (
    "from pathlib import Path\n"
    "def loader():\n"
    "    root = Path(__file__).resolve().parents[3]\n"
    '    return (root / "scripts" / "run_e2e_batched.sh").read_text()\n'
)
_ONE_SEGMENT_READ_SOURCE = (
    "from pathlib import Path\n"
    "def loader():\n"
    '    workflow = Path(__file__).parents[3] / ".github/workflows/release-images.yml"\n'
    "    return workflow.read_text()\n"
)
_OWN_PACKAGE_READ_SOURCE = (
    "from pathlib import Path\n"
    "def loader():\n"
    "    root = Path(__file__).resolve().parents[3]\n"
    '    return (root / "tests" / "backends" / "conftest.py").read_text()\n'
)
_SCAN_AND_READ_SOURCE = (
    "from pathlib import Path\n"
    '_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"\n'
    "def loader():\n"
    '    return [p.read_text() for p in _SCRIPTS.rglob("*.sh")] + [\n'
    '        (_SCRIPTS / "run_e2e_batched.sh").read_text()\n'
    "    ]\n"
)
_FILE_WATCHING_WORKFLOW = Workflow(
    "watching.yml",
    commit_gating=True,
    trigger_filters=(
        ("tests/backends/**", "scripts/**", ".github/workflows/release-images.yml"),
    ),
    selections=(Selection("watching.yml", "unit", ("tests/backends/unit",), None),),
)


def test_read_detector_resolves_both_repo_rooted_join_forms() -> None:
    """``/ "a" / "b.sh"`` and ``/ "a/b.sh"`` must land on the same file."""
    assert (
        foreign_files(_OFFENDER, _SPLIT_JOIN_READ_SOURCE),
        foreign_files(_OFFENDER, _ONE_SEGMENT_READ_SOURCE),
    ) == (
        ("scripts/run_e2e_batched.sh",),
        (".github/workflows/release-images.yml",),
    )


def test_read_detector_reports_a_file_outside_the_filter() -> None:
    findings = unwatched_file_reads(
        [_NARROW_WORKFLOW],
        {
            _OFFENDER: _SPLIT_JOIN_READ_SOURCE,
            "tests/backends/unit/test_synthetic_images.py": _ONE_SEGMENT_READ_SOURCE,
        },
    )
    assert findings == [
        f"{_OFFENDER}: reads scripts/run_e2e_batched.sh but ['narrow.yml'] "
        "does not fire on changes there",
        "tests/backends/unit/test_synthetic_images.py: reads "
        ".github/workflows/release-images.yml but ['narrow.yml'] does not fire "
        "on changes there",
    ]


def test_read_detector_accepts_a_filter_that_covers_each_read_file() -> None:
    """``scripts/**`` covers the script; the literal filename covers the workflow."""
    assert (
        unwatched_file_reads(
            [_FILE_WATCHING_WORKFLOW], {_OFFENDER: _SPLIT_JOIN_READ_SOURCE}
        ),
        unwatched_file_reads(
            [_FILE_WATCHING_WORKFLOW], {_OFFENDER: _ONE_SEGMENT_READ_SOURCE}
        ),
    ) == ([], [])


def test_read_detector_leaves_a_file_under_a_scanned_tree_to_the_tree_finding() -> None:
    """One finding per blind spot: the tree already names the whole subtree."""
    assert (
        foreign_files(_OFFENDER, _SCAN_AND_READ_SOURCE),
        unwatched_file_reads([_NARROW_WORKFLOW], {_OFFENDER: _SCAN_AND_READ_SOURCE}),
        unwatched_scans([_NARROW_WORKFLOW], {_OFFENDER: _SCAN_AND_READ_SOURCE}),
    ) == (
        (),
        [],
        [
            f"{_OFFENDER}: reads scripts/ but ['narrow.yml'] does not fire on "
            "changes there"
        ],
    )


def test_read_detector_ignores_a_file_in_the_modules_own_package() -> None:
    assert (
        read_files(_OFFENDER, _OWN_PACKAGE_READ_SOURCE),
        foreign_files(_OFFENDER, _OWN_PACKAGE_READ_SOURCE),
    ) == (("tests/backends/conftest.py",), ())


def test_file_filters_match_with_github_glob_semantics() -> None:
    """``*`` stops at ``/`` while ``**`` crosses it, as GitHub matches ``paths``."""
    workflow = Workflow(
        "globs.yml",
        commit_gating=True,
        trigger_filters=((".github/workflows/release-images.yml", "scripts/*.sh"),),
        selections=(),
    )
    assert (
        workflow.watches_file(".github/workflows/release-images.yml"),
        workflow.watches_file("scripts/run_e2e_batched.sh"),
        workflow.watches_file("scripts/nested/run.sh"),
        workflow.watches_file(".github/workflows/cli-tests.yml"),
    ) == (True, True, False, False)


# --------------------------------------------------------------------------
# The repo-wide assertions
# --------------------------------------------------------------------------


def test_every_ci_declared_test_is_reachable(workflows, repo_marker_map) -> None:
    selections = [s for w in workflows if w.commit_gating for s in w.selections]
    unreachable = unreachable_ci_tests(
        repo_marker_map, selections, [tree for tree, _ in EXCLUDED_TREES]
    )
    assert unreachable == [], (
        f"{len(unreachable)} tests mark themselves {CI_MARKER} yet no CI "
        "selection both names their path and keeps their markers:\n"
        + "\n".join(unreachable)
    )


def test_every_repo_wide_guard_runs_on_the_tree_it_reads(
    workflows, guard_modules
) -> None:
    findings = unwatched_scans(workflows, guard_modules)
    assert findings == [], (
        "a guard is skipped on exactly the commits that break it when its "
        "workflow does not fire on the tree it reads:\n" + "\n".join(findings)
    )


def test_every_guard_runs_on_the_files_it_reads(workflows, guard_modules) -> None:
    findings = unwatched_file_reads(workflows, guard_modules)
    assert findings == [], (
        "a guard is skipped on exactly the commits that break it when its "
        "workflow does not fire on the file it reads:\n" + "\n".join(findings)
    )


def test_every_selection_fires_when_its_own_tests_change(workflows) -> None:
    findings = unwatched_selections(workflows)
    assert findings == [], (
        "editing these tests runs nothing — the workflow selecting them does "
        "not fire on their directory:\n" + "\n".join(findings)
    )
