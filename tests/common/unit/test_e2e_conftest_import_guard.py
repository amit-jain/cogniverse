"""No test module outside tests/e2e imports the e2e conftest.

Importing ``tests/e2e/conftest.py`` runs its module body, which publishes the
e2e cluster's Vespa endpoint as ``TEST_BACKEND_URL``/``TEST_BACKEND_PORT``.
The root ``backend_config_env`` fixture reads those for the whole session, so
a unit or integration session that imports the conftest -- directly, by path,
or through a ``tests/e2e`` module that imports it -- resolves backend config
against the live cluster. Helpers other suites need live in plain modules
under ``tests/e2e``.

The import graph is read statically: every enclosing package ``__init__``,
every import statement anywhere in a module (function bodies included), every
dotted ``tests.*`` string (an ``importlib.import_module`` or ``pytest_plugins``
target) and every ``spec_from_file_location`` path built from ``__file__``.
A loader path that cannot be resolved but names ``e2e`` counts as reaching
the conftest.
"""

from __future__ import annotations

import ast
import re
from collections import deque
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_DOTTED = re.compile(r"tests(?:\.\w+)+")
_UNRESOLVED = Path("<unresolved e2e loader path>")


def _module_files(root: Path, dotted: str) -> list[Path]:
    """Files importing ``dotted`` executes: each package ``__init__``, then the module."""
    parts = dotted.split(".")
    files: list[Path] = []
    for depth in range(1, len(parts) + 1):
        base = root.joinpath(*parts[:depth])
        for candidate in (base / "__init__.py", base.with_suffix(".py")):
            if candidate.is_file():
                files.append(candidate)
    return files


def _package(root: Path, path: Path) -> list[str]:
    return list(path.relative_to(root).parent.parts)


def _assignments(body: list[ast.stmt]) -> dict[str, ast.expr]:
    found: dict[str, ast.expr] = {}
    for stmt in body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            found[stmt.targets[0].id] = stmt.value
    return found


class _LoaderPath:
    """Best-effort value of a ``spec_from_file_location`` path argument."""

    def __init__(self, path: Path, scopes: list[dict[str, ast.expr]]):
        self._file = path
        self._scopes = scopes

    def value(self, node: ast.expr, depth: int = 0) -> Path | str | None:
        if depth > 20:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            if node.id == "__file__":
                return self._file
            for scope in self._scopes:
                if node.id in scope:
                    return self.value(scope[node.id], depth + 1)
            return None
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", None)
            )
            if name in {"Path", "str"} and len(node.args) == 1:
                inner = self.value(node.args[0], depth + 1)
                if name == "Path" and isinstance(inner, str):
                    return Path(inner)
                return inner
            if name == "resolve" and isinstance(func, ast.Attribute):
                return self.value(func.value, depth + 1)
            return None
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            base = self.value(node.value, depth + 1)
            return base.parent if isinstance(base, Path) else None
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "parents"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
        ):
            base = self.value(node.value.value, depth + 1)
            if isinstance(base, Path):
                return base.parents[node.slice.value]
            return None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left = self.value(node.left, depth + 1)
            right = self.value(node.right, depth + 1)
            if isinstance(left, Path) and isinstance(right, str):
                return left / right
            return None
        return None


def _strings(
    node: ast.AST, scopes: list[dict[str, ast.expr]], depth: int = 0
) -> set[str]:
    """Every string constant in ``node``, following names to their assignments."""
    found: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            found.add(child.value)
        if isinstance(child, ast.Name) and depth < 10:
            for scope in scopes:
                if child.id in scope:
                    found |= _strings(scope[child.id], scopes, depth + 1)
                    break
    return found


def import_edges(root: Path, path: Path) -> list[Path]:
    """Files under ``root`` that importing or running ``path`` may execute."""
    tree = ast.parse(path.read_text(), filename=str(path))
    package = _package(root, path)
    # Importing a module inside a package runs every enclosing __init__ first.
    edges: list[Path] = [
        parent / "__init__.py"
        for parent in path.parents
        if parent.is_relative_to(root)
        and parent != root
        and parent / "__init__.py" != path
        and (parent / "__init__.py").is_file()
    ]
    module_scope = _assignments(tree.body)

    def visit(node: ast.AST, scopes: list[dict[str, ast.expr]]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes = [_assignments(node.body), *scopes]
        if isinstance(node, ast.Import):
            for alias in node.names:
                edges.extend(_module_files(root, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                dotted = ".".join([*base, *([node.module] if node.module else [])])
            else:
                dotted = node.module or ""
            if dotted:
                edges.extend(_module_files(root, dotted))
                for alias in node.names:
                    edges.extend(_module_files(root, f"{dotted}.{alias.name}"))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _DOTTED.fullmatch(node.value):
                edges.extend(_module_files(root, node.value))
        elif isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", None)
            )
            if name == "spec_from_file_location" and len(node.args) >= 2:
                target = _LoaderPath(path, scopes).value(node.args[1])
                if isinstance(target, Path):
                    target = target.resolve()
                    if target.is_file() and target.is_relative_to(root):
                        edges.append(target)
                elif any("e2e" in text for text in _strings(node.args[1], scopes)):
                    edges.append(_UNRESOLVED)
        for child in ast.iter_child_nodes(node):
            visit(child, scopes)

    visit(tree, [module_scope])
    return edges


def conftest_import_chains(root: Path) -> dict[str, list[str]]:
    """``{module: chain}`` for every module outside ``tests/e2e`` that reaches
    ``tests/e2e/conftest.py``; the chain runs from the module to the conftest."""
    root = root.resolve()
    tests_dir = root / "tests"
    e2e_dir = tests_dir / "e2e"
    conftest = e2e_dir / "conftest.py"
    edge_cache: dict[Path, list[Path]] = {}

    def edges(path: Path) -> list[Path]:
        if path not in edge_cache:
            edge_cache[path] = import_edges(root, path)
        return edge_cache[path]

    def label(path: Path) -> str:
        return str(path) if path == _UNRESOLVED else path.relative_to(root).as_posix()

    chains: dict[str, list[str]] = {}
    for start in sorted(tests_dir.rglob("*.py")):
        if start.is_relative_to(e2e_dir):
            continue
        previous: dict[Path, Path | None] = {start: None}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if current in (conftest, _UNRESOLVED):
                chain = []
                step: Path | None = current
                while step is not None:
                    chain.append(label(step))
                    step = previous[step]
                chains[label(start)] = chain[::-1]
                break
            for target in edges(current):
                if target not in previous:
                    previous[target] = current
                    queue.append(target)
    return chains


def test_no_module_outside_e2e_reaches_the_e2e_conftest():
    assert conftest_import_chains(_REPO) == {}


def test_the_scan_sees_e2e_modules_that_import_the_conftest():
    """The repository scan is not vacuous: a real e2e module's own edges reach
    the conftest, so a broken root or resolver would fail here first."""
    api_module = _REPO / "tests" / "e2e" / "test_api_e2e.py"

    assert _REPO / "tests" / "e2e" / "conftest.py" in import_edges(_REPO, api_module)


_CONFTEST_SRC = (
    'import os\nos.environ.setdefault("TEST_BACKEND_PORT", "33080")\nRUNTIME = "x"\n'
)
_SYNTHETIC_TREE = {
    "tests/__init__.py": "",
    "tests/e2e/__init__.py": "",
    "tests/e2e/conftest.py": _CONFTEST_SRC,
    "tests/e2e/clean_helper.py": "RUNTIME = 'http://localhost:33000'\n",
    "tests/e2e/dirty_helper.py": (
        "def context():\n    from tests.e2e.conftest import RUNTIME\n\n    return RUNTIME\n"
    ),
    "tests/e2e/relative_helper.py": "from .conftest import RUNTIME\n",
    "tests/e2e/test_feature_e2e.py": "from tests.e2e.conftest import RUNTIME\n",
    "tests/pkg/__init__.py": "from tests.e2e import conftest\n",
    "tests/pkg/mod.py": "VALUE = 1\n",
}


def _synthetic_repo(tmp_path: Path, consumer: str) -> Path:
    for relative, source in _SYNTHETIC_TREE.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    consumer_path = tmp_path / "tests" / "common" / "unit" / "test_consumer.py"
    consumer_path.parent.mkdir(parents=True, exist_ok=True)
    consumer_path.write_text(consumer)
    return tmp_path


_CONSUMER = "tests/common/unit/test_consumer.py"
_CONFTEST = "tests/e2e/conftest.py"


@pytest.mark.parametrize(
    ("consumer", "chain"),
    [
        ("import tests.e2e.conftest\n", [_CONSUMER, _CONFTEST]),
        ("import tests.e2e.conftest as e2e_conftest\n", [_CONSUMER, _CONFTEST]),
        ("from tests.e2e.conftest import RUNTIME\n", [_CONSUMER, _CONFTEST]),
        ("from tests.e2e import conftest\n", [_CONSUMER, _CONFTEST]),
        (
            "def read():\n    from tests.e2e import conftest as c\n\n    return c\n",
            [_CONSUMER, _CONFTEST],
        ),
        (
            'import importlib\n\nimportlib.import_module("tests.e2e.conftest")\n',
            [_CONSUMER, _CONFTEST],
        ),
        (
            'MODULES = {"tests.e2e.test_feature_e2e": 1}\n',
            [_CONSUMER, "tests/e2e/test_feature_e2e.py", _CONFTEST],
        ),
        (
            "from tests.e2e import test_feature_e2e\n",
            [_CONSUMER, "tests/e2e/test_feature_e2e.py", _CONFTEST],
        ),
        (
            "from tests.e2e.dirty_helper import context\n",
            [_CONSUMER, "tests/e2e/dirty_helper.py", _CONFTEST],
        ),
        (
            "from tests.e2e import relative_helper\n",
            [_CONSUMER, "tests/e2e/relative_helper.py", _CONFTEST],
        ),
        (
            "import tests.pkg.mod\n",
            [_CONSUMER, "tests/pkg/__init__.py", _CONFTEST],
        ),
        (
            "import importlib.util\nfrom pathlib import Path\n\n"
            '_PATH = Path(__file__).resolve().parents[2] / "e2e" / "conftest.py"\n'
            '_SPEC = importlib.util.spec_from_file_location("c", _PATH)\n',
            [_CONSUMER, _CONFTEST],
        ),
        (
            "import importlib.util\nimport pathlib\n\n\ndef load():\n"
            "    script = pathlib.Path(__file__).parents[3] / 'tests/e2e/conftest.py'\n"
            "    return importlib.util.spec_from_file_location('c', script)\n",
            [_CONSUMER, _CONFTEST],
        ),
        (
            "import importlib.util\n\n\ndef load(name):\n"
            "    return importlib.util.spec_from_file_location(\n"
            "        'c', f'{ROOT}/tests/e2e/{name}.py'\n    )\n",
            [_CONSUMER, str(_UNRESOLVED)],
        ),
    ],
)
def test_detector_reports_every_route_to_the_conftest(tmp_path, consumer, chain):
    root = _synthetic_repo(tmp_path, consumer)

    chains = conftest_import_chains(root)

    assert chains[_CONSUMER] == chain
    assert set(chains) == {_CONSUMER, "tests/pkg/__init__.py", "tests/pkg/mod.py"}


@pytest.mark.parametrize(
    "consumer",
    [
        "from tests.e2e.clean_helper import RUNTIME\n",
        "from tests.e2e import clean_helper\n",
        "import tests.e2e.clean_helper as helper\n",
        'CONFTEST_PATH = "tests/e2e/conftest.py"\n',
        "import importlib.util\nfrom pathlib import Path\n\n"
        '_PATH = Path(__file__).resolve().parents[2] / "e2e" / "clean_helper.py"\n'
        '_SPEC = importlib.util.spec_from_file_location("c", _PATH)\n',
    ],
)
def test_detector_passes_modules_that_never_reach_the_conftest(tmp_path, consumer):
    root = _synthetic_repo(tmp_path, consumer)

    chains = conftest_import_chains(root)

    assert set(chains) == {"tests/pkg/__init__.py", "tests/pkg/mod.py"}
