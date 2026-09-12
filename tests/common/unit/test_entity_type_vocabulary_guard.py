"""Fixture entity types stay inside the vocabulary the approval gate enforces.

``validate_approved_training_values`` refuses any entity type outside
``ENTITY_TYPES``, and ``EntityMention`` binds the same set as a ``Literal``.
A fixture carrying ``ORG`` or ``PRODUCT`` models a row production cannot emit
and the gate cannot accept: it stays green only while its own path stops short
of the gate, and turns red the moment that path is wired through.

The scan reads ``tests/`` and reports every entity-type literal outside the
shipped set. ``_DELIBERATE_NEGATIVES`` carries the fixtures whose purpose is to
be refused. Synthetic offenders drive each detector directly, because a
repo-wide "no offenders remain" assertion cannot protect its own detector once
the tree is clean.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from cogniverse_foundation.common.entity_types import ENTITY_TYPES

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

REPO_ROOT = Path(__file__).resolve().parents[3]
TESTS_ROOT = REPO_ROOT / "tests"

SCANNED_SUFFIXES = (".py", ".json", ".jsonl")

# The owner name that makes a list of records entity records: a dict key, a
# keyword argument, an assignment target, a parameter, or a callee.
_ENTITY_OWNER = re.compile(r"entit", re.IGNORECASE)
# An entity type as production writes it, so a record using the token shape is
# reported wherever it sits and an LM content block is not.
_TYPE_TOKEN = re.compile(r"^[A-Z][A-Z0-9_]*$")

_ENTITY_TYPE_KEYS = ("entity_type", "entity_types")
_DETECTOR_RANK = (
    "entity constructor",
    "entity-type key",
    "entity-owned list",
    "entity record",
)


class Finding(tuple):
    """``(path, line, enclosing scope, entity type, detector)``."""

    __slots__ = ()

    def __new__(cls, path: str, line: int, enclosing: str, value: str, detector: str):
        return super().__new__(cls, (path, line, enclosing, value, detector))

    @property
    def site(self) -> tuple[str, str, str]:
        return (self[0], self[2], self[3])

    def __str__(self) -> str:
        path, line, enclosing, value, detector = self
        return f"{path}:{line} {value!r} in {enclosing} [{detector}]"


def _dedup(findings: list[Finding]) -> list[Finding]:
    """One finding per site; the most specific detector names it."""
    best: dict[tuple[str, int, str, str], Finding] = {}
    for finding in findings:
        key = (finding[0], finding[1], finding[2], finding[3])
        current = best.get(key)
        if current is None or _DETECTOR_RANK.index(finding[4]) < _DETECTOR_RANK.index(
            current[4]
        ):
            best[key] = finding
    return sorted(best.values())


def _scopes(tree: ast.AST) -> list[tuple[int, int, str]]:
    """Line spans of every function and class, decorators included."""
    spans: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            for decorator in node.decorator_list:
                start = min(start, decorator.lineno)
            spans.append((start, node.end_lineno or start, node.name))
    return spans


def _enclosing(spans: list[tuple[int, int, str]], line: int) -> str:
    innermost = "<module>"
    width: int | None = None
    for start, end, name in spans:
        if start <= line <= end and (width is None or end - start < width):
            width, innermost = end - start, name
    return innermost


def _callee_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return getattr(func, "id", "")


def _parameter_tables(
    tree: ast.AST,
) -> tuple[list[tuple[int, int, dict[str, list[str]]]], dict[str, list[str]]]:
    """Per-class and module-wide parameter names, collected in one walk."""
    classes: list[tuple[int, int, dict[str, list[str]]]] = []
    module: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            module.setdefault(node.name, [arg.arg for arg in node.args.args])
        elif isinstance(node, ast.ClassDef):
            methods = {
                child.name: [arg.arg for arg in child.args.args]
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            classes.append((node.lineno, node.end_lineno or node.lineno, methods))
    return classes, module


def _parameter_names(
    tables: tuple[list[tuple[int, int, dict[str, list[str]]]], dict[str, list[str]]],
    call: ast.Call,
) -> list[str]:
    """Parameter names of a same-file function the call resolves to.

    A definition inside the class enclosing the call wins, so sibling helpers
    sharing one name stay distinguishable.
    """
    name = _callee_name(call)
    if not name:
        return []
    classes, module = tables
    enclosing = [
        methods
        for start, end, methods in classes
        if start <= call.lineno <= end and name in methods
    ]
    names = enclosing[-1][name] if enclosing else module.get(name, [])
    if names and names[0] == "self":
        names = names[1:]
    return names


def _record_types(node: ast.expr) -> list[ast.Constant]:
    """``type`` values of the dict records inside a list or tuple literal."""
    if not isinstance(node, (ast.List, ast.Tuple)):
        return []
    found: list[ast.Constant] = []
    for element in node.elts:
        if not isinstance(element, ast.Dict):
            continue
        for key, value in zip(element.keys, element.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "type"
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                found.append(value)
    return found


def _type_tokens(value: str) -> list[str]:
    """The entity types a declared value carries, serialized forms included.

    ``entity_types`` reaches a DSPy boundary as a JSON array or a
    comma-separated string, so the raw value is not always one type.
    """
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, list):
        return [item for item in decoded if isinstance(item, str)]
    return [part.strip() for part in value.split(",") if part.strip()]


def _string_constants(node: ast.expr) -> list[ast.Constant]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [
            element
            for element in node.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        ]
    return []


def scan_python(path: str, source: str) -> list[Finding]:
    """Entity-type literals outside ``ENTITY_TYPES`` in one Python module."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    spans = _scopes(tree)
    tables = _parameter_tables(tree)
    findings: list[Finding] = []

    def report(node: ast.Constant, detector: str) -> None:
        for token in (
            _type_tokens(node.value) if detector == "entity-type key" else [node.value]
        ):
            if token not in ENTITY_TYPES:
                findings.append(
                    Finding(
                        path,
                        node.lineno,
                        _enclosing(spans, node.lineno),
                        token,
                        detector,
                    )
                )

    def report_records(node: ast.expr) -> None:
        for constant in _record_types(node):
            report(constant, "entity-owned list")

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
            for key, value in zip(node.keys, node.values):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    continue
                if _ENTITY_OWNER.search(key.value):
                    report_records(value)
                if key.value in _ENTITY_TYPE_KEYS:
                    for constant in _string_constants(value):
                        report(constant, "entity-type key")
                if (
                    key.value == "type"
                    and "text" in keys
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and _TYPE_TOKEN.match(value.value)
                ):
                    report(value, "entity record")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                name = getattr(target, "id", "") or getattr(target, "attr", "")
                if name and _ENTITY_OWNER.search(name):
                    report_records(node.value)
        elif isinstance(node, ast.Call):
            callee = _callee_name(node)
            owned_callee = bool(_ENTITY_OWNER.search(callee))
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                if _ENTITY_OWNER.search(keyword.arg):
                    report_records(keyword.value)
                if keyword.arg in _ENTITY_TYPE_KEYS:
                    for constant in _string_constants(keyword.value):
                        report(constant, "entity-type key")
                if (
                    keyword.arg == "type"
                    and owned_callee
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ):
                    report(keyword.value, "entity constructor")
            if node.args:
                names = [] if owned_callee else _parameter_names(tables, node)
                for index, argument in enumerate(node.args):
                    if owned_callee or (
                        index < len(names) and _ENTITY_OWNER.search(names[index])
                    ):
                        report_records(argument)
    return _dedup(findings)


def scan_json(path: str, payload: object, line: int = 0) -> list[Finding]:
    """Entity-type literals outside ``ENTITY_TYPES`` in one decoded document."""
    findings: list[Finding] = []

    def report(value: str, detector: str) -> None:
        for token in _type_tokens(value) if detector == "entity-type key" else [value]:
            if token not in ENTITY_TYPES:
                findings.append(Finding(path, line, "<document>", token, detector))

    def records(value: object) -> None:
        if isinstance(value, list):
            for element in value:
                if isinstance(element, dict) and isinstance(element.get("type"), str):
                    report(element["type"], "entity-owned list")

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(key, str) and _ENTITY_OWNER.search(key):
                    records(value)
                if key in _ENTITY_TYPE_KEYS:
                    for element in value if isinstance(value, list) else [value]:
                        if isinstance(element, str):
                            report(element, "entity-type key")
                walk(value)
            if (
                "text" in node
                and isinstance(node.get("type"), str)
                and _TYPE_TOKEN.match(node["type"])
            ):
                report(node["type"], "entity record")
        elif isinstance(node, list):
            for element in node:
                walk(element)

    walk(payload)
    return _dedup(findings)


def _locate(source: str, value: str) -> int:
    needle = json.dumps(value)
    for number, line in enumerate(source.splitlines(), 1):
        if needle in line:
            return number
    return 0


def scan_tree(root: Path) -> list[Finding]:
    """Every off-vocabulary entity-type literal under ``root``."""
    guard = Path(__file__).resolve()
    findings: list[Finding] = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in SCANNED_SUFFIXES or not path.is_file():
            continue
        if path.resolve() == guard:
            continue
        anchor = REPO_ROOT if REPO_ROOT in path.parents else root
        relative = path.relative_to(anchor).as_posix()
        source = path.read_text(errors="replace")
        if path.suffix == ".py":
            findings.extend(scan_python(relative, source))
        elif path.suffix == ".json":
            try:
                document = json.loads(source)
            except json.JSONDecodeError:
                continue
            findings.extend(
                Finding(f[0], _locate(source, f[3]), f[2], f[3], f[4])
                for f in scan_json(relative, document)
            )
        else:
            for number, line in enumerate(source.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    document = json.loads(line)
                except json.JSONDecodeError:
                    continue
                findings.extend(scan_json(relative, document, number))
    return _dedup(findings)


# (path, enclosing scope, entity type) of a fixture that exists to be refused.
_DELIBERATE_NEGATIVES = frozenset(
    {
        # test_gate_verdict_names_an_entity_type_outside_the_shipped_vocabulary
        # drifts the advertised example and pins the gate's refusal message.
        (
            "tests/routing/unit/synthetic/test_schemas.py",
            "test_gate_verdict_names_an_entity_type_outside_the_shipped_vocabulary",
            "ORG",
        ),
        # test_rejects_invalid_ground_truth_rows expects
        # "type must be in ENTITY_TYPES" for this parametrized corpus row.
        (
            "tests/runtime/unit/test_admin_entity_extraction_ground_truth.py",
            "test_upload_rejects_invalid_rows_before_store",
            "HUMAN",
        ),
        # The same parametrized corpus row for the whitespace-only type,
        # expecting "must be non-empty after stripping whitespace".
        (
            "tests/runtime/unit/test_admin_entity_extraction_ground_truth.py",
            "test_upload_rejects_invalid_rows_before_store",
            "  ",
        ),
        # test_mention_outside_the_schema_cannot_be_built asserts EntityMention
        # refuses a type whose case differs from the Literal.
        (
            "tests/agents/unit/test_entity_extraction_agent.py",
            "test_mention_outside_the_schema_cannot_be_built",
            "Person",
        ),
        # SCHEMA_VIOLATING_COMPLETIONS["type_outside_vocabulary"] is the server
        # response test_entity_extraction_structured_output requires to fail.
        (
            "tests/agents/unit/test_entity_extraction_structured_output.py",
            "<module>",
            "Person",
        ),
        # test_record_with_a_type_outside_the_vocabulary_is_refused_by_name
        # pins the optimizer CLI's refusal of a reviewer's free-text type.
        (
            "tests/runtime/unit/test_batch_optimization_modes.py",
            "test_record_with_a_type_outside_the_vocabulary_is_refused_by_name",
            "Technology",
        ),
        # test_entity_extraction_case_insensitive pairs a lowercased prediction
        # against the shipped type to pin the matcher's case handling.
        (
            "tests/finetuning/test_adapter_evaluator.py",
            "test_entity_extraction_case_insensitive",
            "person",
        ),
    }
)


@pytest.fixture(scope="module")
def tree_findings() -> list[Finding]:
    return scan_tree(TESTS_ROOT)


def test_no_fixture_carries_an_entity_type_the_gate_refuses(tree_findings) -> None:
    offenders = [
        finding
        for finding in tree_findings
        if finding.site not in _DELIBERATE_NEGATIVES
    ]

    assert offenders == [], (
        f"{len(offenders)} fixture entity types are outside ENTITY_TYPES "
        f"({', '.join(sorted(ENTITY_TYPES))}); rewrite each to the type the "
        "extractor emits for that mention, or allowlist it when the fixture "
        "exists to be refused:\n" + "\n".join(str(f) for f in offenders)
    )


def test_every_allowlisted_negative_still_exists(tree_findings) -> None:
    """A stale entry silently widens the allowlist for a future offender."""
    present = {finding.site for finding in tree_findings}

    assert sorted(_DELIBERATE_NEGATIVES - present) == []


_OFFENDING_MODULE = """
from cogniverse_agents.entity_extraction_agent import Entity


def build():
    entities = [{"text": "Meta AI", "type": "ORG"}]
    row = {"query": "q", "entities": [{"text": "JAX", "type": "PRODUCT"}]}
    mention = Entity(text="Vespa", type="TOOL")
    return entities, row, mention


class Holder:
    def _span(self, query, entities):
        return query, entities

    def probe(self):
        return self._span("q", [{"text": "eiffel tower", "type": "landmark"}])


def forward(generator):
    return generator.forward(entities=["radium"], entity_types=["MATERIAL"])
"""

_CLEAN_MODULE = (
    _OFFENDING_MODULE.replace('"ORG"', '"ORGANIZATION"')
    .replace('"PRODUCT"', '"TECHNOLOGY"')
    .replace('"TOOL"', '"TECHNOLOGY"')
    .replace('"landmark"', '"PLACE"')
    .replace('"MATERIAL"', '"CONCEPT"')
)


def test_detector_reports_every_offending_shape_in_a_synthetic_module() -> None:
    findings = scan_python("synthetic.py", _OFFENDING_MODULE)

    assert [tuple(finding)[1:] for finding in findings] == [
        (6, "build", "ORG", "entity-owned list"),
        (7, "build", "PRODUCT", "entity-owned list"),
        (8, "build", "TOOL", "entity constructor"),
        (17, "probe", "landmark", "entity-owned list"),
        (21, "forward", "MATERIAL", "entity-type key"),
    ]


def test_detector_reports_nothing_when_the_same_module_uses_shipped_types() -> None:
    assert scan_python("synthetic.py", _CLEAN_MODULE) == []


def test_detector_reads_an_entity_record_outside_an_entity_owner() -> None:
    """The record shape alone carries a type token, wherever it is written."""
    source = 'def f():\n    return render([{"text": "Meta AI", "type": "ORG"}])\n'

    assert [tuple(f)[1:] for f in scan_python("s.py", source)] == [
        (2, "f", "ORG", "entity record")
    ]


def test_detector_ignores_a_content_block_that_shares_the_record_keys() -> None:
    """``{"type": "text", "text": ...}`` is an LM message part, not an entity."""
    source = 'def f():\n    return [{"type": "text", "text": "hello"}]\n'

    assert scan_python("s.py", source) == []


def test_detector_reports_a_json_document_and_stays_quiet_on_a_clean_one() -> None:
    offending = {"query": "q", "entities": [{"text": "Meta AI", "type": "ORG"}]}
    clean = {"query": "q", "entities": [{"text": "Meta AI", "type": "ORGANIZATION"}]}

    assert (scan_json("corpus.json", offending), scan_json("corpus.json", clean)) == (
        [Finding("corpus.json", 0, "<document>", "ORG", "entity-owned list")],
        [],
    )


def test_tree_scan_reads_python_json_and_jsonl_and_skips_other_suffixes(
    tmp_path,
) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "test_fixture.py").write_text(
        'ROW = {"entities": [{"text": "Meta AI", "type": "ORG"}]}\n'
    )
    (tmp_path / "pkg" / "golden.json").write_text(
        json.dumps({"entities": [{"text": "JAX", "type": "PRODUCT"}]}, indent=2)
    )
    (tmp_path / "pkg" / "rows.jsonl").write_text(
        json.dumps({"entities": [{"text": "Vespa", "type": "TOOL"}]}) + "\n"
    )
    (tmp_path / "pkg" / "ignored.txt").write_text('{"entities": [{"type": "ORG"}]}\n')

    assert [tuple(f)[:2] + tuple(f)[3:4] for f in scan_tree(tmp_path)] == [
        ("pkg/golden.json", 5, "PRODUCT"),
        ("pkg/rows.jsonl", 1, "TOOL"),
        ("pkg/test_fixture.py", 1, "ORG"),
    ]
