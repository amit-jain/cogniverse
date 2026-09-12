"""A pinned digest agrees with the bytes pinned beside it.

``CanonicalReplacementRecord`` carries the canonical JSON and its SHA-256, and
a test pins both. Two independent literals for one derived fact drift apart: a
fixture rewrite updates the bytes and leaves the digest behind, and the pair
turns unsatisfiable — no run can pass it, and the red reports an arithmetic
mismatch instead of naming the fixture that moved.

The scan reads ``tests/`` and recomputes every digest pinned next to its bytes
in two positions: ``x.json`` and ``x.sha256`` compared against constants on one
object, and ``<prefix>json`` and ``<prefix>sha256`` keys in one dict literal. A
name bound to a string constant in the same scope resolves to that constant, so
the pinned bytes are found whether they sit inline or in a local.

Synthetic offenders drive each detector directly, because a repo-wide "no
offenders remain" assertion cannot protect its own detector once the tree is
clean.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import NamedTuple

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

REPO_ROOT = Path(__file__).resolve().parents[3]
TESTS_ROOT = REPO_ROOT / "tests"

_HEX_DIGITS = frozenset("0123456789abcdef")
_SHA256_HEX_LENGTH = 64
_BYTES_NAME = "json"
_DIGEST_NAME = "sha256"

ATTRIBUTE_PIN = "attribute pin"
DICT_KEY_PIN = "dict-key pin"


class Finding(NamedTuple):
    """One digest literal that no bytes pinned beside it can produce."""

    path: str
    line: int
    owner: str
    pinned: str
    recomputed: str
    detector: str


def digest_of(value: str) -> str:
    """The digest a store writes for ``value``: SHA-256 over its UTF-8 bytes."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and set(value) <= _HEX_DIGITS
    )


def _literal_str(node: ast.AST) -> str | None:
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, str) else None


def _resolved_str(node: ast.AST, bindings: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    return _literal_str(node)


def _scope_bindings(nodes: list[ast.AST]) -> dict[str, str]:
    """Names assigned exactly one string constant in this scope."""
    bindings: dict[str, str] = {}
    ambiguous: set[str] = set()
    for node in nodes:
        if not isinstance(node, ast.Assign):
            continue
        value = _literal_str(node.value)
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if value is None or target.id in bindings:
                ambiguous.add(target.id)
            else:
                bindings[target.id] = value
    for name in ambiguous:
        bindings.pop(name, None)
    return bindings


def _own_nodes(scope: ast.AST) -> list[ast.AST]:
    """Every descendant of ``scope`` that no nested function owns."""
    owned: list[ast.AST] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            owned.append(child)
            visit(child)

    visit(scope)
    return owned


def _scopes(tree: ast.Module) -> list[ast.AST]:
    return [tree] + [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _pairs_to_findings(
    path: str,
    owner: str,
    bytes_pins: list[tuple[int, str]],
    digest_pins: list[tuple[int, str]],
    detector: str,
) -> list[Finding]:
    """Every digest pin no bytes pin under the same owner can produce."""
    if not bytes_pins or not digest_pins:
        return []
    produced = {digest_of(value) for _, value in bytes_pins}
    return [
        Finding(path, line, owner, pinned, digest_of(bytes_pins[0][1]), detector)
        for line, pinned in digest_pins
        if pinned not in produced
    ]


def _attribute_findings(
    path: str, nodes: list[ast.AST], bindings: dict[str, str]
) -> list[Finding]:
    bytes_pins: dict[str, list[tuple[int, str]]] = {}
    digest_pins: dict[str, list[tuple[int, str]]] = {}
    for node in nodes:
        if not isinstance(node, ast.Compare) or len(node.comparators) != 1:
            continue
        if not isinstance(node.ops[0], ast.Eq):
            continue
        for attribute, other in (
            (node.left, node.comparators[0]),
            (node.comparators[0], node.left),
        ):
            if not isinstance(attribute, ast.Attribute):
                continue
            if attribute.attr not in (_BYTES_NAME, _DIGEST_NAME):
                continue
            value = _resolved_str(other, bindings)
            if value is None:
                continue
            owner = ast.unparse(attribute.value)
            if attribute.attr == _BYTES_NAME:
                bytes_pins.setdefault(owner, []).append((attribute.lineno, value))
            elif _is_digest(value):
                digest_pins.setdefault(owner, []).append((attribute.lineno, value))
    return [
        finding
        for owner in sorted(digest_pins)
        for finding in _pairs_to_findings(
            path,
            f"{owner}.{_DIGEST_NAME}",
            bytes_pins.get(owner, []),
            digest_pins[owner],
            ATTRIBUTE_PIN,
        )
    ]


def _dict_findings(
    path: str, nodes: list[ast.AST], bindings: dict[str, str]
) -> list[Finding]:
    findings: list[Finding] = []
    for node in nodes:
        if not isinstance(node, ast.Dict):
            continue
        bytes_pins: dict[str, list[tuple[int, str]]] = {}
        digest_pins: dict[str, list[tuple[int, str]]] = {}
        for key, value_node in zip(node.keys, node.values):
            key_name = _literal_str(key) if key is not None else None
            if key_name is None:
                continue
            value = _resolved_str(value_node, bindings)
            if value is None:
                continue
            if key_name.endswith(_BYTES_NAME):
                prefix = key_name[: -len(_BYTES_NAME)]
                bytes_pins.setdefault(prefix, []).append((value_node.lineno, value))
            elif key_name.endswith(_DIGEST_NAME) and _is_digest(value):
                prefix = key_name[: -len(_DIGEST_NAME)]
                digest_pins.setdefault(prefix, []).append((value_node.lineno, value))
        for prefix in sorted(digest_pins):
            findings.extend(
                _pairs_to_findings(
                    path,
                    f"{prefix}{_DIGEST_NAME}",
                    bytes_pins.get(prefix, []),
                    digest_pins[prefix],
                    DICT_KEY_PIN,
                )
            )
    return findings


def scan_python(path: str, source: str) -> list[Finding]:
    """Every stale digest pin in one module, ordered by line."""
    tree = ast.parse(source)
    findings: list[Finding] = []
    module_nodes = _own_nodes(tree)
    module_bindings = _scope_bindings(module_nodes)
    for scope in _scopes(tree):
        nodes = module_nodes if scope is tree else _own_nodes(scope)
        bindings = (
            module_bindings
            if scope is tree
            else {**module_bindings, **_scope_bindings(nodes)}
        )
        findings.extend(_attribute_findings(path, nodes, bindings))
        findings.extend(_dict_findings(path, nodes, bindings))
    return sorted(findings, key=lambda finding: (finding.line, finding.owner))


def scan_tree(root: Path) -> list[Finding]:
    return [
        finding
        for path in sorted(root.rglob("*.py"))
        for finding in scan_python(
            str(path.relative_to(root)), path.read_text(errors="replace")
        )
    ]


def _line_of(source: str, needle: str) -> int:
    for number, line in enumerate(source.splitlines(), start=1):
        if needle in line:
            return number
    raise AssertionError(f"{needle!r} is not in the synthetic module")


PINNED_BYTES = '{"item_id":"routing_17","status":"approved"}'
OTHER_BYTES = '{"item_id":"routing_18","status":"approved"}'
RECORD_KEY = "metadata.approval_record_json"
RECORD_DIGEST_KEY = "metadata.approval_record_sha256"
DECISION_DIGEST_KEY = "metadata.approval_decision_sha256"

_OFFENDING_MODULE = f"""
async def test_inline_attribute_pair(store):
    selected = await store.select_canonical()
    assert selected.json == {PINNED_BYTES!r}
    assert selected.sha256 == {digest_of(OTHER_BYTES)!r}


def test_dict_pair(row):
    assert row == {{
        {RECORD_KEY!r}: {PINNED_BYTES!r},
        {RECORD_DIGEST_KEY!r}: {digest_of(OTHER_BYTES)!r},
    }}
"""

_CLEAN_MODULE = f"""
async def test_inline_attribute_pair(store):
    selected = await store.select_canonical()
    assert selected.json == {PINNED_BYTES!r}
    assert selected.sha256 == {digest_of(PINNED_BYTES)!r}


def test_dict_pair(row):
    assert row == {{
        {RECORD_KEY!r}: {PINNED_BYTES!r},
        {RECORD_DIGEST_KEY!r}: {digest_of(PINNED_BYTES)!r},
    }}
"""

_LOCAL_NAME_MODULE = f"""
async def test_named_bytes(store):
    selected = await store.select_canonical()
    expected_json = (
        {PINNED_BYTES!r}
    )
    assert selected.json == expected_json
    assert selected.sha256 == {digest_of(OTHER_BYTES)!r}
"""

_UNPAIRED_DIGEST_MODULE = f"""
def test_decision_digest_has_no_bytes_beside_it(row):
    assert row == {{
        {DECISION_DIGEST_KEY!r}: {digest_of(OTHER_BYTES)!r},
    }}


async def test_digest_without_pinned_bytes(store):
    selected = await store.select_canonical()
    assert selected.sha256 == {digest_of(OTHER_BYTES)!r}
"""

_SEVERAL_RECORDS_MODULE = f"""
def test_two_records_in_one_scope(rows):
    assert rows == [
        {{
            {RECORD_KEY!r}: {PINNED_BYTES!r},
            {RECORD_DIGEST_KEY!r}: {digest_of(PINNED_BYTES)!r},
        }},
        {{
            {RECORD_KEY!r}: {OTHER_BYTES!r},
            {RECORD_DIGEST_KEY!r}: {digest_of(OTHER_BYTES)!r},
        }},
    ]
"""

_DIFFERENT_OBJECT_MODULE = f"""
async def test_two_objects(store):
    first = await store.select_canonical()
    second = await store.select_review_decision()
    assert first.json == {PINNED_BYTES!r}
    assert second.sha256 == {digest_of(OTHER_BYTES)!r}
"""


@pytest.fixture(scope="module")
def tree_findings() -> list[Finding]:
    return scan_tree(TESTS_ROOT)


def test_no_pinned_digest_disagrees_with_the_bytes_pinned_beside_it(
    tree_findings,
) -> None:
    assert tree_findings == []


def test_detector_reports_a_stale_attribute_pin_and_a_stale_dict_pin() -> None:
    stale = digest_of(OTHER_BYTES)
    fresh = digest_of(PINNED_BYTES)
    assert scan_python("synthetic.py", _OFFENDING_MODULE) == [
        Finding(
            "synthetic.py",
            _line_of(_OFFENDING_MODULE, "selected.sha256"),
            f"selected.{_DIGEST_NAME}",
            stale,
            fresh,
            ATTRIBUTE_PIN,
        ),
        Finding(
            "synthetic.py",
            _line_of(_OFFENDING_MODULE, RECORD_DIGEST_KEY),
            RECORD_DIGEST_KEY,
            stale,
            fresh,
            DICT_KEY_PIN,
        ),
    ]


def test_detector_stays_quiet_when_each_digest_matches_its_own_bytes() -> None:
    assert scan_python("synthetic.py", _CLEAN_MODULE) == []


def test_detector_resolves_bytes_bound_to_a_local_name() -> None:
    assert scan_python("synthetic.py", _LOCAL_NAME_MODULE) == [
        Finding(
            "synthetic.py",
            _line_of(_LOCAL_NAME_MODULE, "selected.sha256"),
            f"selected.{_DIGEST_NAME}",
            digest_of(OTHER_BYTES),
            digest_of(PINNED_BYTES),
            ATTRIBUTE_PIN,
        )
    ]


def test_detector_ignores_a_digest_with_no_bytes_pinned_beside_it() -> None:
    assert scan_python("synthetic.py", _UNPAIRED_DIGEST_MODULE) == []


def test_detector_accepts_each_record_when_one_scope_pins_several() -> None:
    assert scan_python("synthetic.py", _SEVERAL_RECORDS_MODULE) == []


def test_detector_pairs_a_digest_only_with_its_own_object() -> None:
    assert scan_python("synthetic.py", _DIFFERENT_OBJECT_MODULE) == []


def test_tree_scan_reads_python_and_skips_other_suffixes(tmp_path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "test_stale.py").write_text(_OFFENDING_MODULE)
    (tmp_path / "pkg" / "test_clean.py").write_text(_CLEAN_MODULE)
    (tmp_path / "pkg" / "ignored.txt").write_text(_OFFENDING_MODULE)

    assert [(finding.path, finding.detector) for finding in scan_tree(tmp_path)] == [
        ("pkg/test_stale.py", ATTRIBUTE_PIN),
        ("pkg/test_stale.py", DICT_KEY_PIN),
    ]
