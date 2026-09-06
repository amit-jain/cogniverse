"""Find Vespa schema names in Python source and schema-bearing text."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping
from pathlib import Path

_SCHEMA_KEYS = {
    "schema",
    "schemas",
    "schema_name",
    "schema_names",
    "base_schema",
    "base_schemas",
    "base_schema_name",
    "base_schema_names",
    "full_schema_name",
    "full_schema_names",
    "tenant_schema",
    "tenant_schema_name",
    "default_schema",
    "default_schema_name",
    "restrict",
}
_SCHEMA_TEXT = re.compile(
    r"\b(?:video|image|audio|document|code)_[A-Za-z0-9_]+_(?:sv|mv)(?:_[A-Za-z0-9_]+)?\b"
)
_SCHEMA_FILE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)_schema\.json\b")
_YQL_SOURCE = re.compile(
    r"\bfrom\s+(?:sources\s+)?([A-Za-z][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z][A-Za-z0-9_]*)*)\s+where\b",
    re.IGNORECASE,
)
_DOCUMENT_URL = re.compile(r"/document/v1/[^/\s]+/([A-Za-z][A-Za-z0-9_]*)/")
_SCHEMA_ARGUMENTS = {
    "load_schema": (0,),
    "get_schema": (0,),
    "get_schema_info": (0,),
    "get_schema_fields": (0,),
    "get_schema_definition": (0,),
    "get_embedding_requirements": (0,),
    "get_required_embeddings": (0,),
    "get_embedding_field_names": (0,),
    "validate_schema": (0,),
    "schema_exists": (0,),
    "load_raw_schema_json": (0,),
    "schema_tensor_dim": (0,),
    "schema_full_name": (0,),
    "_is_single_vector_schema": (0,),
    "document_namespace": (0,),
    "deploy_schema": (1,),
    "delete_schema": (1,),
    "get_tenant_schema_name": (1,),
    "tenant_schema_exists": (1,),
}
_SCOPE_TYPES = (
    ast.Module,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
)


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Subscript):
        return _name(node.slice)
    return ""


def _schema_key(name: str) -> bool:
    return (
        name.lower() in _SCHEMA_KEYS
        or name.lower().endswith("_schema_name")
        or (name.isupper() and name.endswith("_SCHEMA"))
    )


def schema_references(source: str) -> set[tuple[int, str]]:
    """Return literal locations; aliases retain their defining line numbers."""
    tree = ast.parse(source)
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    bindings: dict[ast.AST, dict[str, list[ast.AST]]] = {}
    positions = dict(_SCHEMA_ARGUMENTS)

    def scope(node: ast.AST) -> ast.AST:
        while node in parents:
            node = parents[node]
            if isinstance(node, _SCOPE_TYPES):
                return node
        return tree

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and node.value:
                    bindings.setdefault(scope(node), {}).setdefault(
                        target.id, []
                    ).append(node.value)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args.posonlyargs + node.args.args
            if args and args[0].arg in {"self", "cls"}:
                args = args[1:]
            indexes = tuple(i for i, arg in enumerate(args) if _schema_key(arg.arg))
            if indexes:
                positions[node.name] = indexes
            bindings[node] = {arg.arg: [] for arg in args + node.args.kwonlyargs}

    def values(node: ast.AST, seen: frozenset[ast.AST] = frozenset()):
        if node in seen:
            return set()
        seen = seen | {node}
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {(node.lineno, node.value)} if node.value.strip() else set()
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return {item for element in node.elts for item in values(element, seen)}
        if isinstance(node, ast.Name):
            owner = scope(node)
            while True:
                definitions = bindings.get(owner, {})
                if node.id in definitions:
                    return {
                        item
                        for definition in definitions[node.id]
                        for item in values(definition, seen)
                    }
                if owner is tree:
                    return set()
                owner = scope(owner)
        return set()

    def tenant_argument(node: ast.AST, seen: frozenset[ast.AST] = frozenset()) -> bool:
        if node in seen:
            return False
        seen = seen | {node}
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return bool(re.fullmatch(r"[A-Za-z0-9_-]+:[A-Za-z0-9_-]+", node.value))
        if isinstance(node, (ast.Name, ast.Attribute)):
            if _name(node).lower().lstrip("_") in {"tenant", "tenant_id"}:
                return True
        if isinstance(node, ast.Name):
            owner = scope(node)
            while True:
                definitions = bindings.get(owner, {})
                if node.id in definitions:
                    return bool(definitions[node.id]) and all(
                        tenant_argument(definition, seen)
                        for definition in definitions[node.id]
                    )
                if owner is tree:
                    return False
                owner = scope(owner)
        return False

    text_exclusions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            _name(target) == "SCHEMA_REFERENCE_FIXTURES" for target in node.targets
        ):
            text_exclusions.append(
                ((node.lineno, node.col_offset), (node.end_lineno, node.end_col_offset))
            )
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            owner = parents.get(node)
            if isinstance(owner, ast.Expr):
                continue
            try:
                embedded = ast.parse(node.value)
            except (SyntaxError, ValueError):
                continue
            if any(
                isinstance(item, (ast.Assign, ast.FunctionDef, ast.Call))
                or (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant))
                for item in ast.walk(embedded)
            ):
                text_exclusions.append(
                    (
                        (node.lineno, node.col_offset),
                        (node.end_lineno, node.end_col_offset),
                    )
                )

    references = set()
    for node in ast.walk(tree):
        candidates = []
        if isinstance(node, ast.keyword) and _schema_key(node.arg or ""):
            candidates.append(node.value)
        elif isinstance(node, ast.Dict):
            candidates.extend(
                value
                for key, value in zip(node.keys, node.values)
                if key and _schema_key(_name(key))
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if node.value and any(_schema_key(_name(target)) for target in targets):
                candidates.append(node.value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args.posonlyargs + node.args.args
            defaults = [None] * (
                len(args) - len(node.args.defaults)
            ) + node.args.defaults
            for arg, default in zip(
                args + node.args.kwonlyargs, defaults + node.args.kw_defaults
            ):
                if default and _schema_key(arg.arg):
                    candidates.append(default)
        elif isinstance(node, ast.Call):
            indexes = positions.get(_name(node.func), ())
            if _name(node.func) == "schema_exists":
                if len(node.args) == 1:
                    indexes = (0,)
                elif len(node.args) == 2:
                    tenants = {
                        index
                        for index, argument in enumerate(node.args)
                        if tenant_argument(argument)
                    }
                    indexes = tuple(
                        index
                        for index in (0, 1)
                        if len(tenants) != 1 or index not in tenants
                    )
            candidates.extend(
                node.args[index] for index in indexes if index < len(node.args)
            )
            if _name(node.func) == "parametrize" and len(node.args) >= 2:
                names = values(node.args[0])
                if len(names) == 1:
                    columns = [name.strip() for name in next(iter(names))[1].split(",")]
                    rows = node.args[1]
                    if isinstance(rows, (ast.List, ast.Tuple)):
                        for row in rows.elts:
                            cells = (
                                row.elts
                                if len(columns) != 1
                                and isinstance(row, (ast.List, ast.Tuple))
                                else [row]
                            )
                            candidates.extend(
                                cell
                                for name, cell in zip(columns, cells)
                                if _schema_key(name)
                            )
        for candidate in candidates:
            references.update(values(candidate))

    for lineno, line in enumerate(source.splitlines(), 1):

        def visible(match):
            position = (lineno, len(line[: match.start()].encode("utf-8")))
            return not any(start <= position < end for start, end in text_exclusions)

        file_spans = [match.span() for match in _SCHEMA_FILE.finditer(line)]
        references.update(
            (
                lineno,
                match.group().rstrip("_")
                if line[match.end() :].startswith("{")
                else match.group(),
            )
            for match in _SCHEMA_TEXT.finditer(line)
            if visible(match)
            and not any(start <= match.start() < end for start, end in file_spans)
        )
        references.update(
            (lineno, match.group(1))
            for match in _SCHEMA_FILE.finditer(line)
            if visible(match)
        )
        references.update(
            (lineno, match.group(1))
            for match in _DOCUMENT_URL.finditer(line)
            if visible(match)
        )
        for match in _YQL_SOURCE.finditer(line):
            references.update(
                (lineno, value.strip())
                for value in match.group(1).split(",")
                if visible(match)
            )
    return references


def _source_facts(source: str) -> tuple[set[str], set[str], set[tuple[int, str]]]:
    """Collect local definitions, literal tenant IDs, and base-only references."""
    tree = ast.parse(source)
    defined = set()
    tenants = set()
    base_only = set()
    assignments = {
        target.id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    def strings(node, seen=frozenset()):
        if node in seen:
            return set()
        seen = seen | {node}
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {(node.lineno, node.value)}
        if isinstance(node, ast.Name) and node.id in assignments:
            return strings(assignments[node.id], seen)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.BinOp)):
            return {
                item
                for child in ast.iter_child_nodes(node)
                for item in strings(child, seen)
            }
        return set()

    def field(name, value):
        literals = strings(value)
        if name in {"tenant", "tenant_id", "TENANT", "TENANT_ID"}:
            tenants.update(value for _, value in literals)
        if name.lower().startswith("base_schema"):
            base_only.update(literals)

    def definition(value):
        if isinstance(value, dict) and ({"document", "fields"} & value.keys()):
            name = value.get("name")
            if isinstance(name, str):
                defined.add(name)

    for node in ast.walk(tree):
        if isinstance(node, ast.keyword):
            field(node.arg or "", node.value)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if node.value:
                    field(_name(target), node.value)
        elif isinstance(node, ast.Dict):
            items = {_name(key): value for key, value in zip(node.keys, node.values)}
            for name, value in items.items():
                field(name, value)
            if "name" in items and {"document", "fields"} & items.keys():
                defined.update(name for _, name in strings(items["name"]))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("{"):
                try:
                    definition(json.loads(node.value))
                except (json.JSONDecodeError, TypeError):
                    pass
        elif isinstance(node, ast.Call):
            method = _name(node.func)
            if (
                method in {"load_schema", "load_raw_schema_json", "schema_tensor_dim"}
                and node.args
            ):
                base_only.update(strings(node.args[0]))
            if method == "Schema":
                for keyword in node.keywords:
                    if keyword.arg == "name":
                        defined.update(name for _, name in strings(keyword.value))
            tenant_index = {
                "schema_full_name": 1,
                "deploy_schema": 0,
                "delete_schema": 0,
                "get_tenant_schema_name": 0,
            }.get(method)
            if tenant_index in range(len(node.args)):
                tenants.update(name for _, name in strings(node.args[tenant_index]))
            if method in {"write_text", "write_bytes"} and isinstance(
                node.func, ast.Attribute
            ):
                for _, value in strings(node.func.value):
                    if "configs/schemas/" not in value:
                        defined.update(
                            match.group(1) for match in _SCHEMA_FILE.finditer(value)
                        )

    tenants.update(re.findall(r"\btenant_id\s*=\s*['\"]([^'\"]+)['\"]", source))
    for line, text in enumerate(source.splitlines(), 1):
        if "configs/schemas/" in text:
            base_only.update(
                (line, match.group(1)) for match in _SCHEMA_FILE.finditer(text)
            )
    return defined, tenants, base_only


def missing_schema_references(
    root: Path,
    *,
    synthetic_schemas: Mapping[str, Mapping[str, str]] | None = None,
) -> list[str]:
    """Check libs/ and tests/; exceptions name an exact module, name, and reason."""
    synthetic_schemas = synthetic_schemas or {}
    for declarations in synthetic_schemas.values():
        for name, reason in declarations.items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_:]*", name) or not reason.strip():
                raise ValueError(
                    "Synthetic schema declarations need exact names and reasons"
                )
    available = {
        path.stem.removesuffix("_schema")
        for path in (root / "configs/schemas").glob("*_schema.json")
    }
    missing = []
    for directory in ("libs", "tests"):
        for path in sorted((root / directory).rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            source = path.read_text(encoding="utf-8")
            lines = source.splitlines()
            defined, tenants, base_only = _source_facts(source)
            permitted = available | defined
            declarations = synthetic_schemas.get(relative, {})
            qualified = {
                f"{name}_{suffix}"
                for name in permitted
                for tenant in tenants
                for suffix in {
                    tenant.replace(":", "_"),
                    tenant.replace(":", "_") if ":" in tenant else f"{tenant}_{tenant}",
                }
            }
            for line, name in sorted(schema_references(source)):
                candidates = (
                    permitted if (line, name) in base_only else permitted | qualified
                )
                shipped_file = "configs/schemas/" in lines[line - 1] and bool(
                    _SCHEMA_FILE.search(lines[line - 1])
                )
                if shipped_file:
                    candidates = available
                elif name in declarations:
                    continue
                if name not in candidates:
                    missing.append(f"{relative}:{line}: {name}")
    return missing
