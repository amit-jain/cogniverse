"""Code file extractor — tree-sitter AST walk to find functions, classes, imports.

Produces Node/Edge objects with EXTRACTED provenance (structural facts,
not LLM guesses). Coverage is whatever the ``_DEFINITION_TYPES`` /
``_IMPORT_TYPES`` / ``_CALL_TYPES`` node-type sets below match — primarily
Python, JavaScript/TypeScript, Go, Rust, and Java — and depends on which
tree-sitter grammars are installed.
"""

import logging
from pathlib import Path
from typing import List, Optional, Set

from cogniverse_agents.graph.graph_schema import Edge, ExtractionResult, Mention, Node

logger = logging.getLogger(__name__)

_LANG_BY_EXT = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
}


def supported_extensions() -> Set[str]:
    return set(_LANG_BY_EXT.keys())


class CodeExtractor:
    """Tree-sitter based extractor for code files."""

    def __init__(self) -> None:
        self._parsers: dict = {}

    def _get_parser(self, language: str):
        """Lazily load a tree-sitter parser for the language."""
        if language in self._parsers:
            return self._parsers[language]

        try:
            from tree_sitter import Language, Parser

            module_name, lang_attr = {
                "python": ("tree_sitter_python", "language"),
                "javascript": ("tree_sitter_javascript", "language"),
                "typescript": ("tree_sitter_typescript", "language_typescript"),
                "go": ("tree_sitter_go", "language"),
            }[language]

            import importlib

            mod = importlib.import_module(module_name)
            lang = Language(getattr(mod, lang_attr)())
            parser = Parser(lang)
            self._parsers[language] = parser
            return parser
        except Exception as exc:
            logger.warning("tree-sitter parser for %s unavailable: %s", language, exc)
            return None

    def extract(
        self,
        file_path: Path,
        tenant_id: str,
        source_doc_id: str,
    ) -> Optional[ExtractionResult]:
        """Extract nodes + edges from a code file. Returns None if unsupported."""
        ext = file_path.suffix.lower()
        language = _LANG_BY_EXT.get(ext)
        if not language:
            return None

        parser = self._get_parser(language)
        if parser is None:
            return None

        try:
            source = file_path.read_bytes()
        except OSError:
            return None

        tree = parser.parse(source)
        walker = _TreeWalker(source, language, tenant_id, source_doc_id, file_path.stem)
        walker.walk(tree.root_node)

        return ExtractionResult(
            source_doc_id=source_doc_id,
            nodes=walker.nodes,
            edges=walker.edges,
        )


class _TreeWalker:
    """Walks a tree-sitter AST and collects nodes/edges."""

    _DEFINITION_TYPES = {
        "function_definition",
        "method_definition",
        "function_declaration",
        "method_declaration",
        "class_definition",
        "class_declaration",
        "struct_item",
        "impl_item",
        "trait_item",
        "interface_declaration",
    }

    _IMPORT_TYPES = {
        "import_statement",
        "import_from_statement",
        "import_declaration",
        "use_declaration",
    }

    _CALL_TYPES = {
        "call",
        "call_expression",
        "function_call",
    }

    def __init__(
        self,
        source: bytes,
        language: str,
        tenant_id: str,
        source_doc_id: str,
        module_name: str,
    ) -> None:
        self._source = source
        self._language = language
        self._tenant_id = tenant_id
        self._source_doc_id = source_doc_id
        self._module_name = module_name
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self._defined_names: Set[str] = set()
        self._imports: List[str] = []
        self._seen_node_names: Set[str] = set()

        self._module_node = Node(
            tenant_id=tenant_id,
            name=module_name,
            description=f"Module {module_name} ({language})",
            kind="entity",
            mentions=[
                Mention(
                    source_doc_id=source_doc_id,
                    segment_id="module",
                    ts_start=0.0,
                    ts_end=0.0,
                    modality="code",
                    evidence_span=f"Module {module_name} ({language})",
                )
            ],
        )
        self.nodes.append(self._module_node)
        self._seen_node_names.add(module_name)

    def walk(self, node) -> None:
        """Depth-first walk of the AST."""
        node_type = node.type

        if node_type in self._DEFINITION_TYPES:
            name = self._extract_name(node)
            if name:
                self._add_definition_node(name, node_type)
                self._add_edge(
                    source=self._module_name,
                    target=name,
                    relation="defines",
                    segment_id=f"{self._segment_prefix(node_type)}:{name}",
                    evidence_span=self._truncate(self._text(node), 200),
                )

        elif node_type in self._IMPORT_TYPES:
            imported = self._extract_import(node)
            import_text = self._truncate(self._text(node), 200)
            for target in imported:
                self._imports.append(target)
                self._add_imported_node(target, import_text)
                self._add_edge(
                    source=self._module_name,
                    target=target,
                    relation="imports",
                    segment_id=f"import:{target}",
                    evidence_span=import_text,
                )

        elif node_type in self._CALL_TYPES:
            callee = self._extract_call_target(node)
            if callee and callee != self._module_name:
                enclosing = self._find_enclosing_def(node)
                if enclosing and callee != enclosing:
                    enclosing_segment = f"function:{enclosing}"
                    self._add_edge(
                        source=enclosing,
                        target=callee,
                        relation="calls",
                        segment_id=enclosing_segment,
                        evidence_span=self._truncate(self._text(node), 200),
                    )

        for child in node.children:
            self.walk(child)

    def _extract_name(self, node) -> Optional[str]:
        """Find the identifier child of a definition node."""
        for child in node.children:
            if child.type in (
                "identifier",
                "type_identifier",
                "name",
                "property_identifier",
            ):
                return self._text(child)
        return None

    def _extract_import(self, node) -> List[str]:
        """Extract imported module/name identifiers from an import node."""
        names: List[str] = []

        def walk_for_identifiers(n):
            if n.type in ("dotted_name", "identifier", "scoped_identifier"):
                text = self._text(n)
                if text:
                    names.append(text.split(".")[-1])
            for child in n.children:
                walk_for_identifiers(child)

        walk_for_identifiers(node)
        return names

    def _extract_call_target(self, node) -> Optional[str]:
        """Extract the callee name from a call expression."""
        for child in node.children:
            if child.type in (
                "identifier",
                "attribute",
                "member_expression",
                "field_expression",
            ):
                text = self._text(child)
                if text:
                    if "." in text:
                        return text.split(".")[-1]
                    return text
        return None

    def _find_enclosing_def(self, node) -> Optional[str]:
        """Walk up to find the nearest enclosing function/class name."""
        parent = node.parent
        while parent is not None:
            if parent.type in self._DEFINITION_TYPES:
                return self._extract_name(parent)
            parent = parent.parent
        return self._module_name

    def _text(self, node) -> str:
        return self._source[node.start_byte : node.end_byte].decode(
            "utf-8", errors="replace"
        )

    def _add_definition_node(self, name: str, node_type: str) -> None:
        if name in self._seen_node_names:
            return
        kind = "entity"
        description = f"{node_type.replace('_', ' ')}: {name}"
        segment_prefix = self._segment_prefix(node_type)
        self.nodes.append(
            Node(
                tenant_id=self._tenant_id,
                name=name,
                description=description,
                kind=kind,
                mentions=[
                    Mention(
                        source_doc_id=self._source_doc_id,
                        segment_id=f"{segment_prefix}:{name}",
                        ts_start=0.0,
                        ts_end=0.0,
                        modality="code",
                        evidence_span=description,
                    )
                ],
            )
        )
        self._seen_node_names.add(name)
        self._defined_names.add(name)

    def _add_imported_node(self, name: str, evidence_span: str) -> None:
        if name in self._seen_node_names:
            return
        self.nodes.append(
            Node(
                tenant_id=self._tenant_id,
                name=name,
                description=f"Imported symbol: {name}",
                kind="entity",
                mentions=[
                    Mention(
                        source_doc_id=self._source_doc_id,
                        segment_id=f"import:{name}",
                        ts_start=0.0,
                        ts_end=0.0,
                        modality="code",
                        evidence_span=evidence_span,
                    )
                ],
            )
        )
        self._seen_node_names.add(name)

    def _add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        segment_id: str,
        evidence_span: str,
    ) -> None:
        self.edges.append(
            Edge(
                tenant_id=self._tenant_id,
                source=source,
                target=target,
                relation=relation,
                evidence_span=evidence_span,
                segment_id=segment_id,
                ts_start=0.0,
                ts_end=0.0,
                modality="code",
                provenance="EXTRACTED",
                source_doc_id=self._source_doc_id,
            )
        )

    @staticmethod
    def _segment_prefix(node_type: str) -> str:
        if "class" in node_type or "struct" in node_type or "interface" in node_type:
            return "class"
        if "method" in node_type:
            return "method"
        return "function"

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1] + "…"
