"""Codebase indexing command — uploads files to runtime for Vespa indexing.

``cogniverse index <path> --type code|docs`` does two things in parallel:

1. **Content indexing** — uploads each file to ``/ingestion/upload`` with
   the appropriate profile so it becomes searchable via the normal
   semantic / hybrid search.

2. **Knowledge graph extraction** (code + text docs only) — pulls entities
   and relationships out of each file and upserts them to the graph via
   ``/graph/upsert``. The graph is queryable via ``cogniverse graph``.

The ``docs`` type fans out per file extension to the right content
profile (text → ``document_text_semantic``, video → video profile,
image → image profile, audio → audio profile). Code only maps to
``code_lateon_mv``.
"""

from pathlib import Path
from typing import Dict, List, Optional

import httpx
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

console = Console()

RUNTIME_URL = "http://localhost:28000"

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".cpp",
    ".h", ".hpp", ".rb", ".php", ".swift", ".kt", ".scala", ".sh", ".bash",
    ".yaml", ".yml", ".toml", ".json", ".sql", ".r", ".jl", ".lua", ".zig",
}

DOCS_EXT_TO_PROFILE = {
    ".md": "document_text_semantic",
    ".txt": "document_text_semantic",
    ".rst": "document_text_semantic",
    ".html": "document_text_semantic",
    ".htm": "document_text_semantic",
    ".pdf": "document_text_semantic",
    ".mp4": "video_colpali_smol500_mv_frame",
    ".mov": "video_colpali_smol500_mv_frame",
    ".mkv": "video_colpali_smol500_mv_frame",
    ".avi": "video_colpali_smol500_mv_frame",
    ".webm": "video_colpali_smol500_mv_frame",
    ".jpg": "image_colpali_mv",
    ".jpeg": "image_colpali_mv",
    ".png": "image_colpali_mv",
    ".webp": "image_colpali_mv",
    ".gif": "image_colpali_mv",
    ".wav": "audio_clap_semantic",
    ".mp3": "audio_clap_semantic",
    ".m4a": "audio_clap_semantic",
    ".flac": "audio_clap_semantic",
}

GRAPH_TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".html", ".htm", ".pdf"}

IGNORE_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".tox", "dist", "build", ".eggs",
    ".ruff_cache", ".cache", ".idea", ".vscode",
}

IGNORE_EXTENSIONS = {
    ".lock", ".sum", ".pyc", ".pyo", ".pyd", ".so", ".dylib", ".dll",
    ".class", ".o", ".a", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
    ".DS_Store",
}


def _should_ignore(path: Path) -> bool:
    for part in path.parts:
        if part in IGNORE_DIRS or part.endswith(".egg-info"):
            return True
    return False


def _load_gitignore(root: Path) -> list:
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return []
    patterns = []
    for line in gitignore.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(line)
    return patterns


def _matches_gitignore(path: Path, root: Path, patterns: list) -> bool:
    rel = str(path.relative_to(root))
    for pattern in patterns:
        pattern = pattern.rstrip("/")
        if pattern in rel or rel.startswith(pattern):
            return True
        if path.name == pattern:
            return True
    return False


def collect_files(root: Path, content_type: str) -> List[Path]:
    """Collect files to index based on content type.

    - ``code``: only files with known code extensions
    - ``docs``: everything else except obviously-binary / junk extensions
    """
    files: List[Path] = []
    gitignore_patterns = _load_gitignore(root)

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _should_ignore(path):
            continue
        if _matches_gitignore(path, root, gitignore_patterns):
            continue

        ext = path.suffix.lower()

        if content_type == "code":
            if ext in CODE_EXTENSIONS:
                files.append(path)
        elif content_type == "docs":
            if ext in IGNORE_EXTENSIONS:
                continue
            if ext in CODE_EXTENSIONS:
                continue
            if ext in DOCS_EXT_TO_PROFILE:
                files.append(path)

    return sorted(files)


def _profile_for_file(path: Path, content_type: str) -> str:
    """Return the Vespa profile to use for a given file."""
    if content_type == "code":
        return "code_lateon_mv"
    return DOCS_EXT_TO_PROFILE.get(path.suffix.lower(), "document_text_semantic")


def _upload_file(
    client: httpx.Client,
    file_path: Path,
    rel_path: Path,
    profile: str,
    tenant_id: str,
) -> Optional[Dict]:
    with open(file_path, "rb") as f:
        resp = client.post(
            "/ingestion/upload",
            files={"file": (str(rel_path), f, "application/octet-stream")},
            data={
                "profile": profile,
                "backend": "vespa",
                "tenant_id": tenant_id,
            },
        )
    if resp.status_code == 200:
        return resp.json()
    return {"error": resp.text[:200], "status": resp.status_code}


def _extract_and_upsert_graph(
    client: httpx.Client,
    file_path: Path,
    source_doc_id: str,
    tenant_id: str,
) -> Optional[Dict]:
    """Extract a file's nodes+edges locally and POST them to /graph/upsert.

    Graph extraction runs in the CLI process (not the runtime) so the user's
    local tree-sitter / GLiNER do the work. Only the resulting nodes+edges
    are shipped to the runtime — the runtime doesn't need to read the file.
    """
    try:
        from cogniverse_agents.graph.code_extractor import CodeExtractor
        from cogniverse_agents.graph.code_extractor import (
            supported_extensions as code_exts,
        )
        from cogniverse_agents.graph.doc_extractor import DocExtractor
    except ImportError:
        return None

    ext = file_path.suffix.lower()
    result = None
    try:
        if ext in code_exts():
            result = CodeExtractor().extract(file_path, tenant_id, source_doc_id)
        elif ext in GRAPH_TEXT_EXTENSIONS:
            result = DocExtractor().extract(file_path, tenant_id, source_doc_id)
    except Exception:
        return None

    if result is None or (not result.nodes and not result.edges):
        return None

    payload = {
        "tenant_id": tenant_id,
        "source_doc_id": source_doc_id,
        "nodes": [
            {
                "name": n.name,
                "description": n.description,
                "kind": n.kind,
                "mentions": n.mentions,
            }
            for n in result.nodes
        ],
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "relation": e.relation,
                "provenance": e.provenance,
                "confidence": e.confidence,
            }
            for e in result.edges
        ],
    }

    try:
        resp = client.post("/graph/upsert", json=payload, timeout=60.0)
        if resp.status_code == 200:
            return resp.json()
        return {"error": resp.text[:200], "status": resp.status_code}
    except Exception as exc:
        return {"error": str(exc)}


def index_files(
    root: Path,
    content_type: str,
    tenant_id: str,
    profile: Optional[str] = None,
    runtime_url: str = RUNTIME_URL,
) -> dict:
    """Index files from a directory into Vespa via the runtime ingestion API.

    Also extracts a knowledge graph (nodes + edges) from code and text
    files, and POSTs it to ``/graph/upsert``.

    Returns a summary dict with counts.
    """
    files = collect_files(root, content_type)

    if not files:
        console.print(f"[yellow]No {content_type} files found in {root}[/yellow]")
        return {
            "files_found": 0,
            "files_indexed": 0,
            "chunks_created": 0,
            "graph_nodes": 0,
            "graph_edges": 0,
        }

    console.print(f"Found [bold]{len(files)}[/bold] {content_type} files in {root}")

    indexed = 0
    total_chunks = 0
    total_docs = 0
    total_nodes = 0
    total_edges = 0
    errors: List = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing", total=len(files))

        with httpx.Client(timeout=120.0, base_url=runtime_url) as client:
            for file_path in files:
                rel_path = file_path.relative_to(root)
                progress.update(task, description=f"Indexing {rel_path}")

                file_profile = profile or _profile_for_file(file_path, content_type)

                upload_result = _upload_file(
                    client, file_path, rel_path, file_profile, tenant_id,
                )

                if upload_result and "error" not in upload_result:
                    total_chunks += upload_result.get("chunks_created", 0)
                    total_docs += upload_result.get("documents_fed", 0)
                    indexed += 1
                    source_doc_id = upload_result.get(
                        "video_id", str(rel_path)
                    ) or str(rel_path)
                elif upload_result:
                    errors.append((str(rel_path), upload_result.get("status", 0), upload_result.get("error", "")))
                    source_doc_id = str(rel_path)
                else:
                    source_doc_id = str(rel_path)

                graph_result = _extract_and_upsert_graph(
                    client, file_path, source_doc_id, tenant_id,
                )
                if graph_result and "error" not in graph_result:
                    total_nodes += graph_result.get("nodes_upserted", 0)
                    total_edges += graph_result.get("edges_upserted", 0)

                progress.advance(task)

    summary = {
        "files_found": len(files),
        "files_indexed": indexed,
        "chunks_created": total_chunks,
        "documents_fed": total_docs,
        "graph_nodes": total_nodes,
        "graph_edges": total_edges,
        "errors": len(errors),
    }

    console.print()
    console.print(f"[bold green]Indexed {indexed}/{len(files)} files[/bold green]")
    console.print(f"  Chunks created: {total_chunks}")
    console.print(f"  Documents fed: {total_docs}")
    if total_nodes or total_edges:
        console.print(f"  [cyan]Graph:[/cyan] {total_nodes} nodes, {total_edges} edges")
    if errors:
        console.print(f"  [red]Errors: {len(errors)}[/red]")
        for path, code, msg in errors[:5]:
            console.print(f"    {path}: {code} {msg[:100]}")

    return summary
