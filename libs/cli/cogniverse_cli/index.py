"""Codebase indexing command — uploads files to runtime for Vespa indexing."""

from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

console = Console()

RUNTIME_URL = "http://localhost:28000"

TYPE_TO_PROFILE = {
    "code": "code_lateon_mv",
    "docs": "document_text_semantic",
    "video": "video_colpali_smol500_mv_frame",
}

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".cpp",
    ".h", ".hpp", ".rb", ".php", ".swift", ".kt", ".scala", ".sh", ".bash",
    ".yaml", ".yml", ".toml", ".json", ".sql", ".r", ".jl", ".lua", ".zig",
}

IGNORE_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".tox", "dist", "build", ".eggs", "*.egg-info",
    ".ruff_cache", ".cache", ".idea", ".vscode",
}


def _should_ignore(path: Path) -> bool:
    """Check if path should be ignored based on directory names."""
    for part in path.parts:
        if part in IGNORE_DIRS or part.endswith(".egg-info"):
            return True
    return False


def _load_gitignore(root: Path) -> list:
    """Load .gitignore patterns from the root directory."""
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
    """Check if a path matches any gitignore pattern (basic matching)."""
    rel = str(path.relative_to(root))
    for pattern in patterns:
        pattern = pattern.rstrip("/")
        if pattern in rel or rel.startswith(pattern):
            return True
        if path.name == pattern:
            return True
    return False


def collect_files(root: Path, content_type: str) -> list:
    """Collect files to index based on content type."""
    files = []
    gitignore_patterns = _load_gitignore(root)

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _should_ignore(path):
            continue
        if _matches_gitignore(path, root, gitignore_patterns):
            continue

        if content_type == "code":
            if path.suffix.lower() in CODE_EXTENSIONS:
                files.append(path)
        elif content_type == "docs":
            if path.suffix.lower() in {".md", ".txt", ".rst", ".pdf", ".html"}:
                files.append(path)
        elif content_type == "video":
            if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
                files.append(path)

    return sorted(files)


def index_files(
    root: Path,
    content_type: str,
    tenant_id: str,
    profile: Optional[str] = None,
    runtime_url: str = RUNTIME_URL,
) -> dict:
    """Index files from a directory into Vespa via the runtime ingestion API.

    Returns summary dict with counts.
    """
    profile = profile or TYPE_TO_PROFILE.get(content_type, "code_lateon_mv")
    files = collect_files(root, content_type)

    if not files:
        console.print(f"[yellow]No {content_type} files found in {root}[/yellow]")
        return {"files_found": 0, "files_indexed": 0, "chunks_created": 0}

    console.print(f"Found [bold]{len(files)}[/bold] {content_type} files in {root}")

    indexed = 0
    total_chunks = 0
    total_docs = 0
    errors = []

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

                try:
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
                        data = resp.json()
                        total_chunks += data.get("chunks_created", 0)
                        total_docs += data.get("documents_fed", 0)
                        indexed += 1
                    else:
                        errors.append((str(rel_path), resp.status_code, resp.text[:200]))
                except Exception as exc:
                    errors.append((str(rel_path), 0, str(exc)))

                progress.advance(task)

    summary = {
        "files_found": len(files),
        "files_indexed": indexed,
        "chunks_created": total_chunks,
        "documents_fed": total_docs,
        "errors": len(errors),
    }

    console.print()
    console.print(f"[bold green]Indexed {indexed}/{len(files)} files[/bold green]")
    console.print(f"  Chunks created: {total_chunks}")
    console.print(f"  Documents fed: {total_docs}")
    if errors:
        console.print(f"  [red]Errors: {len(errors)}[/red]")
        for path, code, msg in errors[:5]:
            console.print(f"    {path}: {code} {msg[:100]}")

    return summary
