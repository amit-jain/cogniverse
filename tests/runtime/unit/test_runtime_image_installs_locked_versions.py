"""Every package the runtime image installs comes at its ``uv.lock`` version.

``uv sync --frozen`` installs the lock. A ``uv pip install`` after it resolves
against PyPI instead: an unconstrained ``datasets`` moved ``huggingface-hub``
past the range the locked ``transformers`` accepts, and ``import transformers``
raised in the deployed runtime while every test, run against the lock, passed.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DOCKERFILE = REPO / "libs" / "runtime" / "Dockerfile"

EXPORT = (
    "uv export --frozen --no-hashes --no-emit-workspace --all-packages "
    "--all-groups -o /tmp/locked.txt"
)


def _commands(dockerfile: str) -> list[str]:
    """Each shell command of each RUN instruction, continuations joined."""
    joined = re.sub(r"\\\n\s*", " ", dockerfile)
    commands: list[str] = []
    for line in joined.splitlines():
        if not line.startswith("RUN "):
            continue
        body = re.sub(r"^RUN\s+(--\S+\s+)*", "", line)
        commands += [" ".join(part.split()) for part in body.split("&&")]
    return commands


def pip_installs(dockerfile: str) -> list[str]:
    return [
        command
        for command in _commands(dockerfile)
        if re.match(r"(uv )?pip install\b", command)
    ]


def unlocked_installs(dockerfile: str, constraints: str) -> list[str]:
    """The pip installs that do not resolve against ``constraints``."""
    return [
        command
        for command in pip_installs(dockerfile)
        if not re.search(rf"(-c|--constraint)[ =]{re.escape(constraints)}\b", command)
    ]


def test_runtime_image_pip_installs_resolve_against_the_lock():
    dockerfile = DOCKERFILE.read_text()
    commands = _commands(dockerfile)

    assert pip_installs(dockerfile) == [
        "uv pip install pylate --no-deps -c /tmp/locked.txt",
        "uv pip install datasets -c /tmp/locked.txt",
        "uv pip install -e ./libs/telemetry-phoenix -c /tmp/locked.txt",
    ]
    assert unlocked_installs(dockerfile, "/tmp/locked.txt") == []
    assert commands.count(EXPORT) == 1
    assert commands.index(EXPORT) < commands.index(pip_installs(dockerfile)[0])


def test_runtime_image_build_imports_what_the_installs_can_break():
    assert (
        '/app/.venv/bin/python -c "import umap, transformers, datasets"'
        in _commands(DOCKERFILE.read_text())
    )


_UNLOCKED = """\
FROM python:3.12-slim AS builder
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \\
    uv sync --package cogniverse-runtime --frozen && \\
    uv pip install pylate --no-deps && \\
    uv pip install datasets -c /tmp/other.txt && \\
    uv pip install -e ./libs/telemetry-phoenix --constraint=/tmp/locked.txt && \\
    pip install requests && \\
    rm -rf /root/.cache/pip
RUN uv pip install umap-learn -c /tmp/locked.txt
"""


def test_detector_names_exactly_the_installs_outside_the_lock():
    assert pip_installs(_UNLOCKED) == [
        "uv pip install pylate --no-deps",
        "uv pip install datasets -c /tmp/other.txt",
        "uv pip install -e ./libs/telemetry-phoenix --constraint=/tmp/locked.txt",
        "pip install requests",
        "uv pip install umap-learn -c /tmp/locked.txt",
    ]
    assert unlocked_installs(_UNLOCKED, "/tmp/locked.txt") == [
        "uv pip install pylate --no-deps",
        "uv pip install datasets -c /tmp/other.txt",
        "pip install requests",
    ]
