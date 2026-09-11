"""Build-time network fetches in every shipped Dockerfile retry and never pipe into a shell.

One un-retried download inside a RUN layer fails a 40-minute image build on a
single dropped stream; a remote script piped into ``sh`` hides which download
failed and runs whatever came back. Both shapes are refused in every Dockerfile
the image tooling builds.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from cogniverse_cli.images import IMAGE_INPUT_PATHS

_REPO_ROOT = Path(__file__).resolve().parents[3]

_REMOTE_FETCH = re.compile(
    r"\b(?:curl|wget)\b[^|;&]*?https?://(?!localhost\b|127\.0\.0\.1)"
)
_RETRY_FLAG = re.compile(r"(?:^|\s)--retry(?:\s|=)")
_PIPE_TO_SHELL = re.compile(r"\|\s*(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*(?:sh|bash)\b")


def _run_instructions(text: str) -> list[tuple[int, str]]:
    """Yield ``(first_line_number, joined_shell)`` for every RUN instruction."""
    out: list[tuple[int, str]] = []
    start = 0
    buf: list[str] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        if not buf:
            if not raw.lstrip().startswith("RUN "):
                continue
            start = number
        buf.append(raw.rstrip("\\").rstrip() if raw.rstrip().endswith("\\") else raw)
        if raw.rstrip().endswith("\\"):
            continue
        out.append((start, " ".join(part.strip() for part in buf)[len("RUN ") :]))
        buf = []
    return out


def offenders(text: str) -> list[tuple[int, str]]:
    """Return ``(line, reason)`` for each RUN that fetches remotely without a retry or pipes into a shell."""
    found: list[tuple[int, str]] = []
    for line, shell in _run_instructions(text):
        if not _REMOTE_FETCH.search(shell):
            continue
        if _PIPE_TO_SHELL.search(shell):
            found.append((line, "remote fetch piped into a shell"))
        if not _RETRY_FLAG.search(shell):
            found.append((line, "remote fetch without --retry"))
    return found


def _built_dockerfiles() -> list[str]:
    return sorted(
        {
            p
            for paths in IMAGE_INPUT_PATHS.values()
            for p in paths
            if Path(p).name == "Dockerfile"
        }
    )


def _dockerfiles_on_disk() -> list[str]:
    return sorted(
        str(p.relative_to(_REPO_ROOT)) for p in _REPO_ROOT.glob("*/*/Dockerfile")
    )


def test_the_tooling_builds_exactly_the_dockerfiles_on_disk() -> None:
    assert _built_dockerfiles() == _dockerfiles_on_disk()


@pytest.mark.parametrize("dockerfile", _dockerfiles_on_disk())
def test_build_time_fetches_retry_and_never_pipe_into_a_shell(dockerfile: str) -> None:
    text = (_REPO_ROOT / dockerfile).read_text()
    assert offenders(text) == [], f"{dockerfile}: {offenders(text)}"


class TestDetector:
    def test_remote_curl_without_retry_is_named_with_its_run_line(self) -> None:
        text = "FROM scratch\nRUN apt-get update\nRUN curl -fsSL https://example.com/x -o /tmp/x\n"
        assert offenders(text) == [(3, "remote fetch without --retry")]

    def test_retrying_download_is_clean(self) -> None:
        text = (
            "RUN curl -fsSL --retry 5 https://example.com/x -o /tmp/x && unzip /tmp/x\n"
        )
        assert offenders(text) == []

    def test_pipe_into_shell_is_refused_even_with_retry(self) -> None:
        text = "RUN curl -fsSL --retry 5 https://example.com/install.sh | PREFIX=/usr/local sh\n"
        assert offenders(text) == [(1, "remote fetch piped into a shell")]

    def test_localhost_healthcheck_style_curl_is_not_a_fetch(self) -> None:
        text = "RUN curl -f http://localhost:8000/health || exit 1\nHEALTHCHECK CMD curl -f http://localhost:8000/health\n"
        assert offenders(text) == []

    def test_wget_is_covered_and_continuations_report_the_first_line(self) -> None:
        text = "FROM scratch\nRUN set -e \\\n    && wget -q \\\n       https://example.com/model.bin \\\n    && echo done\n"
        assert offenders(text) == [(2, "remote fetch without --retry")]
