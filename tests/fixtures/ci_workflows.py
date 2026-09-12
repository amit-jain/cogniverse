"""CI selections and path filters, parsed from ``.github/workflows/*.yml``.

A *selection* is one ``pytest`` invocation in a workflow job: the test paths it
names and the ``-m`` expression it applies. A test runs in CI only when some
selection names a path containing it AND that selection's marker expression
keeps it.

Only workflows that fire on an ordinary branch push or pull request gate a
commit. A tag-only workflow (``publish-packages.yml``) runs after the fact, so
its selections cover nothing on the commit that introduced a regression.

Path filters decide whether a workflow fires at all: a selection that would run
a test is inert on a commit whose files match none of the workflow's
``on.<trigger>.paths`` patterns.
"""

from __future__ import annotations

import dataclasses
import re
import shlex
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import yaml

GATING_TRIGGERS = ("push", "pull_request")


@dataclasses.dataclass(frozen=True)
class Selection:
    """One ``pytest <paths> [-m <expr>] [--ignore <path>]`` invocation."""

    workflow: str
    job: str
    paths: tuple[str, ...]
    marker_expr: str | None
    ignores: tuple[str, ...] = ()

    def names(self, test_path: str) -> bool:
        return any(test_path == p or test_path.startswith(p + "/") for p in self.paths)


@dataclasses.dataclass(frozen=True)
class Workflow:
    name: str
    commit_gating: bool
    # One entry per gating trigger; ``None`` means that trigger has no ``paths``
    # filter and therefore fires on every commit.
    trigger_filters: tuple[tuple[str, ...] | None, ...]
    selections: tuple[Selection, ...]

    def watches(self, root: str) -> bool:
        """Whether every commit touching the tree at ``root`` fires this workflow.

        ``root`` is repo-relative (``""`` is the repo itself). A trigger with no
        ``paths`` filter fires on everything; otherwise some pattern must cover
        the whole subtree, since a guard reads every file under it.
        """
        return self._fires_when(lambda pattern: _pattern_covers(pattern, root))

    def watches_file(self, path: str) -> bool:
        """Whether every commit touching the single file ``path`` fires this.

        One pattern matching that one path is enough, so a literal filename and
        an enclosing ``dir/**`` both qualify.
        """
        return self._fires_when(lambda pattern: _pattern_matches(pattern, path))

    def _fires_when(self, covered: Callable[[str], bool]) -> bool:
        if not self.commit_gating:
            return False
        return all(
            filters is None or any(covered(pattern) for pattern in filters)
            for filters in self.trigger_filters
        )


def _pattern_covers(pattern: str, root: str) -> bool:
    """Whether a GitHub ``paths`` pattern matches every file under ``root``."""
    if pattern == "**":
        return True
    if not pattern.endswith("/**"):
        return False
    base = pattern[: -len("/**")]
    return root == base or root.startswith(base + "/")


def _pattern_matches(pattern: str, path: str) -> bool:
    """Whether a GitHub ``paths`` pattern matches the single file ``path``."""
    return re.fullmatch(_glob_regex(pattern), path) is not None


def _glob_regex(pattern: str) -> str:
    """GitHub filter-pattern glob: ``**`` crosses ``/``, ``*`` and ``?`` do not."""
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "*":
            doubled = pattern[index + 1 : index + 2] == "*"
            parts.append(".*" if doubled else "[^/]*")
            index += 2 if doubled else 1
        elif character == "?":
            parts.append("[^/]")
            index += 1
        else:
            parts.append(re.escape(character))
            index += 1
    return "".join(parts)


def _logical_lines(script: str) -> list[str]:
    """Shell lines with backslash continuations joined into one line each."""
    return re.sub(r"\\\s*\n\s*", " ", script).splitlines()


def _ignored_paths(args: Sequence[str]) -> tuple[str, ...]:
    ignored = []
    for index, arg in enumerate(args):
        if arg.startswith("--ignore="):
            ignored.append(arg.split("=", 1)[1])
        elif arg == "--ignore" and index + 1 < len(args):
            ignored.append(args[index + 1])
    return tuple(path.rstrip("/") for path in ignored)


def _parse_invocation(
    line: str,
) -> tuple[tuple[str, ...], str | None, tuple[str, ...]] | None:
    """``(test paths, marker expression, ignored paths)`` for a pytest call."""
    try:
        tokens = shlex.split(line, comments=True)
    except ValueError:
        return None
    if not tokens or tokens[0] == "echo":
        return None
    for index, token in enumerate(tokens):
        if token != "pytest":
            continue
        if "install" in tokens[:index]:
            return None  # ``pip install pytest ...``
        args = tokens[index + 1 :]
        paths = tuple(a.rstrip("/") for a in args if a.startswith("tests/"))
        marker_expr = None
        if "-m" in args:
            marker_index = args.index("-m")
            if marker_index + 1 < len(args):
                marker_expr = args[marker_index + 1]
        return paths, marker_expr, _ignored_paths(args)
    return None


def _trigger_filters(on: object) -> tuple[bool, tuple[tuple[str, ...] | None, ...]]:
    if isinstance(on, str):
        on = {on: None}
    elif isinstance(on, list):
        on = dict.fromkeys(on)
    if not isinstance(on, dict):
        return False, ()
    filters: list[tuple[str, ...] | None] = []
    for trigger in GATING_TRIGGERS:
        if trigger not in on:
            continue
        config = on[trigger] if isinstance(on.get(trigger), dict) else {}
        if trigger == "push" and config.get("tags") and not config.get("branches"):
            continue  # release trigger: never fires on the commit itself
        paths = config.get("paths")
        filters.append(tuple(paths) if paths else None)
    return bool(filters), tuple(filters)


def _selections(name: str, doc: dict) -> tuple[Selection, ...]:
    found: list[Selection] = []
    for job_name, job in (doc.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            script = step.get("run")
            if not isinstance(script, str):
                continue
            for line in _logical_lines(script):
                parsed = _parse_invocation(line)
                if parsed is None or not parsed[0]:
                    continue
                found.append(Selection(name, job_name, parsed[0], parsed[1], parsed[2]))
    return tuple(found)


def load_workflows(workflows_dir: Path) -> tuple[Workflow, ...]:
    workflows = []
    for path in sorted(workflows_dir.glob("*.yml")):
        doc = yaml.safe_load(path.read_text())
        # PyYAML reads the bare key ``on`` as the boolean True.
        on = doc.get("on", doc.get(True))
        commit_gating, filters = _trigger_filters(on)
        workflows.append(
            Workflow(path.name, commit_gating, filters, _selections(path.name, doc))
        )
    return tuple(workflows)


def gating_selections(workflows: Iterable[Workflow]) -> tuple[Selection, ...]:
    return tuple(s for w in workflows if w.commit_gating for s in w.selections)


def workflows_running(
    workflows: Sequence[Workflow], test_path: str
) -> tuple[Workflow, ...]:
    return tuple(
        w
        for w in workflows
        if w.commit_gating and any(s.names(test_path) for s in w.selections)
    )
