#!/usr/bin/env python3
"""Run each module's exact CI unit-test selection locally, under the same
dead-port backend the test suite defaults to.

CI runs a *filtered* subset per module (each ``.github/workflows/*-tests.yml``
picks its own ``-m`` marker), and a test that silently resolves config against
an ambient Vespa passes locally against a developer's k3d while failing in CI,
where no Vespa is reachable. ``tests/conftest.py`` now defaults the backend to a
dead port so local and CI resolve config identically — this script closes the
loop by running the *same test selection CI runs* (parsed live from the
workflow files, so it can't drift) before a push.

Usage:
    uv run python scripts/ci_local.py                # all modules' unit selections
    uv run python scripts/ci_local.py -m evaluation  # one workflow
    uv run python scripts/ci_local.py --list         # show the commands, run nothing

Exit code is non-zero if any selection has a failing/erroring test.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO / ".github" / "workflows"


def _iter_run_blocks(node):
    """Yield every ``run:`` string in a parsed workflow document."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "run" and isinstance(value, str):
                yield value
            else:
                yield from _iter_run_blocks(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_run_blocks(item)


def _pytest_lines(run_block: str) -> list[str]:
    """Reconstruct pytest command lines from a shell ``run`` block, dropping
    comment lines and joining ``\\`` continuations."""
    kept = [ln for ln in run_block.splitlines() if not ln.strip().startswith("#")]
    joined = re.sub(r"\\\n", " ", "\n".join(kept))
    return [ln.strip() for ln in joined.splitlines() if "python -m pytest" in ln]


def _unit_selection(cmd: str) -> tuple[list[str], str | None] | None:
    """Extract (test paths, marker) from a pytest command, keeping only
    selections that target a ``/unit`` path (integration needs real services).

    Search only the portion after the ``pytest`` token so the ``-m`` in
    ``python -m pytest`` is never mistaken for the pytest ``-m`` marker."""
    after = cmd.split("pytest", 1)[1] if "pytest" in cmd else cmd
    paths = re.findall(r"(tests/[^\s\\]+)", after)
    paths = [p for p in paths if not p.endswith(".xml")]
    if not any("/unit" in p for p in paths):
        return None
    marker_match = (
        re.search(r'-m\s+"([^"]+)"', after)
        or re.search(r"-m\s+'([^']+)'", after)
        or re.search(r"-m\s+(\S+)", after)
    )
    return paths, (marker_match.group(1) if marker_match else None)


def discover() -> list[dict]:
    selections: list[dict] = []
    seen: set[tuple] = set()
    for wf in sorted(WORKFLOWS.glob("*-tests.yml")):
        module = wf.name[: -len("-tests.yml")]
        doc = yaml.safe_load(wf.read_text())
        for block in _iter_run_blocks(doc):
            for cmd in _pytest_lines(block):
                sel = _unit_selection(cmd)
                if sel is None:
                    continue
                paths, marker = sel
                key = (tuple(paths), marker)
                if key in seen:
                    continue
                seen.add(key)
                selections.append({"module": module, "paths": paths, "marker": marker})
    return selections


def build_argv(sel: dict) -> list[str]:
    argv = ["uv", "run", "python", "-m", "pytest", *sel["paths"]]
    if sel["marker"]:
        argv += ["-m", sel["marker"]]
    argv += ["-q", "-p", "no:cacheprovider", "--tb=short"]
    return argv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m", "--module", help="only this workflow module (e.g. evaluation)"
    )
    parser.add_argument(
        "--list", action="store_true", help="print the commands without running"
    )
    args = parser.parse_args()

    selections = discover()
    if args.module:
        selections = [s for s in selections if s["module"] == args.module]
    if not selections:
        print(
            f"No unit selections found{f' for {args.module}' if args.module else ''}."
        )
        return 1

    # Force the dead-port default: strip any ambient BACKEND override so the
    # conftest fallback (an unbound port nothing listens on) takes effect.
    env = {
        k: v for k, v in os.environ.items() if k not in ("BACKEND_URL", "BACKEND_PORT")
    }
    env.setdefault("JAX_PLATFORM_NAME", "cpu")

    results: list[tuple[str, str, int]] = []
    for sel in selections:
        argv = build_argv(sel)
        label = f"{sel['module']}: {' '.join(sel['paths'])} -m {sel['marker']!r}"
        if args.list:
            print(shlex.join(argv))
            continue
        print(f"\n=== {label} ===", flush=True)
        proc = subprocess.run(argv, cwd=REPO, env=env)
        results.append((sel["module"], label, proc.returncode))

    if args.list:
        return 0

    print("\n" + "=" * 70)
    failed = [r for r in results if r[2] != 0]
    for module, label, code in results:
        print(f"  {'PASS' if code == 0 else 'FAIL':4}  {label}")
    print("=" * 70)
    print(f"{len(results) - len(failed)}/{len(results)} selections passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
