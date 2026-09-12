#!/usr/bin/env python3
"""Run each module's exact CI test selections locally, under the same
dead-port backend the test suite defaults to.

CI runs a *filtered* subset per module (each ``.github/workflows/*-tests.yml``
picks its own paths, ``-m`` marker and environment), and a test that silently
resolves config against an ambient Vespa passes locally against a developer's
k3d while failing in CI, where no Vespa is reachable. ``tests/conftest.py``
defaults the backend to a dead port so local and CI resolve config identically;
this script runs the *same selections CI runs*, unit and integration jobs alike,
parsed live from the workflow files, before a push. Integration tests provision
their own services through their fixtures.

A module is a workflow (``runtime`` for ``runtime-tests.yml``) or a test
package (``foundation`` selects every CI selection naming ``tests/foundation``).

Usage:
    uv run python scripts/ci_local.py                # every CI selection
    uv run python scripts/ci_local.py --unit         # unit jobs only
    uv run python scripts/ci_local.py -m evaluation  # one workflow
    uv run python scripts/ci_local.py -m foundation -m runtime  # several
    uv run python scripts/ci_local.py --list         # show the commands, run nothing

Exit code is non-zero if any selection has a failing/erroring test.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.fixtures.ci_workflows import Selection

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO / ".github" / "workflows"
_SUFFIX = "-tests.yml"


def _load_workflows(workflows_dir: Path):
    """Parse the workflows through the same model the CI-coverage guards use,
    so this script and those guards can never disagree about what CI runs."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tests.fixtures.ci_workflows import load_workflows

    return load_workflows(workflows_dir)


def _under_integration(path: str) -> bool:
    return "/integration/" in path.rstrip("/") + "/"


def _needs_real_services(selection: Selection) -> bool:
    """CI splits by job: the integration job provisions Vespa/Phoenix/an LM,
    the unit job provisions nothing. A job that names only integration
    directories is one too, whatever it is called."""
    return "integration" in selection.job or all(
        _under_integration(path) for path in selection.paths
    )


def discover(workflows_dir: Path = WORKFLOWS) -> list[dict]:
    selections: list[dict] = []
    seen: set[tuple] = set()
    for workflow in _load_workflows(workflows_dir):
        if not workflow.name.endswith(_SUFFIX):
            continue
        for selection in workflow.selections:
            key = (
                selection.paths,
                selection.ignores,
                selection.marker_expr,
                selection.env,
            )
            if key in seen:
                continue
            seen.add(key)
            selections.append(
                {
                    "module": workflow.name[: -len(_SUFFIX)],
                    "paths": list(selection.paths),
                    "ignores": list(selection.ignores),
                    "marker": selection.marker_expr,
                    "env": dict(selection.env),
                    "unit": not _needs_real_services(selection),
                }
            )
    return selections


def names_module(sel: dict, module: str) -> bool:
    """Whether ``module`` names this selection's workflow or a package it runs."""
    package = f"tests/{module}"
    return sel["module"] == module or any(
        path == package or path.startswith(package + "/") for path in sel["paths"]
    )


def build_argv(sel: dict) -> list[str]:
    argv = ["uv", "run", "python", "-m", "pytest", *sel["paths"]]
    argv += [f"--ignore={path}" for path in sel["ignores"]]
    if sel["marker"]:
        argv += ["-m", sel["marker"]]
    argv += ["-v", "-p", "no:cacheprovider", "--tb=long"]
    return argv


def command_line(sel: dict) -> str:
    """The shell command for a selection, its CI environment as a prefix."""
    return shlex.join(
        [f"{name}={value}" for name, value in sel["env"].items()] + build_argv(sel)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m",
        "--module",
        action="append",
        dest="modules",
        metavar="MODULE",
        help="only this workflow module (e.g. evaluation); repeat for several",
    )
    parser.add_argument(
        "--unit", action="store_true", help="only the selections of unit jobs"
    )
    parser.add_argument(
        "--list", action="store_true", help="print the commands without running"
    )
    args = parser.parse_args()

    selections = discover()
    if args.unit:
        selections = [s for s in selections if s["unit"]]
    if args.modules:
        unknown = [
            m for m in args.modules if not any(names_module(s, m) for s in selections)
        ]
        if unknown:
            print(f"No CI selections found for {', '.join(unknown)}.")
            return 1
        selections = [
            s for s in selections if any(names_module(s, m) for m in args.modules)
        ]
    if not selections:
        print("No CI selections found.")
        return 1

    # Force the dead-port default: strip any ambient BACKEND override so the
    # conftest fallback (an unbound port nothing listens on) takes effect.
    env = {
        k: v for k, v in os.environ.items() if k not in ("BACKEND_URL", "BACKEND_PORT")
    }
    env.setdefault("JAX_PLATFORM_NAME", "cpu")

    results: list[tuple[str, str, int]] = []
    for sel in selections:
        label = f"{sel['module']}: {' '.join(sel['paths'])} -m {sel['marker']!r}"
        if args.list:
            print(command_line(sel))
            continue
        print(f"\n=== {label} ===", flush=True)
        proc = subprocess.run(build_argv(sel), cwd=REPO, env={**env, **sel["env"]})
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
