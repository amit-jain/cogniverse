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
    uv run python scripts/ci_local.py -m agents -m runtime  # several
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
            if _needs_real_services(selection):
                continue
            key = (selection.paths, selection.ignores, selection.marker_expr)
            if key in seen:
                continue
            seen.add(key)
            selections.append(
                {
                    "module": workflow.name[: -len(_SUFFIX)],
                    "paths": list(selection.paths),
                    "ignores": list(selection.ignores),
                    "marker": selection.marker_expr,
                }
            )
    return selections


def build_argv(sel: dict) -> list[str]:
    argv = ["uv", "run", "python", "-m", "pytest", *sel["paths"]]
    argv += [f"--ignore={path}" for path in sel["ignores"]]
    if sel["marker"]:
        argv += ["-m", sel["marker"]]
    argv += ["-v", "-p", "no:cacheprovider", "--tb=long"]
    return argv


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
        "--list", action="store_true", help="print the commands without running"
    )
    args = parser.parse_args()

    selections = discover()
    if args.modules:
        known = {s["module"] for s in selections}
        unknown = [m for m in args.modules if m not in known]
        if unknown:
            print(f"No unit selections found for {', '.join(unknown)}.")
            return 1
        selections = [s for s in selections if s["module"] in args.modules]
    if not selections:
        print("No unit selections found.")
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
