"""Session guard for e2e CronWorkflows.

Each e2e session owns a UUID token. CronWorkflows that this session suspends
carry ``cogniverse.io/e2e-suspended=<token>`` so teardown can restore exactly
those workflows, and a later run can clear stale annotations from a prior
session before touching anything else.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

CRON_NAMESPACE = "cogniverse"
E2E_SUSPENDED_ANNOTATION = "cogniverse.io/e2e-suspended"


@dataclass(frozen=True, slots=True)
class CronRestoreResult:
    """Names restored in this call, plus any cronworkflows that failed."""

    restored_names: tuple[str, ...]
    failures: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CronSuspendResult:
    """Cronworkflows owned by this session, plus any failures encountered."""

    restore_names: tuple[str, ...]
    failures: tuple[str, ...] = ()


def new_session_token() -> str:
    """Return a fresh ownership token for one e2e session."""
    return uuid.uuid4().hex


def _run_kubectl(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    command = ["kubectl", *args]
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"kubectl command failed: {' '.join(command)}: {type(exc).__name__}: {exc}"
        ) from exc


def _get_cronworkflows() -> list[dict]:
    result = _run_kubectl(["get", "cronworkflows", "-n", CRON_NAMESPACE, "-o", "json"])
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            f"kubectl get cronworkflows failed: rc={result.returncode} "
            f"stderr={stderr!r}"
        )
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        raise RuntimeError(
            "kubectl get cronworkflows returned rc=0 with unparseable JSON: "
            f"stderr={stderr!r} stdout={stdout!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            "kubectl get cronworkflows returned rc=0 with unparseable JSON: "
            f"stderr={(result.stderr or '').strip()!r} stdout={(result.stdout or '').strip()!r}"
        )
    items = payload.get("items")
    if items is None:
        return []
    if not isinstance(items, list):
        raise RuntimeError(
            "kubectl get cronworkflows returned rc=0 with unparseable JSON: "
            f"stderr={(result.stderr or '').strip()!r} stdout={(result.stdout or '').strip()!r}"
        )
    return items


def _item_name(item: dict) -> str | None:
    metadata = item.get("metadata") or {}
    name = metadata.get("name")
    return name if isinstance(name, str) else None


def _item_annotation(item: dict) -> str | None:
    metadata = item.get("metadata") or {}
    annotations = metadata.get("annotations") or {}
    annotation = annotations.get(E2E_SUSPENDED_ANNOTATION)
    return annotation if isinstance(annotation, str) else None


def _item_suspended(item: dict) -> bool:
    spec = item.get("spec") or {}
    return bool(spec.get("suspend"))


def _cron_map() -> dict[str, dict]:
    return {
        name: item
        for item in _get_cronworkflows()
        if (name := _item_name(item)) is not None
    }


def _unique_names(names: Iterable[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)


def _patch_suspend(name: str, suspend: bool) -> subprocess.CompletedProcess[str]:
    patch = '{"spec":{"suspend":true}}' if suspend else '{"spec":{"suspend":false}}'
    return _run_kubectl(
        [
            "patch",
            "cronworkflow",
            name,
            "-n",
            CRON_NAMESPACE,
            "--type",
            "merge",
            "-p",
            patch,
        ]
    )


def _annotate_session(name: str, token: str) -> subprocess.CompletedProcess[str]:
    return _run_kubectl(
        [
            "annotate",
            "cronworkflow",
            name,
            f"{E2E_SUSPENDED_ANNOTATION}={token}",
            "-n",
            CRON_NAMESPACE,
            "--overwrite",
        ]
    )


def _remove_annotation(name: str) -> subprocess.CompletedProcess[str]:
    return _run_kubectl(
        [
            "annotate",
            "cronworkflow",
            name,
            f"{E2E_SUSPENDED_ANNOTATION}-",
            "-n",
            CRON_NAMESPACE,
        ]
    )


def restore_cronworkflows(names: Sequence[str]) -> CronRestoreResult:
    """Re-enable the named CronWorkflows when this session owns their annotation."""
    requested = _unique_names(names)
    if not requested:
        return CronRestoreResult(())

    cron_map = _cron_map()
    restored: list[str] = []
    failures: list[str] = []

    for name in requested:
        item = cron_map.get(name)
        if item is None:
            failures.append(name)
            continue
        if _item_annotation(item) is None:
            continue
        patch_result = _patch_suspend(name, False)
        if patch_result.returncode != 0:
            failures.append(name)
            continue
        remove_result = _remove_annotation(name)
        if remove_result.returncode != 0:
            failures.append(name)
            continue
        restored.append(name)

    return CronRestoreResult(tuple(restored), tuple(failures))


def restore_stale_cronworkflows(current_session_token: str) -> CronRestoreResult:
    """Restore every annotated CronWorkflow from a previous session."""
    cron_map = _cron_map()
    stale_names = [
        name
        for name, item in cron_map.items()
        if (annotation := _item_annotation(item)) is not None
        and annotation != current_session_token
    ]
    return restore_cronworkflows(stale_names)


def suspend_cronworkflows_for_session(
    current_session_token: str,
) -> CronSuspendResult:
    """Annotate and suspend every CronWorkflow this session now owns."""
    cron_map = _cron_map()
    restore_names: list[str] = []
    failures: list[str] = []
    seen: set[str] = set()

    for name, item in cron_map.items():
        annotation = _item_annotation(item)
        suspended = _item_suspended(item)

        if annotation == current_session_token:
            if name not in seen:
                restore_names.append(name)
                seen.add(name)
            if suspended:
                continue
            patch_result = _patch_suspend(name, True)
            if patch_result.returncode != 0:
                failures.append(name)
            continue

        if annotation is not None:
            continue

        if suspended:
            continue

        annotate_result = _annotate_session(name, current_session_token)
        if annotate_result.returncode != 0:
            failures.append(name)
            continue

        if name not in seen:
            restore_names.append(name)
            seen.add(name)

        patch_result = _patch_suspend(name, True)
        if patch_result.returncode != 0:
            failures.append(name)

    return CronSuspendResult(tuple(restore_names), tuple(failures))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.e2e.cron_guard")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("restore-stale", help="restore stale CronWorkflows")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "restore-stale":
        result = restore_stale_cronworkflows(new_session_token())
        if result.failures:
            print(
                "restore-stale failed for: " + ", ".join(result.failures),
                file=sys.stderr,
            )
            return 1
        if result.restored_names:
            print(", ".join(result.restored_names))
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
