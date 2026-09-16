#!/usr/bin/env python
"""Copy every base-name serving blob into ring slot 0.

``ArtifactManager.load_blob`` reads the ring slots ``--r0``/``--r1``/``--r2``.
A store written before the ring holds each blob under its base name
``dspy-{kind}-{tenant}-{key}``, which no reader resolves. This copies each base
name into slot 0 as revision 0 and leaves the base name in place.

Run once, against a stopped fleet, before starting the new pods. A blob
that already has a populated ring slot is left alone unless ``--force`` is
given, so a second run changes nothing.

    uv run python scripts/migrate_blob_ring_slots.py \\
        --phoenix-url http://phoenix:6006 --tenant acme:acme --tenant beta:beta

Exit status is 0 when every base-name blob in the store is attributed to a
supplied tenant and is readable through the ring afterwards, 2 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from phoenix.client import AsyncClient

from cogniverse_agents.optimizer.artifact_manager import (
    _BLOB_RING_SLOTS,
    ArtifactManager,
)
from cogniverse_telemetry_phoenix.provider import PhoenixProvider

# The blob categories ``save_blob``/``load_blob`` are called with.
BLOB_KINDS = ("config", "model", "xgboost", "workflow")

_SLOT_SUFFIX = re.compile(r"--r\d+$")
_VERSION_SUFFIX = re.compile(r"-v\d+$")

MIGRATION_REVISION = 0


def _manager(phoenix_url: str, tenant_id: str) -> ArtifactManager:
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": phoenix_url,
            "grpc_endpoint": phoenix_url,
        }
    )
    return ArtifactManager(provider, tenant_id=tenant_id)


def _is_base_name(name: str) -> bool:
    return (
        name.startswith("dspy-")
        and not _SLOT_SUFFIX.search(name)
        and not _VERSION_SUFFIX.search(name)
    )


def _blob_identity(manager: ArtifactManager, name: str) -> Optional[Tuple[str, str]]:
    """``(kind, key)`` when ``name`` is this tenant's base-name blob dataset."""
    for kind in BLOB_KINDS:
        prefix = f"dspy-{kind}-{manager._tenant_id}-"
        if name.startswith(prefix) and len(name) > len(prefix):
            return kind, name[len(prefix) :]
    return None


def _base_content(frame: pd.DataFrame, name: str) -> str:
    """The single string payload a pre-ring ``save_blob`` wrote."""
    if "content" in frame.columns:
        rows = [row.to_dict() for _, row in frame.iterrows()]
    elif "input" in frame.columns:
        rows = [
            row if isinstance(row, dict) else {"content": row}
            for row in frame["input"].tolist()
        ]
    else:
        raise ValueError(f"{name} has unexpected columns: {list(frame.columns)}")
    contents = [row.get("content") for row in rows]
    if len(contents) != 1 or not isinstance(contents[0], str):
        raise ValueError(
            f"{name} is not a single-row blob dataset: rows={len(contents)}"
        )
    return contents[0]


async def _migrate_blob(
    manager: ArtifactManager,
    kind: str,
    key: str,
    base_name: str,
    existing: set,
    *,
    force: bool,
    dry_run: bool,
) -> str:
    occupied = [
        manager._blob_slot_name(kind, key, slot)
        for slot in range(_BLOB_RING_SLOTS)
        if manager._blob_slot_name(kind, key, slot) in existing
    ]
    if occupied and not force:
        return "already_in_ring"

    frame = await manager._provider.datasets.get_dataset(name=base_name)
    content = _base_content(frame, base_name)
    if dry_run:
        return "would_migrate"

    slot_name = manager._blob_slot_name(kind, key, MIGRATION_REVISION)
    await manager._provider.datasets.delete_dataset(slot_name)
    await manager._provider.datasets.create_dataset(
        name=slot_name,
        data=pd.DataFrame(
            [{"content": content, "blob_revision": str(MIGRATION_REVISION)}]
        ),
        metadata={
            "artifact_type": f"blob_{kind}",
            "key": key,
            "tenant_id": manager._tenant_id,
            "blob_revision": MIGRATION_REVISION,
            "input_keys": ["content", "blob_revision"],
            "output_keys": [],
        },
    )
    if await manager.load_blob(kind, key) != content:
        raise RuntimeError(
            f"{base_name} is not readable through the ring after migration"
        )
    return "migrated"


async def migrate(
    phoenix_url: str,
    tenants: Sequence[str],
    *,
    force: bool = False,
    dry_run: bool = False,
) -> Tuple[Dict[str, Dict[str, List[str]]], List[str]]:
    """Migrate every tenant's base-name blobs; return the summary and strays."""
    client = AsyncClient(base_url=phoenix_url)
    names = {row["name"] for row in await client.datasets.list()}
    base_names = {name for name in names if _is_base_name(name)}

    summary: Dict[str, Dict[str, List[str]]] = {}
    attributed: set = set()
    for tenant_id in tenants:
        manager = _manager(phoenix_url, tenant_id)
        outcomes: Dict[str, List[str]] = {}
        for base_name in sorted(base_names):
            identity = _blob_identity(manager, base_name)
            if identity is None:
                continue
            attributed.add(base_name)
            kind, key = identity
            outcome = await _migrate_blob(
                manager,
                kind,
                key,
                base_name,
                names,
                force=force,
                dry_run=dry_run,
            )
            outcomes.setdefault(outcome, []).append(base_name)
        summary[manager._tenant_id] = outcomes
    return summary, sorted(base_names - attributed)


def _print_summary(
    summary: Dict[str, Dict[str, List[str]]], unattributed: List[str]
) -> None:
    for tenant_id in sorted(summary):
        outcomes = summary[tenant_id]
        counts = ", ".join(
            f"{outcome}={len(blobs)}" for outcome, blobs in sorted(outcomes.items())
        )
        print(f"{tenant_id}: {counts or 'no base-name blobs'}")
        for outcome, blobs in sorted(outcomes.items()):
            for blob in blobs:
                print(f"  {outcome}: {blob}")
    for blob in unattributed:
        print(f"UNATTRIBUTED: {blob}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phoenix-url", required=True)
    parser.add_argument(
        "--tenant",
        dest="tenants",
        action="append",
        required=True,
        help="tenant id to migrate; repeat for each tenant",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite slot 0 even when the blob already has a populated slot",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    summary, unattributed = asyncio.run(
        migrate(
            args.phoenix_url,
            args.tenants,
            force=args.force,
            dry_run=args.dry_run,
        )
    )
    _print_summary(summary, unattributed)
    return 2 if unattributed else 0


if __name__ == "__main__":
    sys.exit(main())
