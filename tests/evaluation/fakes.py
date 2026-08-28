"""In-memory DatasetStore implementations shared by evaluation unit tests.

Real subclasses of the production ABC (not MagicMocks), so tests exercise
the same method signatures and the not-found/outage contract production
code depends on.
"""

from typing import Any, Dict, Optional

import pandas as pd

from cogniverse_foundation.telemetry.providers.base import (
    DatasetNotFoundError,
    DatasetStore,
)


class InMemoryDatasetStore(DatasetStore):
    """Real DatasetStore implementation over a dict."""

    def __init__(self):
        self._frames: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    async def create_dataset(
        self, name: str, data: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        if name in self._frames:
            self._frames[name] = pd.concat(
                [self._frames[name], data], ignore_index=True
            )
        else:
            self._frames[name] = data.copy()
        self.metadata[name] = metadata or {}
        return f"ds-{name}"

    async def get_dataset(self, name: str) -> pd.DataFrame:
        if name not in self._frames:
            raise DatasetNotFoundError(f"Dataset '{name}' not found")
        return self._frames[name].copy()

    async def append_to_dataset(
        self,
        name: str,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if name not in self._frames:
            raise DatasetNotFoundError(f"Dataset '{name}' does not exist")
        self._frames[name] = pd.concat([self._frames[name], data], ignore_index=True)

    async def delete_dataset(self, name: str) -> bool:
        self.metadata.pop(name, None)
        return self._frames.pop(name, None) is not None


class FailingDatasetStore(InMemoryDatasetStore):
    """Store whose every call fails like a dead backend."""

    async def create_dataset(self, name, data, metadata=None):
        raise ConnectionError("connection refused: telemetry backend down")

    async def get_dataset(self, name):
        raise ConnectionError("connection refused: telemetry backend down")

    async def append_to_dataset(self, name, data, metadata=None):
        raise ConnectionError("connection refused: telemetry backend down")


class StubTelemetryProvider:
    def __init__(self, datasets):
        self.datasets = datasets


class StubArtifactManager:
    def __init__(self, *, raw=None, load_exc=None, save_exc=None):
        self._raw = raw
        self._load_exc = load_exc
        self._save_exc = save_exc
        self._tenant_id = "test_tenant:test_tenant"
        self.load_calls: list[tuple[str, str]] = []
        self.save_calls: list[dict] = []
        self.activate_calls: list[tuple[str, str, int]] = []

    async def load_blob(self, kind, key):
        self.load_calls.append((kind, key))
        if self._load_exc is not None:
            raise self._load_exc
        return self._raw

    async def save_blob_versioned(
        self,
        kind,
        key,
        content,
        *,
        consumed_example_ids,
        decision,
        scored,
        score,
        base_score,
        candidate_score,
    ):
        if self._save_exc is not None:
            raise self._save_exc
        self.save_calls.append(
            {
                "kind": kind,
                "key": key,
                "content": content,
                "consumed_example_ids": list(consumed_example_ids),
                "decision": decision,
                "scored": scored,
                "score": score,
                "base_score": base_score,
                "candidate_score": candidate_score,
            }
        )
        return "dataset-1", 1

    async def activate_version(self, kind, key, version):
        self.activate_calls.append((kind, key, version))
        return {"active": {"version": version, "activated_at": "2026-08-26T00:00:00Z"}}

    async def activate_version_guarded(self, kind, key, version):
        return await self.activate_version(kind, key, version)
