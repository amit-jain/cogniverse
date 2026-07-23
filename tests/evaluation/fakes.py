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
