"""A real ``SearchAgent`` over a fixed result set.

The query encoder and the search backend are the only parts replaced: the
agent's own processing, its telemetry span and the results it records on that
span are the production code.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import numpy as np

from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from tests.utils.memory_store import InMemoryConfigStore

REPO_ROOT = Path(__file__).resolve().parents[2]

# (document id, score, document metadata) per hit, in backend order.
Hit = Tuple[str, float, Dict[str, Any]]


class StubEncoder:
    """Stands in for the remote query encoder."""

    def encode(self, query: str):
        return np.zeros((1, 128), dtype=np.float32)


class StubBackend:
    """Answers every search with ``hits`` as backend ``SearchResult`` rows,
    the shape ``_search_by_text`` reads (``.document.id``, ``.score``,
    ``.document.metadata``)."""

    def __init__(self, hits: List[Hit]) -> None:
        self._hits = hits

    def search(self, query_dict):
        return [
            SimpleNamespace(
                document=SimpleNamespace(id=doc_id, metadata=dict(metadata)),
                score=score,
            )
            for doc_id, score, metadata in self._hits
        ]


def memory_config_manager() -> ConfigManager:
    store = InMemoryConfigStore()
    store.initialize()
    return ConfigManager(store=store)


def stub_encoder_factory():
    """Patches the query encoder a ``SearchAgent`` builds with ``StubEncoder``,
    for the agent's construction."""
    return patch(
        "cogniverse_core.query.encoders.QueryEncoderFactory.create_encoder",
        return_value=StubEncoder(),
    )


def answer_with(agent: SearchAgent, hits: List[Hit]) -> SearchAgent:
    """Point a built ``agent`` at a backend answering ``hits``, with memory
    off so a search touches nothing else."""
    agent.query_encoder = StubEncoder()
    backend = StubBackend(hits)
    agent._get_backend = lambda: backend
    agent.is_memory_enabled = lambda: False
    return agent


def build_stub_search_agent(
    tenant_id: str, hits: List[Hit], config_manager: ConfigManager | None = None
) -> SearchAgent:
    """A ``SearchAgent`` for ``tenant_id`` whose backend answers ``hits``."""
    with stub_encoder_factory():
        agent = SearchAgent(
            deps=SearchAgentDeps(
                tenant_id=tenant_id,
                backend_url="http://localhost",
                backend_port=8080,
                auto_create_memory_schema=False,
            ),
            schema_loader=FilesystemSchemaLoader(
                base_path=REPO_ROOT / "configs" / "schemas"
            ),
            config_manager=config_manager or memory_config_manager(),
            port=8033,
        )
    return answer_with(agent, hits)
