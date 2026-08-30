"""Mem0's LLM provider sends the real inference bearer, never a sentinel.

No caller of ``initialize_memory`` passes ``llm_api_key``, so Mem0's OpenAI
client sent ``not-required`` to the Modal-hosted primary LLM and every memory
extraction failed with 401 — logged by ``MemoryAwareMixin.update_memory`` as
``Failed to update memory`` and read as a memory that never learns. The manager
resolves the key by the same rule as ``create_dspy_lm``, before any side effect.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

PLACEHOLDER = "placeholder-no-auth-needed"
MODAL = "https://amit-jain--cogniverse-vllm-llm-student-inference.modal.run/v1"
IN_CLUSTER = "http://cogniverse-vllm-llm-student:8000/v1"


def _fresh_manager() -> Mem0MemoryManager:
    return Mem0MemoryManager(tenant_id=f"auth:{uuid.uuid4().hex[:12]}")


def _initialize(manager: Mem0MemoryManager, llm_base_url: str, **kwargs) -> None:
    backend = MagicMock()
    backend.get_tenant_schema_name.return_value = "agent_memories_auth"
    registry = MagicMock()
    registry.get_ingestion_backend.return_value = backend
    store = InMemoryConfigStore()
    store.initialize()
    with (
        patch(
            "cogniverse_core.registries.backend_registry.get_backend_registry",
            return_value=registry,
        ),
        patch("cogniverse_core.memory.manager.Memory") as memory_cls,
    ):
        memory_cls.from_config.return_value = MagicMock()
        manager.initialize(
            backend_host="localhost",
            backend_port=8080,
            llm_model="test-llm",
            embedding_model="lightonai/DenseOn",
            llm_base_url=llm_base_url,
            embedder_base_url="http://denseon:8000",
            config_manager=ConfigManager(store=store),
            schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
            **kwargs,
        )
    return backend


def _llm_block(base_url: str, api_key: str) -> dict:
    return {
        "provider": "openai",
        "config": {
            "model": "test-llm",
            "temperature": 0.1,
            "openai_base_url": base_url,
            "api_key": api_key,
        },
    }


def test_modal_llm_gets_the_environment_bearer(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")
    manager = _fresh_manager()

    _initialize(manager, MODAL)

    assert manager.config["llm"] == _llm_block(MODAL, "real-bearer")


def test_in_cluster_llm_also_gets_the_bearer_when_one_exists(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")
    manager = _fresh_manager()

    _initialize(manager, IN_CLUSTER)

    assert manager.config["llm"] == _llm_block(IN_CLUSTER, "real-bearer")


def test_in_cluster_llm_without_a_bearer_keeps_the_keyless_sentinel(monkeypatch):
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)
    manager = _fresh_manager()

    _initialize(manager, IN_CLUSTER)

    assert manager.config["llm"] == _llm_block(IN_CLUSTER, "not-required")


def test_an_explicit_key_is_not_overridden(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")
    manager = _fresh_manager()

    _initialize(manager, MODAL, llm_api_key="an-explicitly-configured-key")

    assert manager.config["llm"] == _llm_block(MODAL, "an-explicitly-configured-key")


@pytest.mark.parametrize("llm_api_key", (None, PLACEHOLDER))
def test_modal_llm_without_a_bearer_fails_before_any_side_effect(
    monkeypatch, llm_api_key
):
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)
    manager = _fresh_manager()
    deployed = []
    build = Mem0MemoryManager._build_and_store_memory

    def _recording_build(self, **kwargs):
        deployed.append(kwargs["storage_tenant_id"])
        return build(self, **kwargs)

    with (
        patch.object(Mem0MemoryManager, "_build_and_store_memory", _recording_build),
        pytest.raises(
            RuntimeError,
            match="Modal inference endpoint requires COGNIVERSE_INFERENCE_API_KEY",
        ),
    ):
        _initialize(manager, MODAL, llm_api_key=llm_api_key)

    assert deployed == []
    assert manager.memory is None
    assert manager.config is None
