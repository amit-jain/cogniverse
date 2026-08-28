"""Shipped audio profile service bindings resolve through runtime wiring."""

from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path

import pytest

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_runtime.ingestion.processor_manager import ProcessorManager
from cogniverse_runtime.ingestion.processors.embedding_generator import (
    embedding_generator_factory as egf,
)
from cogniverse_runtime.ingestion.processors.embedding_generator.embedding_generator_factory import (
    create_embedding_generator,
)
from cogniverse_runtime.ingestion.strategy_factory import StrategyFactory
from tests.utils.memory_store import InMemoryConfigStore

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATHS = [
    REPO_ROOT / "configs" / "config.json",
    REPO_ROOT / "charts" / "cogniverse" / "files" / "config.json",
]


def _load_config(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    if "{{" in raw:
        raw = re.sub(r"\{\{[^}]*\}\}", "http://rendered.invalid", raw)
    return json.loads(raw)


def _service_urls_from_config(config: dict) -> dict[str, str]:
    service_urls: dict[str, str] = {}
    for profile in config["backend"]["profiles"].values():
        for service_name in (profile.get("inference_services") or {}).values():
            if isinstance(service_name, str) and service_name:
                service_urls.setdefault(
                    service_name, f"http://{service_name}.example.test:8000"
                )
    return service_urls


def _config_manager(service_urls: dict[str, str]) -> ConfigManager:
    config_manager = ConfigManager(store=InMemoryConfigStore())
    config_manager.set_system_config(SystemConfig(inference_service_urls=service_urls))
    return config_manager


def _strategy_requirements(profile_config: dict) -> dict[str, dict]:
    strategy_set = StrategyFactory.create_from_profile_config(
        copy.deepcopy(profile_config)
    )
    requirements: dict[str, dict] = {}
    for strategy in strategy_set.get_all_strategies():
        requirements.update(strategy.get_required_processors())
    return requirements


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda path: path.name)
def test_audio_profile_declares_and_resolves_clap_embed(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _load_config(config_path)
    profiles = config["backend"]["profiles"]
    audio_profile = profiles["audio_clap_semantic"]
    assert audio_profile["inference_services"] == {
        "embedding": "colbert_pylate",
        "transcription": "vllm_asr",
        "acoustic_embedding": "clap_embed",
    }

    service_urls = _service_urls_from_config(config)
    manager = ProcessorManager(
        logging.getLogger("test_audio_profile_inference_services"),
        plugin_dir=Path("/does/not/exist"),
    )

    monkeypatch.setattr(
        egf.EmbeddingGeneratorFactory,
        "create",
        lambda *args, **kwargs: kwargs["profile_config"],
    )

    for profile_name, profile_config in profiles.items():
        requirements = _strategy_requirements(profile_config)
        declared_services = {
            processor_name: processor_config["inference_service"]
            for processor_name, processor_config in requirements.items()
            if "inference_service" in processor_config
        }

        manager._resolve_service_urls(requirements, service_urls)

        for processor_name, service_name in declared_services.items():
            assert (
                requirements[processor_name]["endpoint"] == service_urls[service_name]
            )
            assert "inference_service" not in requirements[processor_name]

        if profile_name != "audio_clap_semantic":
            continue

        resolved_profile = create_embedding_generator(
            config={
                "backend": {"profiles": {profile_name: copy.deepcopy(profile_config)}}
            },
            schema_name=profile_name,
            tenant_id="tenant:test",
            logger=logging.getLogger("test_audio_profile_inference_services"),
            config_manager=_config_manager(service_urls),
        )

        assert (
            resolved_profile["remote_inference_url"] == service_urls["colbert_pylate"]
        )
        assert resolved_profile["clap_endpoint_url"] == service_urls["clap_embed"]


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda path: path.name)
def test_audio_profile_missing_clap_embed_url_raises_profile_and_service(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _load_config(config_path)
    audio_profile = copy.deepcopy(config["backend"]["profiles"]["audio_clap_semantic"])
    audio_profile.setdefault("inference_services", {})["acoustic_embedding"] = (
        "clap_embed"
    )

    service_urls = {
        service_name: f"http://{service_name}.example.test:8000"
        for service_name in (audio_profile.get("inference_services") or {}).values()
        if service_name != "clap_embed"
    }
    expected = (
        "Profile 'audio_clap_semantic' specifies "
        "inference_services.acoustic_embedding='clap_embed' but no URL is configured. "
        f"Deployed services: {sorted(service_urls)}."
    )

    monkeypatch.setattr(
        egf.EmbeddingGeneratorFactory,
        "create",
        lambda *args, **kwargs: kwargs["profile_config"],
    )

    with pytest.raises(ValueError) as excinfo:
        create_embedding_generator(
            config={"backend": {"profiles": {"audio_clap_semantic": audio_profile}}},
            schema_name="audio_clap_semantic",
            tenant_id="tenant:test",
            logger=logging.getLogger("test_audio_profile_inference_services"),
            config_manager=_config_manager(service_urls),
        )

    assert str(excinfo.value) == expected
