from __future__ import annotations

from types import SimpleNamespace

import pytest

from cogniverse_runtime.ingestion_worker import worker


class _ConfigManager:
    def __init__(self, urls: dict[str, str]) -> None:
        self.system_config = SimpleNamespace(inference_service_urls=urls)

    def get_system_config(self):
        return self.system_config


def _install_context_dependencies(monkeypatch, urls: dict[str, str]):
    from cogniverse_core.schemas import filesystem_loader
    from cogniverse_foundation.config import utils

    manager = _ConfigManager(urls)
    schema_loader = object()
    graph_factory_calls = []
    monkeypatch.setattr(utils, "create_default_config_manager", lambda: manager)
    monkeypatch.setattr(
        filesystem_loader,
        "FilesystemSchemaLoader",
        lambda _: schema_loader,
    )
    monkeypatch.setattr(
        worker,
        "_ensure_graph_manager_factory",
        lambda config_manager, loader: graph_factory_calls.append(
            (config_manager, loader)
        ),
    )
    return manager, schema_loader, graph_factory_calls


@pytest.mark.unit
@pytest.mark.ci_fast
def test_worker_installs_exact_inference_urls_before_building_graph_context(
    monkeypatch,
):
    parse_calls = []

    def _parse(raw):
        parse_calls.append(raw)
        return {
            "denseon": "http://denseon:8000",
            "vllm_asr": "https://asr.modal.run",
        }

    monkeypatch.setattr(worker, "parse_inference_service_urls", _parse)
    monkeypatch.setenv("REDIS_URL", "redis://worker:6379/0")
    monkeypatch.setenv(
        "INFERENCE_SERVICE_URLS",
        '{"denseon":"http://denseon:8000","vllm_asr":"https://asr.modal.run/"}',
    )
    config = worker.WorkerConfig()
    monkeypatch.setenv(
        "INFERENCE_SERVICE_URLS",
        '{"denseon":"http://changed-after-startup:9000"}',
    )
    manager, schema_loader, graph_factory_calls = _install_context_dependencies(
        monkeypatch,
        {"stale": "http://stale:8000"},
    )

    actual_manager, actual_loader = worker._prepare_job_context(
        config.inference_service_urls
    )

    assert actual_manager is manager
    assert actual_loader is schema_loader
    assert parse_calls == [
        '{"denseon":"http://denseon:8000","vllm_asr":"https://asr.modal.run/"}'
    ]
    assert config.inference_service_urls == {
        "denseon": "http://denseon:8000",
        "vllm_asr": "https://asr.modal.run",
    }
    assert manager.system_config.inference_service_urls == {
        "denseon": "http://denseon:8000",
        "vllm_asr": "https://asr.modal.run",
    }
    assert (
        manager.system_config.inference_service_urls
        is not config.inference_service_urls
    )
    assert graph_factory_calls == [(manager, schema_loader)]


@pytest.mark.unit
@pytest.mark.ci_fast
def test_worker_preserves_persisted_inference_urls_when_environment_is_absent(
    monkeypatch,
):
    monkeypatch.setenv("REDIS_URL", "redis://worker:6379/0")
    monkeypatch.delenv("INFERENCE_SERVICE_URLS", raising=False)
    config = worker.WorkerConfig()
    monkeypatch.setenv(
        "INFERENCE_SERVICE_URLS",
        '{"lateon":"http://late-change:8000"}',
    )
    manager, schema_loader, graph_factory_calls = _install_context_dependencies(
        monkeypatch,
        {"persisted": "http://persisted:8000"},
    )

    actual_manager, actual_loader = worker._prepare_job_context(
        config.inference_service_urls
    )

    assert actual_manager is manager
    assert actual_loader is schema_loader
    assert config.inference_service_urls is None
    assert manager.system_config.inference_service_urls == {
        "persisted": "http://persisted:8000"
    }
    assert graph_factory_calls == [(manager, schema_loader)]


@pytest.mark.unit
@pytest.mark.ci_fast
def test_worker_accepts_pathed_endpoint_urls_exactly(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://worker:6379/0")
    monkeypatch.setenv(
        "INFERENCE_SERVICE_URLS",
        '{"denseon":"http://denseon:8000/v1"}',
    )

    config = worker.WorkerConfig()

    assert config.inference_service_urls == {"denseon": "http://denseon:8000/v1"}


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("not-json", r"^INFERENCE_SERVICE_URLS must be a valid JSON object$"),
        ("[]", r"^INFERENCE_SERVICE_URLS must be a JSON object$"),
        (
            '{"denseon":7}',
            r"^INFERENCE_SERVICE_URLS\['denseon'\] URL must be a string$",
        ),
        (
            '{"denseon":" http://denseon:8000"}',
            r"^INFERENCE_SERVICE_URLS\['denseon'\] URL must not contain whitespace$",
        ),
        (
            '{"denseon":"http://first:8000","denseon":"http://second:8000"}',
            r"^duplicate service name 'denseon'$",
        ),
    ],
)
def test_worker_rejects_malformed_inference_urls_before_graph_setup(
    monkeypatch,
    raw,
    message,
):
    context_calls = []
    monkeypatch.setenv("REDIS_URL", "redis://worker:6379/0")
    monkeypatch.setenv("INFERENCE_SERVICE_URLS", raw)
    monkeypatch.setattr(
        worker,
        "_prepare_job_context",
        lambda config: context_calls.append(config),
    )

    with pytest.raises(ValueError, match=message):
        worker.WorkerConfig()

    assert context_calls == []
