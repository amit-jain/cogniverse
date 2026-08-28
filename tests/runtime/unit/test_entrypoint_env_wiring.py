"""Entrypoints resolve runtime env before their first telemetry use."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cogniverse_agents.optimizer import dspy_agent_optimizer as dspy_optimizer
from cogniverse_core.common.cache.backends import s3 as s3_backend
from cogniverse_runtime import main as runtime_main
from cogniverse_runtime import optimization_cli, quality_monitor_cli
from cogniverse_runtime.ingestion_worker import worker

TELEMETRY_OTLP_ENDPOINT = "phoenix-test:4317"

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.fixture(autouse=True)
def _reset_s3_backend_defaults():
    s3_backend.configure_s3_backend_defaults(
        endpoint=None, access_key=None, secret_key=None
    )
    yield
    s3_backend.configure_s3_backend_defaults(
        endpoint=None, access_key=None, secret_key=None
    )


def _resolver_spy(events: list[str]):
    def _resolve():
        events.append("resolve")
        return {
            "minio_endpoint": "http://minio.internal:9000",
            "minio_access_key": "minio-access",
            "minio_secret_key": "minio-secret",
            "telemetry_otlp_endpoint": TELEMETRY_OTLP_ENDPOINT,
            "telemetry_http_endpoint": None,
            "semantic_embed_url": "http://embed.internal:8000",
            "semantic_embed_model": "embed-model",
            "tenant_cache_capacity": 23,
        }

    return _resolve


def _telemetry_manager_spy(events: list[str]):
    class _TelemetryManager:
        def __init__(self):
            self.config = SimpleNamespace(
                provider_config={}, otlp_endpoint=TELEMETRY_OTLP_ENDPOINT
            )

        def get_provider(self, tenant_id: str):
            events.append(f"provider:{tenant_id}")
            return SimpleNamespace(tenant_id=tenant_id)

    def _get_telemetry_manager(*args, **kwargs):
        events.append(f"telemetry:{kwargs}")
        assert kwargs == {"otlp_endpoint": TELEMETRY_OTLP_ENDPOINT}
        return _TelemetryManager()

    return _get_telemetry_manager


def _patch_shared_resolver(monkeypatch, events: list[str]):
    monkeypatch.setattr(
        "cogniverse_runtime.entrypoint_env.resolve_library_env_defaults",
        _resolver_spy(events),
    )


def test_optimization_cli_resolves_before_telemetry(monkeypatch):
    events: list[str] = []

    async def _fake_simba_optimization(**kwargs):
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        events.append("mode")
        get_telemetry_manager(otlp_endpoint=TELEMETRY_OTLP_ENDPOINT)
        return {"status": "ok"}

    _patch_shared_resolver(monkeypatch, events)
    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        _telemetry_manager_spy(events),
    )
    monkeypatch.setattr(
        optimization_cli, "run_simba_optimization", _fake_simba_optimization
    )
    monkeypatch.setattr(
        sys, "argv", ["optimization_cli.py", "--mode", "simba", "--tenant-id", "acme"]
    )

    with pytest.raises(SystemExit) as exc_info:
        optimization_cli.main()

    assert exc_info.value.code == 0
    assert events == [
        "resolve",
        "mode",
        f"telemetry:{{'otlp_endpoint': '{TELEMETRY_OTLP_ENDPOINT}'}}",
    ]


def test_optimization_cli_monthly_reports_resolves_before_telemetry(
    monkeypatch, tmp_path
):
    events: list[str] = []

    async def _fake_monthly_reports(**kwargs):
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        events.append("mode")
        assert kwargs["output_dir"] == str(tmp_path)
        get_telemetry_manager(otlp_endpoint=TELEMETRY_OTLP_ENDPOINT)
        return {"status": "ok"}

    _patch_shared_resolver(monkeypatch, events)
    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        _telemetry_manager_spy(events),
    )
    monkeypatch.setattr(optimization_cli, "run_monthly_reports", _fake_monthly_reports)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "optimization_cli.py",
            "--mode",
            "monthly-reports",
            "--reports-output-dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        optimization_cli.main()

    assert exc_info.value.code == 0
    assert events == [
        "resolve",
        "mode",
        f"telemetry:{{'otlp_endpoint': '{TELEMETRY_OTLP_ENDPOINT}'}}",
    ]


def test_quality_monitor_cli_resolves_before_telemetry(monkeypatch):
    events: list[str] = []

    class _StubMonitor:
        def __init__(self, **kwargs):
            events.append("monitor")

    async def _fake_annotation_cycle(**kwargs):
        events.append("annotation-cycle")
        return {"identified": 0, "already_annotated": 0, "enqueued": 0}

    _patch_shared_resolver(monkeypatch, events)
    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        _telemetry_manager_spy(events),
    )
    monkeypatch.setattr(
        quality_monitor_cli, "_build_phoenix_provider", lambda **k: None
    )
    monkeypatch.setattr(
        "cogniverse_evaluation.quality_monitor.QualityMonitor", _StubMonitor
    )
    monkeypatch.setattr(
        quality_monitor_cli, "run_annotation_cycle", _fake_annotation_cycle
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quality_monitor_cli.py",
            "--tenant-id",
            "acme",
            "--llm-model",
            "gemma",
            "--annotation-cycle",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        quality_monitor_cli.main()

    assert exc_info.value.code == 0
    assert events == [
        "resolve",
        f"telemetry:{{'otlp_endpoint': '{TELEMETRY_OTLP_ENDPOINT}'}}",
        "monitor",
        "annotation-cycle",
    ]


@pytest.mark.asyncio
async def test_ingestion_worker_resolves_before_telemetry(monkeypatch):
    events: list[str] = []

    class _StubConfig:
        redis_url = "redis://stub"
        inference_service_urls = {}
        consumer_group = "ingestors"
        consumer_id = "worker-1"
        idempotency_ttl = 604800
        claim_block_ms = 5000
        heartbeat_interval_s = 60
        reaper_enabled = False
        reaper_interval_s = 60
        reaper_min_idle_ms = 300000
        reaper_max_deliveries = 5
        job_deadline_s = 7200
        graph_deadline_s = 1800.0

    async def _fake_get_redis(url):
        events.append(f"redis:{url}")
        return object()

    async def _fake_close_redis():
        events.append("close")

    async def _fake_claim_loop(
        redis,
        config,
        stop,
        *,
        processor,
        telemetry_otlp_endpoint=None,
    ):
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        events.append("claim-loop")
        assert telemetry_otlp_endpoint == TELEMETRY_OTLP_ENDPOINT
        get_telemetry_manager(otlp_endpoint=TELEMETRY_OTLP_ENDPOINT)

    _patch_shared_resolver(monkeypatch, events)
    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        _telemetry_manager_spy(events),
    )
    monkeypatch.setattr(worker, "_validate_pipeline_cache_defaults", lambda: None)
    monkeypatch.setattr(worker, "WorkerConfig", lambda: _StubConfig())
    monkeypatch.setattr(worker, "get_redis", _fake_get_redis)
    monkeypatch.setattr(worker, "close_redis", _fake_close_redis)
    monkeypatch.setattr(worker, "_claim_loop", _fake_claim_loop)

    await worker.run(stop=asyncio.Event(), processor=lambda job: {"status": "ok"})

    assert events == [
        "resolve",
        "redis:redis://stub",
        "claim-loop",
        f"telemetry:{{'otlp_endpoint': '{TELEMETRY_OTLP_ENDPOINT}'}}",
        "close",
    ]


@pytest.mark.asyncio
async def test_dspy_optimizer_resolves_before_telemetry(monkeypatch):
    events: list[str] = []

    class _StubLLMConfig:
        primary = SimpleNamespace(model="stub", api_base="http://llm")

        @staticmethod
        def resolve_teacher():
            return None

    class _StubConfig:
        def get_llm_config(self):
            return _StubLLMConfig()

    class _StubPipeline:
        def __init__(self, optimizer):
            self.optimizer = optimizer

        async def optimize_all_modules(self):
            events.append("optimize")
            return []

        async def save_optimized_prompts(self, *, tenant_id, telemetry_provider):
            events.append(f"save:{telemetry_provider}")

    class _StubTelemetryManager:
        def get_provider(self, tenant_id):
            events.append(f"provider:{tenant_id}")
            return "provider"

    _patch_shared_resolver(monkeypatch, events)
    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        _telemetry_manager_spy(events),
    )
    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.create_default_config_manager",
        lambda: object(),
    )
    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.get_config",
        lambda **kwargs: _StubConfig(),
    )
    monkeypatch.setattr(
        dspy_optimizer.DSPyAgentPromptOptimizer,
        "initialize_language_model",
        lambda self, endpoint_config, teacher_endpoint_config=None: True,
    )
    monkeypatch.setattr(dspy_optimizer, "DSPyAgentOptimizerPipeline", _StubPipeline)

    await dspy_optimizer.main()

    assert events == [
        "resolve",
        f"telemetry:{{'otlp_endpoint': '{TELEMETRY_OTLP_ENDPOINT}'}}",
        "provider:__system__",
        "optimize",
        "save:namespace(tenant_id='__system__')",
    ]


def test_text_analysis_agent_main_uses_public_resolver_import():
    text = Path("libs/agents/cogniverse_agents/text_analysis_agent.py").read_text(
        encoding="utf-8"
    )
    assert (
        "from cogniverse_runtime.entrypoint_env import resolve_library_env_defaults"
        in text
    )
    assert (
        "from cogniverse_runtime.main import _resolve_library_env_defaults" not in text
    )


@pytest.mark.asyncio
async def test_worker_bootstrap_sets_exact_s3_defaults(monkeypatch):
    monkeypatch.setattr(
        "cogniverse_runtime.entrypoint_env.resolve_library_env_defaults",
        lambda: {
            "minio_endpoint": "http://minio.internal:9000",
            "minio_access_key": "minio-access",
            "minio_secret_key": "minio-secret",
            "telemetry_otlp_endpoint": TELEMETRY_OTLP_ENDPOINT,
            "telemetry_http_endpoint": None,
            "semantic_embed_url": "http://embed.internal:8000",
            "semantic_embed_model": "embed-model",
            "tenant_cache_capacity": 23,
        },
    )
    monkeypatch.setattr(worker, "_validate_pipeline_cache_defaults", lambda: None)
    monkeypatch.setattr(
        worker,
        "WorkerConfig",
        lambda: SimpleNamespace(redis_url="redis://stub"),
    )

    async def _fake_get_redis(url):
        assert url == "redis://stub"
        assert (
            s3_backend.configured_s3_backend_defaults()
            == s3_backend.S3BackendDefaults(
                endpoint="http://minio.internal:9000",
                access_key="minio-access",
                secret_key="minio-secret",
            )
        )
        assert os.environ["AWS_ACCESS_KEY_ID"] == "minio-access"
        assert os.environ["AWS_SECRET_ACCESS_KEY"] == "minio-secret"
        raise RuntimeError("stop after bootstrap")

    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setattr(worker, "get_redis", _fake_get_redis)

    with pytest.raises(RuntimeError, match="stop after bootstrap"):
        await worker.run(stop=asyncio.Event())

    assert s3_backend.configured_s3_backend_defaults() == s3_backend.S3BackendDefaults(
        endpoint="http://minio.internal:9000",
        access_key="minio-access",
        secret_key="minio-secret",
    )


def test_main_bootstrap_sets_exact_s3_defaults(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setattr(runtime_main, "get_telemetry_manager", lambda *a, **k: None)
    monkeypatch.setattr(
        runtime_main, "configure_semantic_embedder_defaults", lambda **k: None
    )
    monkeypatch.setattr(
        "cogniverse_agents.text_analysis_agent.configure_tenant_cache_capacity",
        lambda *_: None,
    )
    monkeypatch.setattr(
        "cogniverse_core.memory.manager.configure_tenant_cache_capacity",
        lambda *_: None,
    )
    monkeypatch.setattr(
        "cogniverse_core.registries.backend_registry.configure_tenant_cache_capacity",
        lambda *_: None,
    )
    monkeypatch.setattr(
        "cogniverse_foundation.registry.entry_point_registry.configure_tenant_cache_capacity",
        lambda *_: None,
    )

    runtime_main._configure_library_module_defaults(
        config_manager=object(),
        minio_endpoint="http://minio.internal:9000",
        minio_access_key="minio-access",
        minio_secret_key="minio-secret",
        telemetry_otlp_endpoint=TELEMETRY_OTLP_ENDPOINT,
        telemetry_http_endpoint=None,
        semantic_embed_url="http://embed.internal:8000",
        semantic_embed_model="embed-model",
        tenant_cache_capacity=23,
    )

    assert s3_backend.configured_s3_backend_defaults() == s3_backend.S3BackendDefaults(
        endpoint="http://minio.internal:9000",
        access_key="minio-access",
        secret_key="minio-secret",
    )
    assert os.environ["AWS_ACCESS_KEY_ID"] == "minio-access"
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == "minio-secret"


@pytest.mark.asyncio
async def test_worker_bootstrap_fails_without_minio_when_s3_cache_enabled(monkeypatch):
    monkeypatch.setattr(
        "cogniverse_runtime.entrypoint_env.resolve_library_env_defaults",
        lambda: {
            "minio_endpoint": None,
            "minio_access_key": None,
            "minio_secret_key": None,
            "telemetry_otlp_endpoint": TELEMETRY_OTLP_ENDPOINT,
            "telemetry_http_endpoint": None,
            "semantic_embed_url": None,
            "semantic_embed_model": None,
            "tenant_cache_capacity": 23,
        },
    )
    monkeypatch.setenv("REDIS_URL", "redis://stub")
    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.create_default_config_manager",
        lambda: object(),
    )
    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.get_config",
        lambda **kwargs: {
            "pipeline_cache": {
                "enabled": True,
                "backends": [
                    {
                        "backend_type": "s3",
                        "enabled": True,
                    }
                ],
            }
        },
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "S3 cache backend needs MinIO settings at startup: "
            "MINIO_ACCESS_KEY, MINIO_ENDPOINT, MINIO_SECRET_KEY"
        ),
    ):
        await worker.run(stop=asyncio.Event())
