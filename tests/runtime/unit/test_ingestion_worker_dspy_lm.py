"""Worker default-LM construction must go through create_dspy_lm.

The worker previously built ``dspy.LM(...)`` raw from LLM_ENDPOINT/LLM_MODEL
env, so the fallback LM (ClaimExtractor when no per-tenant config resolves)
never got retries/timeout/seed/extra_headers and ignored the config store's
``llm_config.primary`` entirely. These tests pin the resolution order:
config-store primary first, env fallback second, and both paths through the
``create_dspy_lm`` factory — plus how the resolved LM reaches the job.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_runtime.ingestion_worker.worker import _worker_dspy_lm

PRIMARY = {
    "model": "hosted_vllm/Qwen/Qwen2.5-7B-Instruct",
    "api_base": "http://vllm:8000/v1",
    "temperature": 0.2,
    "max_tokens": 512,
    "num_retries": 4,
    "request_timeout": 33.0,
    "seed": 7,
    "extra_headers": {"x-vsr-task": "kg"},
}


@pytest.fixture(autouse=True)
def _fresh_worker_lm(monkeypatch):
    """The resolved LM is cached for the life of the process; give every test
    an unresolved cache so it exercises the resolve it is about."""
    from cogniverse_runtime.ingestion_worker import worker

    monkeypatch.setattr(worker, "_WORKER_LM", None)
    monkeypatch.setattr(worker, "_WORKER_LM_RESOLVED", False)


@pytest.fixture
def factory_capture(monkeypatch):
    """Capture the LLMEndpointConfig handed to create_dspy_lm, without
    constructing a real dspy.LM."""
    captured = {}
    fake_lm = MagicMock(name="fake_lm")

    def _create(config):
        captured["config"] = config
        return fake_lm

    monkeypatch.setattr(
        "cogniverse_foundation.config.llm_factory.create_dspy_lm", _create
    )
    captured["fake_lm"] = fake_lm
    return captured


def _config_json(tmp_path, monkeypatch, payload):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(path))
    return path


@pytest.mark.unit
class TestWorkerDspyLmResolution:
    def test_config_store_primary_reaches_factory(
        self, tmp_path, monkeypatch, factory_capture
    ):
        _config_json(tmp_path, monkeypatch, {"llm_config": {"primary": PRIMARY}})
        monkeypatch.delenv("LLM_ENDPOINT", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)

        lm = _worker_dspy_lm(MagicMock(name="config_manager"))

        assert factory_capture["config"] == LLMEndpointConfig(**PRIMARY)
        assert lm is factory_capture["fake_lm"]

    def test_config_store_primary_wins_over_env(
        self, tmp_path, monkeypatch, factory_capture
    ):
        _config_json(tmp_path, monkeypatch, {"llm_config": {"primary": PRIMARY}})
        monkeypatch.setenv("LLM_ENDPOINT", "http://other:9999/v1")
        monkeypatch.setenv("LLM_MODEL", "gemma3:4b")

        _worker_dspy_lm(MagicMock(name="config_manager"))

        assert factory_capture["config"] == LLMEndpointConfig(**PRIMARY)

    def test_env_fallback_constructs_through_factory(
        self, tmp_path, monkeypatch, factory_capture
    ):
        _config_json(tmp_path, monkeypatch, {})
        monkeypatch.setenv("LLM_ENDPOINT", "http://vllm:8000/v1/")
        monkeypatch.setenv("LLM_MODEL", "gemma3:4b")

        lm = _worker_dspy_lm(MagicMock(name="config_manager"))

        assert factory_capture["config"] == LLMEndpointConfig(
            model="openai/gemma3:4b",
            api_base="http://vllm:8000/v1",
            temperature=0.0,
        )
        assert lm is factory_capture["fake_lm"]

    def test_env_fallback_keeps_existing_provider_prefix(
        self, tmp_path, monkeypatch, factory_capture
    ):
        _config_json(tmp_path, monkeypatch, {})
        monkeypatch.setenv("LLM_ENDPOINT", "http://ollama:11434")
        monkeypatch.setenv("LLM_MODEL", "ollama_chat/llama3")

        _worker_dspy_lm(MagicMock(name="config_manager"))

        assert factory_capture["config"] == LLMEndpointConfig(
            model="ollama_chat/llama3",
            api_base="http://ollama:11434",
            temperature=0.0,
        )

    def test_unreachable_config_store_falls_back_to_env(
        self, tmp_path, monkeypatch, factory_capture
    ):
        _config_json(tmp_path, monkeypatch, {"llm_config": {"primary": PRIMARY}})
        monkeypatch.setenv("LLM_ENDPOINT", "http://vllm:8000/v1")
        monkeypatch.setenv("LLM_MODEL", "gemma3:4b")

        broken = MagicMock(name="config_manager")
        broken.get_system_config.side_effect = RuntimeError("config store down")

        _worker_dspy_lm(broken)

        assert factory_capture["config"] == LLMEndpointConfig(
            model="openai/gemma3:4b",
            api_base="http://vllm:8000/v1",
            temperature=0.0,
        )

    def test_nothing_available_resolves_to_no_lm(
        self, tmp_path, monkeypatch, factory_capture, caplog
    ):
        _config_json(tmp_path, monkeypatch, {})
        monkeypatch.delenv("LLM_ENDPOINT", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)

        with caplog.at_level("WARNING"):
            lm = _worker_dspy_lm(MagicMock(name="config_manager"))

        assert lm is None
        assert "config" not in factory_capture
        assert "No LM is loaded" in caplog.text

    def test_concurrent_first_touches_build_one_lm(self, tmp_path, monkeypatch):
        """Two jobs whose first LM touch overlaps share one cold build — the
        resolve is a config-store read and the LM is process-wide, so the
        second caller must wait for the first and reuse what it built."""
        import threading

        from cogniverse_runtime.ingestion_worker import worker

        _config_json(tmp_path, monkeypatch, {"llm_config": {"primary": PRIMARY}})

        resolve_entered = threading.Event()
        finish_resolve = threading.Event()
        resolves = []

        def _slow_resolve(config_manager):
            resolves.append(config_manager)
            resolve_entered.set()
            # Stay inside the build (holding the lock) until the second caller
            # is in flight, so the two first-touches genuinely overlap.
            assert finish_resolve.wait(timeout=10)
            return LLMEndpointConfig(**PRIMARY)

        monkeypatch.setattr(worker, "_resolve_worker_llm_config", _slow_resolve)

        first_result = []
        second_result = []
        second_started = threading.Event()

        def _first():
            first_result.append(worker._worker_dspy_lm(MagicMock(name="cm-1")))

        def _second():
            second_started.set()
            second_result.append(worker._worker_dspy_lm(MagicMock(name="cm-2")))

        first = threading.Thread(target=_first)
        first.start()
        assert resolve_entered.wait(timeout=10)

        second = threading.Thread(target=_second)
        second.start()
        assert second_started.wait(timeout=10)
        # The first build still holds the lock, so the second caller cannot
        # have produced anything yet.
        assert second_result == []

        finish_resolve.set()
        first.join(timeout=10)
        second.join(timeout=10)

        assert len(resolves) == 1
        assert first_result[0] is second_result[0]
        assert first_result[0].model == PRIMARY["model"]
        assert first_result[0].kwargs["api_base"] == PRIMARY["api_base"]


class TestRunEntrypointWiring:
    """run() must parse WorkerConfig from env, install SIGTERM/SIGINT
    handlers that set the stop event, hand the redis client + processor to
    the claim loop, and close the redis client on the way out — including
    when the claim loop dies."""

    def _set_env(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://testhost:6379/3")
        monkeypatch.setenv("INGEST_CONSUMER_GROUP", "cg-test")
        monkeypatch.setenv("INGEST_CONSUMER_ID", "worker-abc")
        monkeypatch.setenv("INGEST_IDEMPOTENCY_TTL_SECONDS", "123")
        monkeypatch.setenv("INGEST_CLAIM_BLOCK_MS", "777")
        monkeypatch.setenv(
            "INFERENCE_SERVICE_URLS",
            '{"denseon":"http://denseon:8000/"}',
        )

    def _wire(self, monkeypatch, claim_loop):
        from cogniverse_runtime.ingestion_worker import worker

        recorded = {"closed": 0}
        fake_redis = object()

        async def _fake_get_redis(url):
            recorded["get_redis_url"] = url
            return fake_redis

        async def _fake_close_redis():
            recorded["closed"] += 1

        monkeypatch.setattr(worker, "get_redis", _fake_get_redis)
        monkeypatch.setattr(worker, "close_redis", _fake_close_redis)
        monkeypatch.setattr(worker, "_claim_loop", claim_loop)
        return worker, recorded, fake_redis

    @pytest.mark.asyncio
    async def test_run_wires_config_signals_loop_and_closes_redis(self, monkeypatch):
        import asyncio
        import signal as signal_module

        self._set_env(monkeypatch)
        claim_call = {}

        async def _claim_loop(redis, config, stop, processor=None):
            claim_call.update(
                redis=redis, config=config, stop=stop, processor=processor
            )

        worker, recorded, fake_redis = self._wire(monkeypatch, _claim_loop)

        loop = asyncio.get_running_loop()
        installed = []
        monkeypatch.setattr(
            loop,
            "add_signal_handler",
            lambda sig, cb, *args: installed.append((sig, cb)),
        )

        sentinel_processor = object()
        await worker.run(processor=sentinel_processor)

        cfg = claim_call["config"]
        assert (
            cfg.redis_url,
            cfg.consumer_group,
            cfg.consumer_id,
            cfg.idempotency_ttl,
            cfg.claim_block_ms,
        ) == ("redis://testhost:6379/3", "cg-test", "worker-abc", 123, 777)
        assert cfg.inference_service_urls == {"denseon": "http://denseon:8000"}

        assert recorded["get_redis_url"] == "redis://testhost:6379/3"
        assert claim_call["redis"] is fake_redis
        assert claim_call["processor"] is sentinel_processor

        stop_event = claim_call["stop"]
        assert isinstance(stop_event, asyncio.Event)
        assert [sig for sig, _ in installed] == [
            signal_module.SIGTERM,
            signal_module.SIGINT,
        ]
        for _sig, callback in installed:
            assert callback.__self__ is stop_event
            assert callback.__name__ == "set"
        assert not stop_event.is_set()
        installed[0][1]()  # delivering SIGTERM sets the stop event
        assert stop_event.is_set()

        assert recorded["closed"] == 1

    @pytest.mark.asyncio
    async def test_run_binds_one_startup_config_to_claim_and_reaper(self, monkeypatch):
        import asyncio

        from cogniverse_runtime.ingestion_worker import reaper

        self._set_env(monkeypatch)
        default_calls = []
        claim_call = {}
        reaper_call = {}
        reaper_started = asyncio.Event()

        async def _default(job, *, config):
            default_calls.append((job, config))
            return {"status": "success"}

        async def _claim_loop(redis, config, stop, *, processor):
            claim_call.update(config=config, processor=processor)
            await processor("claim-job")
            await reaper_started.wait()

        async def _reaper_loop(redis, config, stop, *, processor):
            reaper_call.update(config=config, processor=processor)
            await processor("reaper-job")
            reaper_started.set()

        worker, recorded, _ = self._wire(monkeypatch, _claim_loop)
        monkeypatch.setattr(worker, "_default_processor", _default)
        monkeypatch.setattr(reaper, "reaper_loop", _reaper_loop)

        await worker.run(stop=asyncio.Event())

        config = claim_call["config"]
        assert reaper_call["config"] is config
        assert claim_call["processor"] is reaper_call["processor"]
        assert default_calls == [
            ("claim-job", config),
            ("reaper-job", config),
        ]
        assert recorded["closed"] == 1

    @pytest.mark.asyncio
    async def test_run_rejects_malformed_inference_urls_before_dependencies(
        self, monkeypatch
    ):
        import asyncio

        from cogniverse_foundation.config import utils
        from cogniverse_runtime.ingestion_worker import worker

        self._set_env(monkeypatch)
        monkeypatch.setenv("INFERENCE_SERVICE_URLS", "not-json")
        dependency_calls = []

        async def _get_redis(url):
            dependency_calls.append(("redis", url))
            return object()

        def _config_manager():
            dependency_calls.append(("config-manager", None))
            return object()

        async def _claim_loop(redis, config, stop, *, processor):
            dependency_calls.append(("claim", config))

        monkeypatch.setattr(worker, "get_redis", _get_redis)
        monkeypatch.setattr(worker, "_claim_loop", _claim_loop)
        monkeypatch.setattr(utils, "create_default_config_manager", _config_manager)

        with pytest.raises(
            ValueError,
            match=(
                "^INFERENCE_SERVICE_URLS must be a JSON object of "
                "root HTTP\\(S\\) URLs$"
            ),
        ):
            await worker.run(stop=asyncio.Event())

        assert dependency_calls == []

    @pytest.mark.asyncio
    async def test_run_closes_redis_when_claim_loop_dies(self, monkeypatch):
        import asyncio

        self._set_env(monkeypatch)

        async def _boom(redis, config, stop, processor=None):
            raise RuntimeError("redis stream gone")

        worker, recorded, _ = self._wire(monkeypatch, _boom)

        loop = asyncio.get_running_loop()
        installed = []
        monkeypatch.setattr(
            loop,
            "add_signal_handler",
            lambda sig, cb, *args: installed.append((sig, cb)),
        )

        with pytest.raises(RuntimeError, match="redis stream gone"):
            await worker.run(stop=asyncio.Event(), processor=object())

        # Caller-supplied stop event → no signal handlers installed.
        assert installed == []
        assert recorded["closed"] == 1

    @pytest.mark.asyncio
    async def test_run_without_redis_url_raises(self, monkeypatch):
        from cogniverse_runtime.ingestion_worker import worker

        monkeypatch.delenv("REDIS_URL", raising=False)
        with pytest.raises(
            RuntimeError, match="REDIS_URL must be set for the ingestion worker"
        ):
            await worker.run(processor=object())


@pytest.mark.unit
class TestEnsureWorkerDspyLm:
    def test_resolves_once_per_process(self, monkeypatch):
        """The worker-wide default LM is process-global: later jobs reuse the
        first job's LM instead of re-reading the config store."""
        import asyncio

        from cogniverse_runtime.ingestion_worker import worker

        resolves = 0

        def _resolve(config_manager):
            nonlocal resolves
            resolves += 1
            return LLMEndpointConfig(**PRIMARY)

        monkeypatch.setattr(worker, "_resolve_worker_llm_config", _resolve)

        first = asyncio.run(worker._ensure_worker_dspy_lm(MagicMock(name="cm")))
        second = asyncio.run(worker._ensure_worker_dspy_lm(MagicMock(name="cm")))

        assert resolves == 1
        assert first is second
        assert first.model == PRIMARY["model"]


async def _foreign_task_owns_dspy(monkeypatch):
    """Hand DSPy's ambient binding to an async task that has finished by the
    time the job runs, and return that task's LM.

    Ownership is recorded in module globals and never released, so this is the
    state the worker process is in once anything else has configured DSPy."""
    import asyncio
    from importlib import import_module

    import dspy

    # The module, not the ``settings`` singleton the package re-exports —
    # ambient ownership lives in module-level globals.
    dspy_settings = import_module("dspy.dsp.utils.settings")
    monkeypatch.setattr(dspy_settings, "config_owner_async_task", None)
    monkeypatch.setitem(dspy_settings.main_thread_config, "lm", None)

    owner_lm = dspy.LM(
        model="openai/ambient-owner",
        api_base="http://127.0.0.1:29071/v1",
        api_key="not-required",
    )

    async def _claim():
        dspy.configure(lm=owner_lm)

    await asyncio.create_task(_claim())
    return owner_lm


def _ambient_lm():
    """The process-wide LM binding, read the way ``dspy.configure`` writes it."""
    from importlib import import_module

    return import_module("dspy.dsp.utils.settings").main_thread_config["lm"]


@pytest.fixture
def job_context(monkeypatch, tmp_path):
    """Drive the real ``_default_processor`` up to the pipeline.

    The two steps it cannot run in-process are stubbed and nothing else: the
    source localize needs the object store plus a real video, and the job body
    needs Vespa + the encoder stack + the GLiNER sidecar. The LM resolution,
    the ``dspy.context`` binding and the propagation into threads and subtasks
    are the real thing.
    """
    from cogniverse_core.common import media
    from cogniverse_runtime.ingestion_worker import worker

    _config_json(tmp_path, monkeypatch, {"llm_config": {"primary": PRIMARY}})

    class _Locator:
        def __init__(self, tenant_id, config):
            pass

        def localize(self, source_url):
            return tmp_path / "video.mp4"

    monkeypatch.setattr(media, "MediaLocator", _Locator)
    monkeypatch.setattr(
        worker,
        "_prepare_job_context",
        lambda config: (MagicMock(name="config_manager"), MagicMock(name="loader")),
    )

    def _run_body(body):
        """Install ``body`` as the job's work and return a runner for it."""

        async def _ingest(job, *, local_path, config_manager, schema_loader):
            return await body()

        monkeypatch.setattr(worker, "_ingest_and_extract_graph", _ingest)
        return worker

    return _run_body


@pytest.mark.unit
class TestJobLmBinding:
    """The worker binds its default LM per job with ``dspy.context``. Writing
    the ambient binding instead (``dspy.configure``) raises for every async
    task but the one that claimed the slot first, which failed every ingest
    job in a process where anything else had configured DSPy."""

    @pytest.mark.asyncio
    async def test_job_binds_worker_lm_when_another_task_owns_dspy(
        self, monkeypatch, job_context
    ):
        import asyncio

        import dspy

        owner_lm = await _foreign_task_owns_dspy(monkeypatch)
        seen = {}

        async def _body():
            # The KG pass reads the LM from offloaded threads and gathered
            # subtasks; both must see the job's binding.
            seen["task"] = dspy.settings.lm
            seen["thread"] = await asyncio.to_thread(lambda: dspy.settings.lm)
            (seen["subtask"],) = await asyncio.gather(
                asyncio.to_thread(lambda: dspy.settings.lm)
            )
            return {"status": "success"}

        worker = job_context(_body)
        result = await worker._default_processor(
            MagicMock(tenant_id="acme:prod", source_url="s3://b/v.mp4", profile="p"),
            config=MagicMock(name="worker_config"),
        )

        assert result == {"status": "success"}
        assert seen["task"].model == PRIMARY["model"]
        assert seen["task"].kwargs["api_base"] == PRIMARY["api_base"]
        assert seen["thread"] is seen["task"]
        assert seen["subtask"] is seen["task"]
        assert seen["task"] is not owner_lm
        # The ambient binding still belongs to the task that claimed it, and
        # the job's binding is gone once the job is over.
        assert _ambient_lm() is owner_lm
        assert dspy.settings.lm is owner_lm

    @pytest.mark.asyncio
    async def test_concurrent_jobs_share_one_lm_without_touching_ambient(
        self, monkeypatch, job_context
    ):
        """Two jobs in flight at once both run under the worker's LM, and
        neither writes the ambient binding the rest of the process reads."""
        import asyncio

        import dspy

        owner_lm = await _foreign_task_owns_dspy(monkeypatch)
        both_inside = asyncio.Barrier(2)
        seen = []

        async def _body():
            # Hold both jobs inside their bindings at the same time.
            await both_inside.wait()
            seen.append(dspy.settings.lm)
            return {"status": "success"}

        worker = job_context(_body)

        def _job(name):
            return worker._default_processor(
                MagicMock(tenant_id=name, source_url="s3://b/v.mp4", profile="p"),
                config=MagicMock(name="worker_config"),
            )

        results = await asyncio.gather(_job("acme:one"), _job("acme:two"))

        assert results == [{"status": "success"}, {"status": "success"}]
        assert len(seen) == 2
        assert seen[0] is seen[1]
        assert seen[0].model == PRIMARY["model"]
        assert seen[0] is not owner_lm
        assert _ambient_lm() is owner_lm

    @pytest.mark.asyncio
    async def test_job_leaves_binding_alone_when_no_endpoint_resolves(
        self, monkeypatch, job_context, tmp_path
    ):
        """With neither a config-store primary nor LLM_ENDPOINT/LLM_MODEL, the
        job binds nothing — whatever DSPy already has stays in force rather
        than being overwritten with a null LM."""
        import dspy

        from cogniverse_runtime.ingestion_worker import worker

        _config_json(tmp_path, monkeypatch, {})
        monkeypatch.delenv("LLM_ENDPOINT", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.setattr(worker, "_resolve_worker_llm_config", lambda cm: None)

        owner_lm = await _foreign_task_owns_dspy(monkeypatch)
        seen = {}

        async def _body():
            seen["lm"] = dspy.settings.lm
            return {"status": "success"}

        worker = job_context(_body)
        result = await worker._default_processor(
            MagicMock(tenant_id="acme:prod", source_url="s3://b/v.mp4", profile="p"),
            config=MagicMock(name="worker_config"),
        )

        assert result == {"status": "success"}
        assert seen["lm"] is owner_lm
        assert _ambient_lm() is owner_lm

    @pytest.mark.asyncio
    async def test_job_failure_releases_the_binding(self, monkeypatch, job_context):
        """A job that dies mid-pipeline propagates the error and unwinds its
        binding — the next reader sees the ambient LM, not the dead job's."""
        import dspy

        owner_lm = await _foreign_task_owns_dspy(monkeypatch)
        seen = {}

        async def _body():
            seen["lm"] = dspy.settings.lm
            raise RuntimeError("vespa feed refused the connection")

        worker = job_context(_body)

        with pytest.raises(RuntimeError, match="vespa feed refused the connection"):
            await worker._default_processor(
                MagicMock(
                    tenant_id="acme:prod", source_url="s3://b/v.mp4", profile="p"
                ),
                config=MagicMock(name="worker_config"),
            )

        assert seen["lm"].model == PRIMARY["model"]
        assert dspy.settings.lm is owner_lm
        assert _ambient_lm() is owner_lm
