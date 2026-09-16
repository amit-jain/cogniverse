"""Unit tests for opt-in LLM semantic routing.

These exercise only code we own: the tenant-tier header resolver, the
``apply_semantic_routing`` transform, and config serialization. They do NOT
stand up a router — a real semantic-router/Envoy round-trip is a separate
Docker-backed integration suite. Asserting against a stubbed boundary here
would only re-prove internal wiring, so these stay honestly unit-level and
pin exact values on the objects the code produces.

Routing keys on the tenant tier only (the router classifies request content
itself); there is no per-agent task header. And a broken config store raises
rather than silently routing direct.
"""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import MagicMock

import dspy
import pytest

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.semantic_router import (
    apply_semantic_routing,
    create_routed_lm,
    ingest_lm_context_for,
    resolve_semantic_router_config,
    resolve_semantic_router_headers,
    routed_lm_context_for,
)
from cogniverse_foundation.config.unified_config import (
    ROUTER_TIERS,
    LLMConfig,
    LLMEndpointConfig,
    SemanticRouterConfig,
    SystemConfig,
)
from cogniverse_foundation.config.utils import ConfigUtils
from tests.utils.tenant_helpers import config_manager_with_tiers

DIRECT = "http://vllm-student:8101/v1"
SR_URL = "http://semantic-router-envoy:8801/v1"


def _enabled_config(**overrides) -> SemanticRouterConfig:
    base = dict(enabled=True, semantic_router_url=SR_URL)
    base.update(overrides)
    return SemanticRouterConfig(**base)


class TestResolveSemanticRouterHeaders:
    def test_returns_none_when_disabled(self):
        cfg = SemanticRouterConfig(enabled=False)
        assert resolve_semantic_router_headers(cfg, "acme:prod", "pro") is None

    def test_the_resolved_tier_becomes_the_exact_tier_header(self):
        cfg = _enabled_config()
        assert resolve_semantic_router_headers(cfg, "acme:prod", "pro") == {
            "x-authz-user-id": "acme:prod",
            "x-authz-user-groups": "pro",
        }

    def test_every_tier_in_the_vocabulary_reaches_the_header(self):
        cfg = _enabled_config()
        assert {
            tier: resolve_semantic_router_headers(cfg, "acme:prod", tier)[
                "x-authz-user-groups"
            ]
            for tier in sorted(ROUTER_TIERS)
        } == {tier: tier for tier in sorted(ROUTER_TIERS)}

    def test_custom_tier_header_name_is_honored(self):
        cfg = _enabled_config(tier_header="x-tenant-tier")
        assert resolve_semantic_router_headers(cfg, "acme:prod", "pro") == {
            "x-authz-user-id": "acme:prod",
            "x-tenant-tier": "pro",
        }


class TestApplySemanticRouting:
    def test_disabled_returns_the_same_object_untouched(self):
        cfg = SemanticRouterConfig(enabled=False)
        endpoint = LLMEndpointConfig(model="openai/m", api_base=DIRECT)
        routed = apply_semantic_routing(
            endpoint=endpoint,
            config=cfg,
            tenant_id="acme:prod",
            tier="pro",
            call_site="summarizer_agent",
        )
        assert routed is endpoint
        assert routed.api_base == DIRECT
        assert routed.extra_headers is None

    def test_enabled_rewrites_api_base_and_attaches_tier_header(self):
        cfg = _enabled_config()
        endpoint = LLMEndpointConfig(
            model="openai/some-provider-model", api_base=DIRECT
        )
        routed = apply_semantic_routing(
            endpoint=endpoint,
            config=cfg,
            tenant_id="acme:prod",
            tier="pro",
            call_site="summarizer_agent",
        )
        assert routed.api_base == SR_URL
        assert routed.extra_headers == {
            "x-authz-user-id": "acme:prod",
            "x-authz-user-groups": "pro",
        }

    def test_enabled_replaces_model_with_router_auto_alias(self):
        # The router resolves models by its own catalog names / auto alias and
        # 400s on raw provider model ids — the routed request must not carry
        # the endpoint's model.
        cfg = _enabled_config()
        endpoint = LLMEndpointConfig(
            model="openai/google/gemma-4-e4b-it", api_base=DIRECT
        )
        routed = apply_semantic_routing(
            endpoint=endpoint,
            config=cfg,
            tenant_id="acme:prod",
            tier="pro",
            call_site="summarizer_agent",
        )
        assert routed.model == "openai/auto"
        assert endpoint.model == "openai/google/gemma-4-e4b-it"

    def test_merges_onto_preexisting_extra_headers(self):
        cfg = _enabled_config()
        endpoint = LLMEndpointConfig(
            model="openai/router-auto",
            api_base=DIRECT,
            extra_headers={"x-trace-id": "abc123"},
        )
        routed = apply_semantic_routing(
            endpoint=endpoint,
            config=cfg,
            tenant_id="acme:prod",
            tier="pro",
            call_site="summarizer_agent",
        )
        assert routed.extra_headers == {
            "x-trace-id": "abc123",
            "x-authz-user-id": "acme:prod",
            "x-authz-user-groups": "pro",
        }

    def test_tier_header_wins_on_key_collision(self):
        cfg = _enabled_config()
        endpoint = LLMEndpointConfig(
            model="openai/router-auto",
            api_base=DIRECT,
            extra_headers={"x-authz-user-groups": "stale"},
        )
        routed = apply_semantic_routing(
            endpoint=endpoint,
            config=cfg,
            tenant_id="acme:prod",
            tier="pro",
            call_site="summarizer_agent",
        )
        assert routed.extra_headers["x-authz-user-groups"] == "pro"

    def test_does_not_mutate_the_input_endpoint(self):
        cfg = _enabled_config()
        endpoint = LLMEndpointConfig(model="openai/router-auto", api_base=DIRECT)
        apply_semantic_routing(
            endpoint=endpoint,
            config=cfg,
            tenant_id="acme:prod",
            tier="pro",
            call_site="summarizer_agent",
        )
        assert endpoint.api_base == DIRECT
        assert endpoint.extra_headers is None

    def test_enabled_without_semantic_router_url_raises(self):
        cfg = _enabled_config(semantic_router_url="")
        endpoint = LLMEndpointConfig(model="openai/m", api_base=DIRECT)
        with pytest.raises(ValueError, match="semantic_router_url is"):
            apply_semantic_routing(
                endpoint=endpoint,
                config=cfg,
                tenant_id="acme:prod",
                tier="pro",
                call_site="summarizer_agent",
            )

    def test_apply_then_factory_wires_semantic_router_onto_dspy_lm(self):
        # Ties the transform to the factory: the constructed dspy.LM must carry
        # exactly the semantic router api_base and the resolved tier header.
        cfg = _enabled_config()
        endpoint = LLMEndpointConfig(model="openai/router-auto", api_base=DIRECT)
        routed = apply_semantic_routing(
            endpoint=endpoint,
            config=cfg,
            tenant_id="unregistered:tenant",
            tier="default",
            call_site="summarizer_agent",
        )
        lm = create_dspy_lm(routed)
        assert lm.kwargs["api_base"] == SR_URL
        assert lm.kwargs["extra_headers"] == {
            "x-authz-user-id": "unregistered:tenant",
            "x-authz-user-groups": "default",
        }


class TestSemanticRouterConfigSerialization:
    def test_config_round_trips_exactly(self):
        cfg = _enabled_config()
        rt = SemanticRouterConfig.from_dict(cfg.to_dict())
        assert (rt.enabled, rt.semantic_router_url) == (True, SR_URL)
        assert not hasattr(rt, "tenant_tiers")
        assert not hasattr(rt, "default_tier")
        assert rt.tier_header == "x-authz-user-groups"
        assert rt.routed_model == "openai/auto"
        assert rt.classification_model == "openai/cogniverse-classification"
        assert rt.to_dict() == {
            "enabled": True,
            "semantic_router_url": SR_URL,
            "tier_header": "x-authz-user-groups",
            "user_id_header": "x-authz-user-id",
            "routed_model": "openai/auto",
            "response_cache_ttl_seconds": 3600,
            "response_cache_max_entries": 1024,
            "classification_model": "openai/cogniverse-classification",
            "vision_model": "openai/cogniverse-vision",
        }

    def test_response_cache_bounds_survive_a_round_trip(self):
        """Non-default bounds, so a from_dict that dropped them reads back as
        the defaults and fails here instead of silently widening the agent
        cache."""
        cfg = _enabled_config(
            response_cache_ttl_seconds=600, response_cache_max_entries=64
        )
        rt = SemanticRouterConfig.from_dict(cfg.to_dict())
        assert (rt.response_cache_ttl_seconds, rt.response_cache_max_entries) == (
            600,
            64,
        )

    def test_a_configured_classification_model_round_trips(self):
        cfg = SemanticRouterConfig(
            enabled=True,
            semantic_router_url=SR_URL,
            classification_model="openai/another-entrypoint",
        )
        rt = SemanticRouterConfig.from_dict(cfg.to_dict())
        assert rt.classification_model == "openai/another-entrypoint"
        assert rt == cfg

    def test_system_config_default_leaves_semantic_router_disabled(self):
        assert SystemConfig().semantic_router.enabled is False

    def test_system_config_round_trips_semantic_router(self):
        syscfg = SystemConfig(semantic_router=_enabled_config())
        rt = SystemConfig.from_dict(syscfg.to_dict())
        assert rt.semantic_router.enabled is True
        assert rt.semantic_router.semantic_router_url == SR_URL
        assert not hasattr(rt.semantic_router, "tenant_tiers")
        assert rt.semantic_router.to_dict() == _enabled_config().to_dict()


class TestConfigUtilsSemanticRouting:
    """ConfigUtils exposes SystemConfig.semantic_router to the LM-build path."""

    def _config_utils(self, system_config) -> ConfigUtils:
        manager = MagicMock()
        manager.get_system_config.return_value = system_config
        return ConfigUtils("acme:prod", config_manager=manager)

    def test_returns_the_system_config_block(self):
        router = _enabled_config()
        cu = self._config_utils(SystemConfig(semantic_router=router))
        result = cu.get_semantic_router()
        assert result is router
        assert result.enabled is True
        assert result.semantic_router_url == SR_URL

    def test_default_system_config_is_disabled(self):
        cu = self._config_utils(SystemConfig())
        assert cu.get_semantic_router().enabled is False


class TestResolveSemanticRouterConfig:
    def test_absent_accessor_returns_disabled_default(self):
        result = resolve_semantic_router_config(object())
        assert isinstance(result, SemanticRouterConfig)
        assert result.enabled is False

    def test_valid_accessor_returns_the_config(self):
        router = _enabled_config()
        accessor = MagicMock()
        accessor.get_semantic_router.return_value = router
        assert resolve_semantic_router_config(accessor) is router

    def test_mocked_accessor_value_is_rejected(self):
        # A bare MagicMock's get_semantic_router() returns a MagicMock whose
        # .enabled is truthy; the isinstance guard must reject it.
        result = resolve_semantic_router_config(MagicMock())
        assert isinstance(result, SemanticRouterConfig)
        assert result.enabled is False

    def test_raising_accessor_propagates(self):
        # No silent fallback: a broken config store must surface, not disable.
        accessor = MagicMock()
        accessor.get_semantic_router.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            resolve_semantic_router_config(accessor)


class TestCreateRoutedLM:
    def test_enabled_builds_lm_on_semantic_router_with_tier_header(self):
        endpoint = LLMEndpointConfig(model="openai/router-auto", api_base=DIRECT)
        lm = create_routed_lm(
            endpoint=endpoint,
            config=_enabled_config(),
            tenant_id="acme:prod",
            tier="pro",
            call_site="summarizer_agent",
        )
        assert lm.kwargs["api_base"] == SR_URL
        assert lm.kwargs["extra_headers"] == {
            "x-authz-user-id": "acme:prod",
            "x-authz-user-groups": "pro",
        }

    def test_disabled_builds_lm_on_direct_endpoint(self):
        endpoint = LLMEndpointConfig(model="openai/m", api_base=DIRECT)
        lm = create_routed_lm(
            endpoint=endpoint,
            config=SemanticRouterConfig(enabled=False),
            tenant_id="acme:prod",
            tier="pro",
            call_site="summarizer_agent",
        )
        assert lm.kwargs["api_base"] == DIRECT
        assert "extra_headers" not in lm.kwargs


class TestRoutedLMContextFor:
    """Per-request routing context for tenant-agnostic agents."""

    def _patch_get_config(self, monkeypatch, cfg):
        monkeypatch.setattr(
            "cogniverse_foundation.config.utils.get_config", lambda **kw: cfg
        )

    class _Accessor:
        def __init__(self, system_config, llm_config):
            self._system_config = system_config
            self._llm_config = llm_config
            self.config_manager = config_manager_with_tiers({"acme:prod": "pro"})

        def get_semantic_router(self):
            return self._system_config.semantic_router

        def get_llm_config(self):
            return self._llm_config

    def test_no_endpoint_disabled_is_nullcontext(self, monkeypatch):
        # No endpoint supplied (orchestrator case) + disabled => ambient LM.
        cfg = MagicMock()
        cfg.get_semantic_router.return_value = SemanticRouterConfig(enabled=False)
        self._patch_get_config(monkeypatch, cfg)
        ctx = routed_lm_context_for(MagicMock(), "acme:prod", "orchestrator_agent")
        assert isinstance(ctx, nullcontext)

    def test_enabled_routes_through_semantic_router(self, monkeypatch):
        cfg = MagicMock()
        cfg.get_semantic_router.return_value = _enabled_config()
        cfg.config_manager = config_manager_with_tiers({"acme:prod": "pro"})
        cfg.get_llm_config.return_value.resolve.return_value = LLMEndpointConfig(
            model="openai/s", api_base=DIRECT
        )
        self._patch_get_config(monkeypatch, cfg)
        with routed_lm_context_for(MagicMock(), "acme:prod", "query_enhancement_agent"):
            lm = dspy.settings.lm
        assert lm.kwargs["api_base"] == SR_URL
        assert lm.kwargs["extra_headers"] == {
            "x-authz-user-id": "acme:prod",
            "x-authz-user-groups": "pro",
        }

    def test_ingest_context_is_direct_while_agents_route(self, monkeypatch):
        cfg = self._Accessor(
            SystemConfig(semantic_router=_enabled_config()),
            LLMConfig(
                primary=LLMEndpointConfig(
                    model="openai/google/gemma-4-e4b-it",
                    api_base=DIRECT,
                    request_timeout=120.0,
                    num_retries=1,
                )
            ),
        )
        self._patch_get_config(monkeypatch, cfg)
        endpoint = cfg.get_llm_config().primary

        with ingest_lm_context_for(endpoint):
            ingest_lm = dspy.settings.lm
        assert ingest_lm.model == "openai/google/gemma-4-e4b-it"
        assert ingest_lm.kwargs["api_base"] == DIRECT
        assert ingest_lm.kwargs["timeout"] == 120.0
        assert ingest_lm.num_retries == 1

        with routed_lm_context_for(
            object(), "acme:prod", "summarizer_agent", endpoint=endpoint
        ):
            routed_lm = dspy.settings.lm
        assert routed_lm.model == "openai/auto"
        assert routed_lm.kwargs["api_base"] == SR_URL
        assert routed_lm.kwargs["timeout"] == 120.0
        # The routed LM spends the endpoint's one retry itself, on retryable
        # statuses only; litellm is left to retry nothing.
        assert routed_lm.call_attempts == 2
        assert routed_lm.num_retries == 0

    def test_config_error_propagates(self, monkeypatch):
        # No silent fallback: a broken config store surfaces, even with an
        # endpoint in hand — it does NOT quietly build a direct LM.
        def boom(**kw):
            raise RuntimeError("config store down")

        monkeypatch.setattr("cogniverse_foundation.config.utils.get_config", boom)
        endpoint = LLMEndpointConfig(model="openai/local", api_base=DIRECT)
        with pytest.raises(RuntimeError, match="config store down"):
            routed_lm_context_for(
                MagicMock(), "acme:prod", "summarizer_agent", endpoint=endpoint
            )

    def test_enabled_routes_the_given_endpoint(self, monkeypatch):
        # endpoint param: route the agent's own endpoint rather than
        # re-resolving from config. The routed request carries the router's
        # auto alias — the router picks the concrete model.
        cfg = MagicMock()
        cfg.get_semantic_router.return_value = _enabled_config()
        cfg.config_manager = config_manager_with_tiers({"acme:prod": "pro"})
        self._patch_get_config(monkeypatch, cfg)
        endpoint = LLMEndpointConfig(model="openai/tuned", api_base=DIRECT)
        with routed_lm_context_for(
            MagicMock(), "acme:prod", "knowledge_summarization_agent", endpoint=endpoint
        ):
            lm = dspy.settings.lm
        assert lm.model == "openai/auto"  # router's auto alias, not the raw id
        assert lm.kwargs["api_base"] == SR_URL  # routed
        cfg.get_llm_config.assert_not_called()  # endpoint used, no re-resolve

    def test_disabled_with_endpoint_builds_from_that_endpoint(self, monkeypatch):
        cfg = MagicMock()
        cfg.get_semantic_router.return_value = SemanticRouterConfig(enabled=False)
        self._patch_get_config(monkeypatch, cfg)
        endpoint = LLMEndpointConfig(model="openai/tuned", api_base=DIRECT)
        with routed_lm_context_for(
            MagicMock(),
            "acme:prod",
            "multi_document_synthesis_agent",
            endpoint=endpoint,
        ):
            lm = dspy.settings.lm
        assert lm.model == "openai/tuned"
        assert lm.kwargs["api_base"] == DIRECT  # not routed
        assert "extra_headers" not in lm.kwargs


class TestARoutedFailureIsClassifiedByWhatHappened:
    """Timeouts and transport failures are outages, whatever litellm's synthetic status.

    litellm stamps ``408`` on a timeout and ``500`` on a refused connection;
    neither is a status the router answered with, so neither may be read as
    the router refusing the request.
    """

    KW = {"tenant_id": "acme:prod", "tier": "free", "routed_model": "openai/auto"}

    def test_a_timeout_is_an_outage_not_a_router_refusal(self):
        import litellm

        from cogniverse_foundation.config.routed_lm import (
            UpstreamUnavailable,
            classify_routed_failure,
        )

        failure = classify_routed_failure(
            litellm.Timeout(message="timed out", model="auto", llm_provider="openai"),
            **self.KW,
        )

        assert type(failure) is UpstreamUnavailable
        assert failure.status == 408
        assert str(failure) == (
            "the model endpoint did not answer: tenant=acme:prod tier=free "
            "routed_model=openai/auto status=408 router_code=None"
        )

    def test_a_refused_connection_is_an_outage(self):
        import litellm

        from cogniverse_foundation.config.routed_lm import (
            UpstreamUnavailable,
            classify_routed_failure,
        )

        failure = classify_routed_failure(
            litellm.APIConnectionError(
                message="connection refused", model="auto", llm_provider="openai"
            ),
            **self.KW,
        )

        assert type(failure) is UpstreamUnavailable
        assert failure.status == 500

    def test_a_timeout_is_retried_and_a_refusal_is_not(self):
        from cogniverse_foundation.config.routed_lm import (
            RoutedLM,
            RouterDecodeFailed,
            UpstreamAuthRejected,
            UpstreamRateLimited,
            UpstreamUnavailable,
        )

        lm = RoutedLM(
            "openai/auto",
            tenant_id="acme:prod",
            tier="free",
            api_base="http://127.0.0.1:29071/v1",
            api_key="unused",
            num_retries=1,
        )
        assert lm.call_attempts == 2

        def failure(kind, status):
            return kind("s", status=status, router_code=None, **self.KW)

        assert lm._retryable(failure(UpstreamUnavailable, 408), 1) is True
        assert lm._retryable(failure(UpstreamUnavailable, 503), 1) is True
        assert lm._retryable(failure(UpstreamRateLimited, 429), 1) is True
        assert lm._retryable(failure(UpstreamAuthRejected, 401), 1) is False
        assert lm._retryable(failure(RouterDecodeFailed, 400), 1) is False
        # The last allowed attempt is never followed by another.
        assert lm._retryable(failure(UpstreamUnavailable, 503), 2) is False

    def test_a_routed_lm_on_a_dead_port_raises_an_outage(self):
        import openai

        from cogniverse_foundation.config.routed_lm import (
            RoutedLM,
            UpstreamUnavailable,
        )

        # A port the test suite guarantees nothing listens on.
        lm = RoutedLM(
            "openai/auto",
            tenant_id="acme:prod",
            tier="free",
            api_base="http://127.0.0.1:29071/v1",
            api_key="unused",
            cache=False,
            num_retries=0,
            timeout=3,
        )

        with pytest.raises(UpstreamUnavailable) as excinfo:
            lm("hello")

        assert excinfo.value.status == 500
        assert excinfo.value.tenant_id == "acme:prod"
        assert excinfo.value.routed_model == "openai/auto"
        assert isinstance(excinfo.value.__cause__, openai.APIError)
        assert type(excinfo.value.__cause__).__name__ == "InternalServerError"


class TestRecordServedModel:
    """Outside any span the served model has nowhere to go; the call must
    still succeed, and nothing else may be touched."""

    def test_no_active_span_records_nothing_and_does_not_raise(self):
        from opentelemetry import trace

        from cogniverse_foundation.config.semantic_router import record_served_model

        assert not trace.get_current_span().get_span_context().is_valid
        record_served_model({"model": "Qwen/Qwen3-14B-AWQ"})

    def test_an_active_span_receives_the_completion_model(self):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from cogniverse_foundation.config.semantic_router import record_served_model
        from cogniverse_foundation.telemetry.span_contract import (
            LLM_SERVED_MODEL_ATTRIBUTE,
        )

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        with provider.get_tracer("t").start_as_current_span("agent.call"):
            record_served_model({"model": "Qwen/Qwen3-14B-AWQ"})
        (span,) = exporter.get_finished_spans()
        assert dict(span.attributes) == {
            LLM_SERVED_MODEL_ATTRIBUTE: "Qwen/Qwen3-14B-AWQ"
        }


class _EchoingChatEndpoint:
    """A chat-completions endpoint whose reply's ``model`` echoes the request's,
    or that refuses every request with the status asked for."""

    def __init__(self, status: int = 200):
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        wanted = status

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if wanted != 200:
                    payload = json.dumps({"error": {"message": "refused"}}).encode()
                else:
                    payload = json.dumps(
                        {
                            "id": "stub",
                            "object": "chat.completion",
                            "created": 0,
                            "model": body["model"],
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": "ok"},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "total_tokens": 2,
                            },
                        }
                    ).encode()
                self.send_response(wanted)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                # This endpoint owns every connection it serves: closing each
                # one keeps it out of the LM client's shared pool, so a later
                # call cannot be answered on a socket this server is closing.
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)

        class Server(ThreadingHTTPServer):
            # Non-daemon handlers are tracked and joined by server_close().
            daemon_threads = False
            # Every connection the tests open at once is accepted: the
            # socketserver default of 5 is smaller than the concurrency case,
            # and connections past the accept queue are reset.
            request_queue_size = 128

        self._server = Server(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)

    @property
    def api_base(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"


def _routed_lm_to(api_base: str, model: str):
    from cogniverse_foundation.config.semantic_router import create_routed_lm
    from cogniverse_foundation.config.unified_config import LLMEndpointConfig

    lm = create_routed_lm(
        LLMEndpointConfig(
            model="openai/auto",
            api_base="http://unused:1/v1",
            api_key="stub-key",
            temperature=0.0,
            max_tokens=8,
            num_retries=0,
        ),
        SemanticRouterConfig(
            enabled=True, semantic_router_url=api_base, routed_model=model
        ),
        "tenant-a:prod",
        "default",
        call_site="summarizer_agent",
    )
    lm.cache = False
    return lm


class TestRoutedLmUnderFaultAndConcurrency:
    def _tracing(self):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        return exporter, provider.get_tracer("routed-lm-test")

    def test_a_refused_call_raises_and_stamps_nothing(self):
        """A backend failure propagates as the error it is; the span never
        claims a served model for a call that was not served."""
        from cogniverse_foundation.telemetry.span_contract import (
            LLM_SERVED_MODEL_ATTRIBUTE,
        )

        exporter, tracer = self._tracing()
        with _EchoingChatEndpoint(status=503) as endpoint:
            lm = _routed_lm_to(endpoint.api_base, "openai/served-a")
            with tracer.start_as_current_span("agent.call"):
                with pytest.raises(Exception) as raised:
                    lm("hello")
        assert "503" in str(raised.value) or "refused" in str(raised.value)
        (span,) = exporter.get_finished_spans()
        assert LLM_SERVED_MODEL_ATTRIBUTE not in span.attributes

    def test_concurrent_calls_each_stamp_their_own_span(self):
        """Eight calls under eight spans, released together: every span carries
        the model its own call was answered with and no other's."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from cogniverse_foundation.telemetry.span_contract import (
            LLM_SERVED_MODEL_ATTRIBUTE,
        )

        exporter, tracer = self._tracing()
        barrier = threading.Barrier(8)

        def call(index: int) -> None:
            lm = _routed_lm_to(endpoint.api_base, f"openai/served-{index}")
            with tracer.start_as_current_span(f"agent.call.{index}"):
                barrier.wait(timeout=30)
                lm(f"hello {index}")

        with _EchoingChatEndpoint() as endpoint:
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(call, range(8)))
        stamped = {
            span.name: span.attributes[LLM_SERVED_MODEL_ATTRIBUTE]
            for span in exporter.get_finished_spans()
        }
        assert stamped == {f"agent.call.{i}": f"served-{i}" for i in range(8)}


class TestTheVisionEntry:
    """A call carrying image parts cannot be served by a text-only model, so
    the routed LM sends it on the vision entry (whose recipe serves the
    multimodal student for every tier) and text-only calls on the entry the
    call site takes. The choice is per request, read off the messages."""

    def test_vision_model_defaults_and_round_trips(self):
        default = SemanticRouterConfig()
        assert default.vision_model == "openai/cogniverse-vision"
        assert default.to_dict()["vision_model"] == "openai/cogniverse-vision"
        rt = SemanticRouterConfig.from_dict(
            SemanticRouterConfig(vision_model="openai/another-vision").to_dict()
        )
        assert rt.vision_model == "openai/another-vision"
        assert SemanticRouterConfig.from_dict({}).vision_model == (
            "openai/cogniverse-vision"
        )

    def test_image_bearing_calls_take_the_vision_entry_and_text_calls_do_not(self):
        from cogniverse_foundation.config.semantic_router import create_routed_lm
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        with _EchoingChatEndpoint() as endpoint:
            lm = create_routed_lm(
                LLMEndpointConfig(
                    model="openai/auto",
                    api_base="http://unused:1/v1",
                    api_key="stub-key",
                    temperature=0.0,
                    max_tokens=8,
                    num_retries=0,
                ),
                SemanticRouterConfig(
                    enabled=True,
                    semantic_router_url=endpoint.api_base,
                    routed_model="openai/text-entry",
                    vision_model="openai/vision-entry",
                ),
                "tenant-a:prod",
                "pro",
                call_site="summarizer_agent",
            )
            lm.cache = False
            image_part = {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
            }
            served = [
                lm.history[-1]["response"].model
                for messages in (
                    [{"role": "user", "content": "text only"}],
                    [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "look"}, image_part],
                        }
                    ],
                    [{"role": "user", "content": "text again"}],
                )
                if lm(messages=messages) is not None
            ]
        assert served == ["text-entry", "vision-entry", "text-entry"]
        assert lm.model == "openai/text-entry"


class TestStudentRetryRequiresAnOutageType:
    def test_an_auth_rejection_with_an_outage_status_cannot_use_the_student(self):
        from cogniverse_foundation.config.routed_lm import (
            RoutedLM,
            UpstreamAuthRejected,
        )

        lm = RoutedLM(
            "openai/auto",
            tenant_id="acme:prod",
            tier="pro",
            student_model="openai/cogniverse-classification",
            num_retries=0,
        )
        failure = UpstreamAuthRejected(
            "503 timeout no healthy upstream",
            status=503,
            router_code=None,
            tenant_id="acme:prod",
            tier="pro",
            routed_model="openai/auto",
        )
        assert lm._can_use_student(failure) is False

    def test_an_outage_type_with_a_permanent_status_cannot_use_the_student(self):
        from cogniverse_foundation.config.routed_lm import RoutedLM, UpstreamUnavailable

        lm = RoutedLM(
            "openai/auto",
            tenant_id="acme:prod",
            tier="pro",
            student_model="openai/cogniverse-classification",
            num_retries=0,
        )
        failure = UpstreamUnavailable(
            "503 timeout no healthy upstream",
            status=401,
            router_code=None,
            tenant_id="acme:prod",
            tier="pro",
            routed_model="openai/auto",
        )
        assert lm._can_use_student(failure) is False

    def test_a_local_bug_with_outage_words_keeps_its_original_type(self):
        from cogniverse_foundation.config.routed_lm import classify_routed_failure

        failure = classify_routed_failure(
            RuntimeError("503 timeout no healthy upstream"),
            tenant_id="acme:prod",
            tier="pro",
            routed_model="openai/auto",
        )
        assert failure is None
