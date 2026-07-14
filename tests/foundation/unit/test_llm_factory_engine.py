"""Tests for create_dspy_lm.

The factory passes ``LLMEndpointConfig.model`` through to dspy.LM
verbatim — no string manipulation, no provider rewriting. The chart
(or whatever else builds the LLMEndpointConfig) is responsible for
populating ``model`` with whatever litellm-recognised string should be
sent. The factory's only job is to wire api_base / api_key /
extra_body / sampling onto the dspy.LM and emit one well-formed log
line per construction.
"""

from __future__ import annotations

import pytest

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig


class TestModelPassthrough:
    """Whatever model id the caller hands in is what dspy.LM gets."""

    @pytest.mark.parametrize(
        "model",
        [
            "qwen3:4b",
            "ollama_chat/qwen3:4b",
            "hosted_vllm/google/gemma-4-e4b-it",
            "hosted_vllm/cyankiwi/Qwen3.6-27B-AWQ-INT4",
            "openai/gpt-4o-mini",
            "anthropic/claude-3-5-sonnet-20241022",
            "Qwen/Qwen2.5-7B-Instruct",
        ],
    )
    def test_model_passes_through_unchanged(self, model):
        endpoint = LLMEndpointConfig(model=model, api_base="http://endpoint:8000/v1")
        lm = create_dspy_lm(endpoint)
        assert lm.model == model

    def test_endpoint_kwargs_are_wired_onto_dspy_lm(self):
        endpoint = LLMEndpointConfig(
            model="hosted_vllm/test/Model",
            api_base="http://endpoint:8000/v1",
            api_key="sk-test",
            temperature=0.42,
            max_tokens=2048,
            extra_body={"reasoning": "auto"},
        )
        lm = create_dspy_lm(endpoint)
        assert lm.model == "hosted_vllm/test/Model"
        assert lm.kwargs["api_base"] == "http://endpoint:8000/v1"
        assert lm.kwargs["api_key"] == "sk-test"
        assert lm.kwargs["temperature"] == 0.42
        assert lm.kwargs["max_tokens"] == 2048
        assert lm.kwargs["extra_body"] == {"reasoning": "auto"}

    def test_optional_kwargs_omitted_when_none(self):
        endpoint = LLMEndpointConfig(model="test/Model")
        lm = create_dspy_lm(endpoint)
        assert "api_base" not in lm.kwargs
        assert "api_key" not in lm.kwargs
        assert "extra_body" not in lm.kwargs
        assert "extra_headers" not in lm.kwargs

    def test_extra_headers_wired_onto_dspy_lm(self):
        # Routing metadata for an OpenAI-compatible gateway (e.g. a semantic
        # router) is carried as static HTTP headers. The factory must forward
        # the exact dict litellm will put on the wire — nothing added, dropped,
        # or reshaped.
        endpoint = LLMEndpointConfig(
            model="openai/router-auto",
            api_base="http://semantic-router-envoy:8801/v1",
            extra_headers={
                "x-authz-user-groups": "pro",
                "x-vsr-task": "query_enhancement",
            },
        )
        lm = create_dspy_lm(endpoint)
        assert lm.kwargs["extra_headers"] == {
            "x-authz-user-groups": "pro",
            "x-vsr-task": "query_enhancement",
        }

    def test_extra_headers_and_extra_body_coexist(self):
        # Both channels must survive together — headers drive gateway routing,
        # extra_body carries sampling params — with neither clobbering the other.
        endpoint = LLMEndpointConfig(
            model="openai/router-auto",
            api_base="http://envoy:8801/v1",
            extra_body={"reasoning": "auto"},
            extra_headers={"x-authz-user-groups": "free"},
            seed=7,
        )
        lm = create_dspy_lm(endpoint)
        assert lm.kwargs["extra_headers"] == {"x-authz-user-groups": "free"}
        assert lm.kwargs["extra_body"] == {"reasoning": "auto", "seed": 7}

    def test_empty_extra_headers_omitted_from_wire(self):
        # An empty dict must not become an empty header block on the request.
        endpoint = LLMEndpointConfig(
            model="openai/m", api_base="http://x:8000/v1", extra_headers={}
        )
        lm = create_dspy_lm(endpoint)
        assert "extra_headers" not in lm.kwargs

    def test_keyless_api_base_gets_placeholder_key(self):
        # Self-hosted OAI-compat endpoints ignore the key, but the OpenAI
        # client refuses to construct without one — the factory must fill
        # a placeholder or every keyless vLLM/Ollama config fails at the
        # first call with a client-side AuthenticationError.
        endpoint = LLMEndpointConfig(
            model="openai/google/gemma-4-e4b-it",
            api_base="http://127.0.0.1:29110/v1",
        )
        lm = create_dspy_lm(endpoint)
        assert lm.kwargs["api_base"] == "http://127.0.0.1:29110/v1"
        assert lm.kwargs["api_key"] == "not-required"

    def test_explicit_key_not_overridden_by_placeholder(self):
        endpoint = LLMEndpointConfig(
            model="openai/gpt-4o",
            api_base="https://api.openai.com/v1",
            api_key="sk-real",
        )
        lm = create_dspy_lm(endpoint)
        assert lm.kwargs["api_key"] == "sk-real"

    def test_empty_model_raises(self):
        endpoint = LLMEndpointConfig(model="", api_base="http://x")
        with pytest.raises(ValueError, match="model is required"):
            create_dspy_lm(endpoint)


class TestLLMEndpointConfigSerialization:
    """to_dict()/from_dict() must round-trip every field that affects behavior."""

    def test_seed_survives_round_trip(self):
        cfg = LLMEndpointConfig(model="hosted_vllm/m", temperature=0.0, seed=1234)
        assert cfg.to_dict()["seed"] == 1234
        assert LLMEndpointConfig.from_dict(cfg.to_dict()).seed == 1234

    def test_seed_omitted_when_none(self):
        cfg = LLMEndpointConfig(model="hosted_vllm/m")
        assert "seed" not in cfg.to_dict()
        assert LLMEndpointConfig.from_dict(cfg.to_dict()).seed is None

    def test_full_round_trip_preserves_behavioral_fields(self):
        cfg = LLMEndpointConfig(
            model="hosted_vllm/m",
            api_base="http://x:8000/v1",
            temperature=0.0,
            max_tokens=2048,
            extra_body={"reasoning": "auto"},
            seed=7,
        )
        rt = LLMEndpointConfig.from_dict(cfg.to_dict())
        assert (rt.model, rt.api_base, rt.temperature, rt.max_tokens) == (
            "hosted_vllm/m",
            "http://x:8000/v1",
            0.0,
            2048,
        )
        assert rt.extra_body == {"reasoning": "auto"}
        assert rt.seed == 7

    def test_extra_headers_survive_round_trip(self):
        cfg = LLMEndpointConfig(
            model="openai/router-auto",
            extra_headers={"x-authz-user-groups": "pro", "x-vsr-task": "plan"},
        )
        assert cfg.to_dict()["extra_headers"] == {
            "x-authz-user-groups": "pro",
            "x-vsr-task": "plan",
        }
        rt = LLMEndpointConfig.from_dict(cfg.to_dict())
        assert rt.extra_headers == {
            "x-authz-user-groups": "pro",
            "x-vsr-task": "plan",
        }

    def test_extra_headers_omitted_from_dict_when_none(self):
        cfg = LLMEndpointConfig(model="openai/m")
        assert "extra_headers" not in cfg.to_dict()
        assert LLMEndpointConfig.from_dict(cfg.to_dict()).extra_headers is None


class TestFastFailTimeout:
    """A down/unreachable endpoint must fail fast, not hang on litellm's
    ~600s default x dspy's default retries."""

    def test_defaults_bound_timeout_and_retries(self):
        lm = create_dspy_lm(LLMEndpointConfig(model="openai/m", api_base="http://x:1"))
        assert lm.kwargs["timeout"] == 120.0
        assert lm.num_retries == 1

    def test_config_overrides_timeout_and_retries(self):
        lm = create_dspy_lm(
            LLMEndpointConfig(
                model="openai/m",
                api_base="http://x:1",
                request_timeout=30.0,
                num_retries=3,
            )
        )
        assert lm.kwargs["timeout"] == 30.0
        assert lm.num_retries == 3

    def test_timeout_and_retries_round_trip_through_dict(self):
        rt = LLMEndpointConfig.from_dict(
            LLMEndpointConfig(
                model="openai/m", request_timeout=45.0, num_retries=2
            ).to_dict()
        )
        assert rt.request_timeout == 45.0
        assert rt.num_retries == 2


def _llm_config():
    from cogniverse_foundation.config.unified_config import LLMConfig

    return LLMConfig(
        primary=LLMEndpointConfig(
            model="hosted_vllm/org/Primary",
            api_base="http://primary:8000/v1",
            api_key="sk-real-primary-key",
            temperature=0.1,
            max_tokens=1000,
        ),
        teacher=LLMEndpointConfig(
            model="openai/gpt-teacher",
            api_base="http://teacher:9000/v1",
            temperature=0.7,
            max_tokens=2048,
        ),
        overrides={
            "summarizer_agent": {
                "model": "hosted_vllm/org/Alt",
                "temperature": 0.3,
            },
            "null_component": None,
        },
    )


class TestLLMConfigResolve:
    """resolve() merges per-component overrides on the real dataclass.

    The merge must never route through to_dict() — it masks api_key to
    "***", and a resolved endpoint's api_key goes out as the bearer token,
    so the real key must survive resolution for overridden components.
    """

    def test_override_preserves_real_api_key(self):
        resolved = _llm_config().resolve("summarizer_agent")
        assert resolved.api_key == "sk-real-primary-key"
        assert resolved.model == "hosted_vllm/org/Alt"
        assert resolved.temperature == 0.3
        assert resolved.max_tokens == 1000
        assert resolved.api_base == "http://primary:8000/v1"

    def test_override_can_set_its_own_api_key(self):
        cfg = _llm_config()
        cfg.overrides["summarizer_agent"]["api_key"] = "sk-other"
        assert cfg.resolve("summarizer_agent").api_key == "sk-other"

    def test_resolved_key_reaches_dspy_lm(self):
        lm = create_dspy_lm(_llm_config().resolve("summarizer_agent"))
        assert lm.kwargs["api_key"] == "sk-real-primary-key"
        assert lm.model == "hosted_vllm/org/Alt"

    def test_no_override_returns_isolated_copy_with_key(self):
        cfg = _llm_config()
        resolved = cfg.resolve("unknown_component")
        assert resolved.api_key == "sk-real-primary-key"
        assert resolved.model == "hosted_vllm/org/Primary"
        resolved.model = "mutated"
        assert cfg.primary.model == "hosted_vllm/org/Primary"

    def test_none_override_treated_as_primary(self):
        assert _llm_config().resolve("null_component").api_key == (
            "sk-real-primary-key"
        )

    def test_override_result_is_isolated_from_primary(self):
        cfg = _llm_config()
        resolved = cfg.resolve("summarizer_agent")
        resolved.api_key = "mutated"
        assert cfg.primary.api_key == "sk-real-primary-key"

    def test_to_dict_still_masks_api_key(self):
        assert _llm_config().primary.to_dict()["api_key"] == "***"


class TestLLMConfigResolveTeacher:
    """resolve_teacher() hands optimization the configured teacher endpoint."""

    def test_returns_teacher_endpoint_exactly(self):
        teacher = _llm_config().resolve_teacher()
        assert teacher.model == "openai/gpt-teacher"
        assert teacher.api_base == "http://teacher:9000/v1"
        assert teacher.temperature == 0.7
        assert teacher.max_tokens == 2048

    def test_returns_isolated_copy(self):
        cfg = _llm_config()
        teacher = cfg.resolve_teacher()
        teacher.model = "mutated"
        assert cfg.teacher.model == "openai/gpt-teacher"
