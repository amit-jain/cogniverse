"""Runtime router configuration comes from the enable and URL environment values.

Tenant tiers remain in their per-tenant store. The serialized router config
names the free-form, classification, and vision entrypoints.
"""

import pytest

from cogniverse_core.common.cache.backends import s3 as s3_backend
from cogniverse_foundation.config.unified_config import SemanticRouterConfig
from cogniverse_runtime.entrypoint_env import (
    configure_runtime_library_defaults as _configure_runtime_library_defaults,
)
from cogniverse_runtime.main import _semantic_router_config_from_env

_SR_ENV = (
    "SEMANTIC_ROUTER_ENABLED",
    "SEMANTIC_ROUTER_URL",
    "SEMANTIC_ROUTER_TENANT_TIERS",
)


@pytest.fixture(autouse=True)
def _clear_sr_env(monkeypatch):
    for name in _SR_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _clear_s3_defaults():
    s3_backend.configure_s3_backend_defaults(
        endpoint=None, access_key=None, secret_key=None
    )
    yield
    s3_backend.configure_s3_backend_defaults(
        endpoint=None, access_key=None, secret_key=None
    )


@pytest.mark.unit
@pytest.mark.ci_fast
class TestSemanticRouterConfigFromEnv:
    def test_unset_returns_none(self):
        assert _semantic_router_config_from_env() is None

    def test_enabled_without_url_returns_none(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_ROUTER_ENABLED", "true")
        assert _semantic_router_config_from_env() is None

    def test_url_without_enable_returns_none(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_ROUTER_URL", "http://cogniverse-gateway:8801/v1")
        assert _semantic_router_config_from_env() is None

    def test_explicit_false_returns_none(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_ROUTER_ENABLED", "false")
        monkeypatch.setenv("SEMANTIC_ROUTER_URL", "http://cogniverse-gateway:8801/v1")
        assert _semantic_router_config_from_env() is None

    def test_enabled_with_url_builds_config(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_ROUTER_ENABLED", "true")
        monkeypatch.setenv("SEMANTIC_ROUTER_URL", "http://cogniverse-gateway:8801/v1")

        cfg = _semantic_router_config_from_env()

        assert isinstance(cfg, SemanticRouterConfig)
        assert cfg.enabled is True
        assert cfg.semantic_router_url == "http://cogniverse-gateway:8801/v1"
        assert cfg == SemanticRouterConfig(
            enabled=True, semantic_router_url="http://cogniverse-gateway:8801/v1"
        )

    def test_the_config_carries_no_tenant_tier_map(self, monkeypatch):
        """A tenant's tier is its own stored attribute, not deployment env."""
        monkeypatch.setenv("SEMANTIC_ROUTER_ENABLED", "1")
        monkeypatch.setenv("SEMANTIC_ROUTER_URL", "http://cogniverse-gateway:8801/v1")

        cfg = _semantic_router_config_from_env()

        assert not hasattr(cfg, "tenant_tiers")
        assert not hasattr(cfg, "default_tier")
        assert cfg.to_dict() == {
            "enabled": True,
            "semantic_router_url": "http://cogniverse-gateway:8801/v1",
            "tier_header": "x-authz-user-groups",
            "user_id_header": "x-authz-user-id",
            "routed_model": "openai/auto",
            "response_cache_ttl_seconds": 3600,
            "response_cache_max_entries": 1024,
            "classification_model": "openai/cogniverse-classification",
            "vision_model": "openai/cogniverse-vision",
        }

    def test_a_stale_tenant_tiers_env_changes_nothing(self, monkeypatch):
        """A leftover env from an older chart must not resurrect the map."""
        monkeypatch.setenv("SEMANTIC_ROUTER_ENABLED", "yes")
        monkeypatch.setenv("SEMANTIC_ROUTER_URL", "http://cogniverse-gateway:8801/v1")
        monkeypatch.setenv("SEMANTIC_ROUTER_TENANT_TIERS", '{"acme:prod": "pro"}')

        assert _semantic_router_config_from_env() == SemanticRouterConfig(
            enabled=True, semantic_router_url="http://cogniverse-gateway:8801/v1"
        )

    def test_a_malformed_stale_tenant_tiers_env_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_ROUTER_ENABLED", "yes")
        monkeypatch.setenv("SEMANTIC_ROUTER_URL", "http://cogniverse-gateway:8801/v1")
        monkeypatch.setenv("SEMANTIC_ROUTER_TENANT_TIERS", "not-json{")

        assert _semantic_router_config_from_env() == SemanticRouterConfig(
            enabled=True, semantic_router_url="http://cogniverse-gateway:8801/v1"
        )


@pytest.mark.unit
@pytest.mark.ci_fast
class TestMirrorMinioCredentialsToAws:
    """The runtime entrypoint mirrors the MINIO_* secret onto the AWS_* names
    fsspec reads, so answer-time keyframe resolution authenticates against MinIO.
    Without this, agents localize s3:// keyframes and fsspec raises
    NoCredentialsError."""

    _NAMES = (
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    )

    @pytest.fixture(autouse=True)
    def _clear(self, monkeypatch):
        from cogniverse_agents import _rlm_promotion
        from cogniverse_agents.inference import deno_check

        for name in self._NAMES:
            monkeypatch.delenv(name, raising=False)
        for module, attribute in (
            (s3_backend, "_CONFIGURED_ENDPOINT"),
            (s3_backend, "_CONFIGURED_ACCESS_KEY"),
            (s3_backend, "_CONFIGURED_SECRET_KEY"),
            (_rlm_promotion, "_promotion_enabled"),
            (_rlm_promotion, "_promotion_fraction"),
            (deno_check, "_skip_deno_check"),
        ):
            monkeypatch.setattr(module, attribute, getattr(module, attribute))

    @staticmethod
    def _runtime_defaults(access_key, secret_key):
        return {
            "minio_endpoint": None,
            "minio_access_key": access_key,
            "minio_secret_key": secret_key,
            "telemetry_otlp_endpoint": None,
            "telemetry_http_endpoint": None,
            "semantic_embed_url": None,
            "semantic_embed_model": None,
            "tenant_cache_capacity": 1,
            "rlm_promotion_enabled": True,
            "rlm_promotion_fraction": 0.75,
            "rlm_skip_deno_check": True,
        }

    def test_mirrors_minio_secret_onto_aws_names(self, monkeypatch):
        monkeypatch.setenv("MINIO_ACCESS_KEY", "minio-access")
        monkeypatch.setenv("MINIO_SECRET_KEY", "minio-secret")
        _configure_runtime_library_defaults(
            self._runtime_defaults("minio-access", "minio-secret")
        )
        import os

        assert os.environ["AWS_ACCESS_KEY_ID"] == "minio-access"
        assert os.environ["AWS_SECRET_ACCESS_KEY"] == "minio-secret"

    def test_does_not_overwrite_explicit_aws_creds(self, monkeypatch):
        monkeypatch.setenv("MINIO_ACCESS_KEY", "minio-access")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "explicit-aws")
        _configure_runtime_library_defaults(
            self._runtime_defaults("minio-access", None)
        )
        import os

        assert os.environ["AWS_ACCESS_KEY_ID"] == "explicit-aws"

    def test_no_minio_creds_leaves_aws_unset(self, monkeypatch):
        _configure_runtime_library_defaults(self._runtime_defaults(None, None))
        import os

        assert "AWS_ACCESS_KEY_ID" not in os.environ
        assert "AWS_SECRET_ACCESS_KEY" not in os.environ
