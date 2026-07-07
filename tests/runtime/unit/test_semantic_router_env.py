"""Unit tests for the runtime's semantic-router env-var wiring.

``_semantic_router_config_from_env`` is what makes a DEPLOYED runtime boot
with semantic routing on: the chart sets SEMANTIC_ROUTER_ENABLED +
SEMANTIC_ROUTER_URL (+ an optional JSON tenant-tier map) and this builds the
``SemanticRouterConfig`` that gets persisted into SystemConfig. These pin the
exact config produced for each env shape, including the fail-loud path (a
malformed tier map raises rather than silently emptying).
"""

import pytest

from cogniverse_foundation.config.unified_config import SemanticRouterConfig
from cogniverse_runtime.main import (
    _mirror_minio_credentials_to_aws,
    _semantic_router_config_from_env,
)

_SR_ENV = (
    "SEMANTIC_ROUTER_ENABLED",
    "SEMANTIC_ROUTER_URL",
    "SEMANTIC_ROUTER_TENANT_TIERS",
)


@pytest.fixture(autouse=True)
def _clear_sr_env(monkeypatch):
    for name in _SR_ENV:
        monkeypatch.delenv(name, raising=False)


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
        assert cfg.tenant_tiers == {}

    def test_tenant_tiers_parsed(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_ROUTER_ENABLED", "1")
        monkeypatch.setenv("SEMANTIC_ROUTER_URL", "http://cogniverse-gateway:8801/v1")
        monkeypatch.setenv(
            "SEMANTIC_ROUTER_TENANT_TIERS", '{"acme:prod": "pro", "beta:dev": "free"}'
        )

        cfg = _semantic_router_config_from_env()

        assert cfg.tenant_tiers == {"acme:prod": "pro", "beta:dev": "free"}

    def test_malformed_tier_json_raises(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_ROUTER_ENABLED", "yes")
        monkeypatch.setenv("SEMANTIC_ROUTER_URL", "http://cogniverse-gateway:8801/v1")
        monkeypatch.setenv("SEMANTIC_ROUTER_TENANT_TIERS", "not-json{")

        with pytest.raises(ValueError):
            _semantic_router_config_from_env()

    def test_non_object_tier_json_raises(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_ROUTER_ENABLED", "yes")
        monkeypatch.setenv("SEMANTIC_ROUTER_URL", "http://cogniverse-gateway:8801/v1")
        monkeypatch.setenv("SEMANTIC_ROUTER_TENANT_TIERS", "[1, 2, 3]")

        with pytest.raises(ValueError):
            _semantic_router_config_from_env()


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
        for name in self._NAMES:
            monkeypatch.delenv(name, raising=False)

    def test_mirrors_minio_secret_onto_aws_names(self, monkeypatch):
        monkeypatch.setenv("MINIO_ACCESS_KEY", "minio-access")
        monkeypatch.setenv("MINIO_SECRET_KEY", "minio-secret")
        _mirror_minio_credentials_to_aws()
        import os

        assert os.environ["AWS_ACCESS_KEY_ID"] == "minio-access"
        assert os.environ["AWS_SECRET_ACCESS_KEY"] == "minio-secret"

    def test_does_not_overwrite_explicit_aws_creds(self, monkeypatch):
        monkeypatch.setenv("MINIO_ACCESS_KEY", "minio-access")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "explicit-aws")
        _mirror_minio_credentials_to_aws()
        import os

        assert os.environ["AWS_ACCESS_KEY_ID"] == "explicit-aws"

    def test_no_minio_creds_leaves_aws_unset(self, monkeypatch):
        _mirror_minio_credentials_to_aws()
        import os

        assert "AWS_ACCESS_KEY_ID" not in os.environ
        assert "AWS_SECRET_ACCESS_KEY" not in os.environ
