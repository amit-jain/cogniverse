"""
Integration tests for profile-agnostic SearchService.

Validates:
- ONE SearchService instance serves multiple profiles
- Encoder caching: same model loaded once, reused across calls
- Profile/tenant_id passed at search() time, not construction
- Missing profile raises ValueError with available profiles listed
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from cogniverse_agents.search.service import SearchService


@pytest.fixture
def mock_config():
    """Config with multiple profiles sharing/differing embedding models."""
    return {
        "backend": {
            "type": "vespa",
            "url": "http://localhost",
            "port": 8080,
            "profiles": {
                "frame_based_colpali": {
                    "embedding_model": "vidore/colSmol-256M",
                    "embedding_dim": 128,
                    "embedding_format": "binary",
                },
                "video_colqwen_omni": {
                    "embedding_model": "vidore/colqwen2-v1.0",
                    "embedding_dim": 128,
                    "embedding_format": "binary",
                },
                "video_videoprism_base": {
                    "embedding_model": "google/videoprism-base",
                    "embedding_dim": 768,
                    "embedding_format": "float",
                },
            },
        },
    }


@pytest.fixture
def mock_config_manager():
    return Mock()


@pytest.fixture
def mock_schema_loader():
    return Mock()


@pytest.fixture
def search_service(mock_config, mock_config_manager, mock_schema_loader):
    """Create SearchService with mocked dependencies."""
    with patch(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        return_value=Mock(),
    ):
        return SearchService(
            config=mock_config,
            config_manager=mock_config_manager,
            schema_loader=mock_schema_loader,
        )


class TestSearchServiceConstruction:
    """Test SearchService profile-agnostic construction."""

    def test_init_no_profile_no_tenant(self, search_service):
        """SearchService initializes without profile or tenant_id."""
        assert search_service._backends == {}
        assert search_service.config_manager is not None
        assert search_service.schema_loader is not None

    def test_init_requires_config_manager(self, mock_config, mock_schema_loader):
        """Raises ValueError if config_manager is None."""
        with pytest.raises(ValueError, match="config_manager is required"):
            SearchService(
                config=mock_config,
                config_manager=None,
                schema_loader=mock_schema_loader,
            )

    def test_init_requires_schema_loader(self, mock_config, mock_config_manager):
        """Raises ValueError if schema_loader is None."""
        with pytest.raises(ValueError, match="schema_loader is required"):
            with patch(
                "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
                return_value=Mock(),
            ):
                SearchService(
                    config=mock_config,
                    config_manager=mock_config_manager,
                    schema_loader=None,
                )


class TestProfileRouting:
    """Test profile-based routing at search() time."""

    def test_get_profile_config_success(self, search_service):
        """Valid profile returns its config."""
        config = search_service._get_profile_config("frame_based_colpali")
        assert config["embedding_model"] == "vidore/colSmol-256M"
        assert config["embedding_dim"] == 128

    def test_get_profile_config_multiple_profiles(self, search_service):
        """Different profiles return different configs."""
        colpali = search_service._get_profile_config("frame_based_colpali")
        videoprism = search_service._get_profile_config("video_videoprism_base")

        assert colpali["embedding_model"] != videoprism["embedding_model"]
        assert colpali["embedding_dim"] != videoprism["embedding_dim"]

    def test_get_profile_config_unknown_raises(self, search_service):
        """Unknown profile raises ValueError with available profiles."""
        with pytest.raises(ValueError, match="Profile 'nonexistent' not found") as exc:
            search_service._get_profile_config("nonexistent")

        # Error message includes available profiles
        assert "frame_based_colpali" in str(exc.value)
        assert "video_colqwen_omni" in str(exc.value)
        assert "video_videoprism_base" in str(exc.value)


class TestEncoderCaching:
    """Test QueryEncoderFactory caching behavior."""

    def test_encoder_cache_reuse(self):
        """Same model_name returns cached encoder on second call."""
        from cogniverse_core.query.encoders import QueryEncoderFactory

        # Clear any existing cache
        QueryEncoderFactory._encoder_cache.clear()

        mock_encoder = MagicMock()
        mock_encoder.get_embedding_dim.return_value = 128

        config = {
            "backend": {
                "profiles": {
                    "profile_a": {"embedding_model": "test-model"},
                    "profile_b": {"embedding_model": "test-model"},
                }
            }
        }

        with patch.object(
            QueryEncoderFactory,
            "_create_encoder_instance",
            return_value=mock_encoder,
        ) as mock_create:
            # First call — creates encoder
            enc1 = QueryEncoderFactory.create_encoder(
                "profile_a", model_name="test-model", config=config
            )
            # Second call with same model — should reuse cache
            enc2 = QueryEncoderFactory.create_encoder(
                "profile_b", model_name="test-model", config=config
            )

            assert enc1 is enc2
            mock_create.assert_called_once()  # Only created once

        # Cleanup
        QueryEncoderFactory._encoder_cache.clear()

    def test_encoder_cache_different_models(self):
        """Different model_names create separate encoders."""
        from cogniverse_core.query.encoders import QueryEncoderFactory

        QueryEncoderFactory._encoder_cache.clear()

        mock_encoder_a = MagicMock()
        mock_encoder_b = MagicMock()

        config = {
            "backend": {
                "profiles": {
                    "profile_a": {"embedding_model": "model-a"},
                    "profile_b": {"embedding_model": "model-b"},
                }
            }
        }

        with patch.object(
            QueryEncoderFactory,
            "_create_encoder_instance",
            side_effect=[mock_encoder_a, mock_encoder_b],
        ) as mock_create:
            enc1 = QueryEncoderFactory.create_encoder(
                "profile_a", model_name="model-a", config=config
            )
            enc2 = QueryEncoderFactory.create_encoder(
                "profile_b", model_name="model-b", config=config
            )

            assert enc1 is not enc2
            assert mock_create.call_count == 2

        QueryEncoderFactory._encoder_cache.clear()

    def test_encoder_requires_config(self):
        """create_encoder raises ValueError without config."""
        from cogniverse_core.query.encoders import QueryEncoderFactory

        with pytest.raises(ValueError, match="config is required"):
            QueryEncoderFactory.create_encoder("some_profile", config=None)


class TestBackendCaching:
    """Test lazy backend creation per tenant_id."""

    def test_backends_start_empty(self, search_service):
        """No backends until first search() call."""
        assert len(search_service._backends) == 0

    def test_get_backend_caches_by_tenant(self, search_service):
        """_get_backend caches backends by tenant_id."""
        mock_encoder = MagicMock()
        profile_config = {"embedding_model": "test", "schema_name": "test_schema"}

        mock_backend = MagicMock()
        with patch("cogniverse_agents.search.service.get_backend_registry") as mock_reg:
            mock_reg.return_value.get_search_backend.return_value = mock_backend

            # First call — creates backend
            b1 = search_service._get_backend(
                "tenant_a", "frame_based_colpali", profile_config, mock_encoder
            )
            # Second call same tenant — returns cached
            b2 = search_service._get_backend(
                "tenant_a", "frame_based_colpali", profile_config, mock_encoder
            )

            assert b1 is b2
            # Registry only called once (cached for second call)
            assert mock_reg.return_value.get_search_backend.call_count == 1

    def test_different_tenants_create_separate_backends(self, search_service):
        """Different tenant_ids create separate backend entries."""
        mock_encoder = MagicMock()
        profile_config = {"embedding_model": "test", "schema_name": "test_schema"}

        with patch("cogniverse_agents.search.service.get_backend_registry") as mock_reg:
            mock_reg.return_value.get_search_backend.side_effect = [
                MagicMock(name="backend_a"),
                MagicMock(name="backend_b"),
            ]

            b1 = search_service._get_backend(
                "tenant_a", "frame_based_colpali", profile_config, mock_encoder
            )
            b2 = search_service._get_backend(
                "tenant_b", "frame_based_colpali", profile_config, mock_encoder
            )

            assert b1 is not b2
            assert len(search_service._backends) == 2
            assert "tenant_a" in search_service._backends
            assert "tenant_b" in search_service._backends
