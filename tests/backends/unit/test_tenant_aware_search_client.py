"""
Unit tests for TenantAwareVespaSearchClient.

Tests automatic tenant schema routing and lazy creation.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from cogniverse_vespa.tenant_aware_search_client import TenantAwareVespaSearchClient


class TestTenantAwareSearchClientInitialization:
    """Test client initialization and schema resolution"""

    @patch("cogniverse_vespa.tenant_aware_search_client.get_tenant_schema_manager")
    @patch("cogniverse_vespa.tenant_aware_search_client.VespaVideoSearchClient")
    def test_initialization_with_required_params(self, mock_client_class, mock_manager_func):
        """Test successful initialization with required parameters"""
        mock_manager = MagicMock()
        mock_manager.get_tenant_schema_name.return_value = "video_colpali_acme"
        mock_manager.ensure_tenant_schema_exists.return_value = True
        mock_manager_func.return_value = mock_manager

        client = TenantAwareVespaSearchClient(
            tenant_id="acme", base_schema_name="video_colpali", config_manager=MagicMock()
        )

        assert client.tenant_id == "acme"
        assert client.base_schema_name == "video_colpali"
        assert client.tenant_schema_name == "video_colpali_acme"

    @patch("cogniverse_vespa.tenant_aware_search_client.get_tenant_schema_manager")
    def test_initialization_requires_tenant_id(self, mock_manager_func):
        """Test that tenant_id is required"""
        with pytest.raises(ValueError, match="tenant_id is required"):
            TenantAwareVespaSearchClient(tenant_id="", base_schema_name="video_colpali", config_manager=MagicMock())

    @patch("cogniverse_vespa.tenant_aware_search_client.get_tenant_schema_manager")
    def test_initialization_requires_base_schema_name(self, mock_manager_func):
        """Test that base_schema_name is required"""
        with pytest.raises(ValueError, match="base_schema_name is required"):
            TenantAwareVespaSearchClient(tenant_id="acme", base_schema_name="", config_manager=MagicMock())

    @patch("cogniverse_vespa.tenant_aware_search_client.get_tenant_schema_manager")
    @patch("cogniverse_vespa.tenant_aware_search_client.VespaVideoSearchClient")
    def test_schema_name_resolution(self, mock_client_class, mock_manager_func):
        """Test that schema names are resolved correctly"""
        mock_manager = MagicMock()
        mock_manager.get_tenant_schema_name.return_value = (
            "video_colpali_smol500_mv_frame_acme"
        )
        mock_manager.ensure_tenant_schema_exists.return_value = True
        mock_manager_func.return_value = mock_manager

        client = TenantAwareVespaSearchClient(
            tenant_id="acme", base_schema_name="video_colpali_smol500_mv_frame", config_manager=MagicMock()
        )

        # Verify schema manager was called correctly
        mock_manager.get_tenant_schema_name.assert_called_once_with(
            "acme", "video_colpali_smol500_mv_frame"
        )
        assert client.tenant_schema_name == "video_colpali_smol500_mv_frame_acme"

    @patch("cogniverse_vespa.tenant_aware_search_client.get_tenant_schema_manager")
    @patch("cogniverse_vespa.tenant_aware_search_client.VespaVideoSearchClient")
    def test_auto_create_schema_enabled(self, mock_client_class, mock_manager_func):
        """Test that schema is automatically created when auto_create_schema=True"""
        mock_manager = MagicMock()
        mock_manager.get_tenant_schema_name.return_value = "video_colpali_acme"
        mock_manager.ensure_tenant_schema_exists.return_value = True
        mock_manager_func.return_value = mock_manager

        _client = TenantAwareVespaSearchClient(
            tenant_id="acme", base_schema_name="video_colpali", config_manager=MagicMock(), auto_create_schema=True
        )

        # Verify lazy schema creation was called
        mock_manager.ensure_tenant_schema_exists.assert_called_once_with(
            "acme", "video_colpali"
        )

    @patch("cogniverse_vespa.tenant_aware_search_client.get_tenant_schema_manager")
    @patch("cogniverse_vespa.tenant_aware_search_client.VespaVideoSearchClient")
    def test_auto_create_schema_disabled(self, mock_client_class, mock_manager_func):
        """Test that schema is not created when auto_create_schema=False"""
        mock_manager = MagicMock()
        mock_manager.get_tenant_schema_name.return_value = "video_colpali_acme"
        mock_manager_func.return_value = mock_manager

        _client = TenantAwareVespaSearchClient(
            tenant_id="acme", base_schema_name="video_colpali", config_manager=MagicMock(), auto_create_schema=False
        )

        # Verify lazy schema creation was NOT called
        mock_manager.ensure_tenant_schema_exists.assert_not_called()


class TestTenantAwareSearchClientSearchMethods:
    """Test search method delegation"""

    @patch("cogniverse_vespa.tenant_aware_search_client.get_tenant_schema_manager")
    @patch("cogniverse_vespa.tenant_aware_search_client.VespaVideoSearchClient")
    def test_search_delegates_with_tenant_schema(self, mock_client_class, mock_manager_func):
        """Test that search() delegates to underlying client with tenant schema"""
        mock_manager = MagicMock()
        mock_manager.get_tenant_schema_name.return_value = "video_colpali_acme"
        mock_manager.ensure_tenant_schema_exists.return_value = True
        mock_manager_func.return_value = mock_manager

        mock_vespa_client = MagicMock()
        mock_vespa_client.search.return_value = [{"result": "data"}]
        mock_client_class.return_value = mock_vespa_client

        client = TenantAwareVespaSearchClient(
            tenant_id="acme", base_schema_name="video_colpali", config_manager=MagicMock()
        )

        # Perform search
        _results = client.search(query_text="test query", strategy="hybrid_float_bm25", top_k=10)

        # Verify delegation to underlying client with tenant schema
        mock_vespa_client.search.assert_called_once()
        call_args = mock_vespa_client.search.call_args

        # Check that query_params dict contains expected values
        assert "query_params" in call_args.kwargs
        query_params = call_args.kwargs["query_params"]
        assert query_params["query"] == "test query"
        assert query_params["ranking"] == "hybrid_float_bm25"
        assert query_params["top_k"] == 10

        # Check schema is passed correctly
        assert call_args.kwargs["schema"] == "video_colpali_acme"  # ✅ Tenant schema

    @patch("cogniverse_vespa.tenant_aware_search_client.get_tenant_schema_manager")
    @patch("cogniverse_vespa.tenant_aware_search_client.VespaVideoSearchClient")
    def test_hybrid_search_delegates_with_tenant_schema(
        self, mock_client_class, mock_manager_func
    ):
        """Test that hybrid_search() delegates with tenant schema"""
        mock_manager = MagicMock()
        mock_manager.get_tenant_schema_name.return_value = "video_colpali_acme"
        mock_manager.ensure_tenant_schema_exists.return_value = True
        mock_manager_func.return_value = mock_manager

        mock_vespa_client = MagicMock()
        mock_vespa_client.hybrid_search.return_value = [{"result": "data"}]
        mock_client_class.return_value = mock_vespa_client

        client = TenantAwareVespaSearchClient(
            tenant_id="acme", base_schema_name="video_colpali", config_manager=MagicMock()
        )

        # Perform hybrid search
        _results = client.hybrid_search(
            query_text="test", visual_weight=0.7, text_weight=0.3
        )

        # Verify delegation with tenant schema
        mock_vespa_client.hybrid_search.assert_called_once()
        call_args = mock_vespa_client.hybrid_search.call_args

        assert call_args.kwargs["query_text"] == "test"
        assert call_args.kwargs["visual_weight"] == 0.7
        assert call_args.kwargs["text_weight"] == 0.3
        assert call_args.kwargs["schema"] == "video_colpali_acme"  # ✅ Tenant schema

    @patch("cogniverse_vespa.tenant_aware_search_client.get_tenant_schema_manager")
    @patch("cogniverse_vespa.tenant_aware_search_client.VespaVideoSearchClient")
    def test_search_with_embeddings(self, mock_client_class, mock_manager_func):
        """Test search with embeddings"""
        mock_manager = MagicMock()
        mock_manager.get_tenant_schema_name.return_value = "video_colpali_acme"
        mock_manager.ensure_tenant_schema_exists.return_value = True
        mock_manager_func.return_value = mock_manager

        mock_vespa_client = MagicMock()
        mock_vespa_client.search.return_value = []
        mock_client_class.return_value = mock_vespa_client

        client = TenantAwareVespaSearchClient(
            tenant_id="acme", base_schema_name="video_colpali", config_manager=MagicMock()
        )

        embeddings = np.random.rand(10, 128)
        client.search(query_text="test", embeddings=embeddings)

        # Verify embeddings were passed
        call_args = mock_vespa_client.search.call_args
        assert call_args.kwargs["embeddings"] is embeddings


class TestTenantAwareSearchClientUtilityMethods:
    """Test utility methods"""

    @patch("cogniverse_vespa.tenant_aware_search_client.get_tenant_schema_manager")
    @patch("cogniverse_vespa.tenant_aware_search_client.VespaVideoSearchClient")
    def test_health_check(self, mock_client_class, mock_manager_func):
        """Test health_check delegates to underlying client"""
        mock_manager = MagicMock()
        mock_manager.get_tenant_schema_name.return_value = "video_colpali_acme"
        mock_manager.ensure_tenant_schema_exists.return_value = True
        mock_manager_func.return_value = mock_manager

        mock_vespa_client = MagicMock()
        mock_vespa_client.health_check.return_value = True
        mock_client_class.return_value = mock_vespa_client

        client = TenantAwareVespaSearchClient(
            tenant_id="acme", base_schema_name="video_colpali", config_manager=MagicMock()
        )

        result = client.health_check()

        assert result is True
        mock_vespa_client.health_check.assert_called_once()

    @patch("cogniverse_vespa.tenant_aware_search_client.get_tenant_schema_manager")
    @patch("cogniverse_vespa.tenant_aware_search_client.VespaVideoSearchClient")
    def test_get_tenant_info(self, mock_client_class, mock_manager_func):
        """Test get_tenant_info returns tenant information"""
        mock_manager = MagicMock()
        mock_manager.get_tenant_schema_name.return_value = "video_colpali_acme"
        mock_manager.ensure_tenant_schema_exists.return_value = True
        mock_manager_func.return_value = mock_manager

        mock_vespa_client = MagicMock()
        mock_client_class.return_value = mock_vespa_client

        client = TenantAwareVespaSearchClient(
            tenant_id="acme", base_schema_name="video_colpali", config_manager=MagicMock(), backend_port=8080
        )

        info = client.get_tenant_info()

        assert info["tenant_id"] == "acme"
        assert info["base_schema_name"] == "video_colpali"
        assert info["tenant_schema_name"] == "video_colpali_acme"
        assert info["backend_port"] == "8080"

    @patch("cogniverse_vespa.tenant_aware_search_client.get_tenant_schema_manager")
    @patch("cogniverse_vespa.tenant_aware_search_client.VespaVideoSearchClient")
    def test_get_available_strategies(self, mock_client_class, mock_manager_func):
        """Test get_available_strategies delegates to underlying client"""
        mock_manager = MagicMock()
        mock_manager.get_tenant_schema_name.return_value = "video_colpali_acme"
        mock_manager.ensure_tenant_schema_exists.return_value = True
        mock_manager_func.return_value = mock_manager

        mock_vespa_client = MagicMock()
        mock_vespa_client.get_available_strategies.return_value = {"strategy1": {}}
        mock_client_class.return_value = mock_vespa_client

        client = TenantAwareVespaSearchClient(
            tenant_id="acme", base_schema_name="video_colpali", config_manager=MagicMock()
        )

        strategies = client.get_available_strategies()

        assert "strategy1" in strategies
        mock_vespa_client.get_available_strategies.assert_called_once()

    @patch("cogniverse_vespa.tenant_aware_search_client.get_tenant_schema_manager")
    @patch("cogniverse_vespa.tenant_aware_search_client.VespaVideoSearchClient")
    def test_repr(self, mock_client_class, mock_manager_func):
        """Test string representation"""
        mock_manager = MagicMock()
        mock_manager.get_tenant_schema_name.return_value = "video_colpali_acme"
        mock_manager.ensure_tenant_schema_exists.return_value = True
        mock_manager_func.return_value = mock_manager

        mock_vespa_client = MagicMock()
        mock_client_class.return_value = mock_vespa_client

        client = TenantAwareVespaSearchClient(
            tenant_id="acme", base_schema_name="video_colpali", config_manager=MagicMock()
        )

        repr_str = repr(client)

        assert "acme" in repr_str
        assert "video_colpali_acme" in repr_str


class TestTenantAwareSearchClientMultipleTenants:
    """Test behavior with multiple tenants"""

    @patch("cogniverse_vespa.tenant_aware_search_client.get_tenant_schema_manager")
    @patch("cogniverse_vespa.tenant_aware_search_client.VespaVideoSearchClient")
    def test_different_tenants_get_different_schemas(
        self, mock_client_class, mock_manager_func
    ):
        """Test that different tenants get different schema names"""
        mock_manager = MagicMock()

        def side_effect_schema_name(tenant_id, base_schema):
            return f"{base_schema}_{tenant_id}"

        mock_manager.get_tenant_schema_name.side_effect = side_effect_schema_name
        mock_manager.ensure_tenant_schema_exists.return_value = True
        mock_manager_func.return_value = mock_manager

        mock_vespa_client = MagicMock()
        mock_client_class.return_value = mock_vespa_client

        # Create client for tenant A
        client_a = TenantAwareVespaSearchClient(
            tenant_id="acme", base_schema_name="video_colpali", config_manager=MagicMock()
        )

        # Create client for tenant B
        client_b = TenantAwareVespaSearchClient(
            tenant_id="startup", base_schema_name="video_colpali", config_manager=MagicMock()
        )

        # Verify different schema names
        assert client_a.tenant_schema_name == "video_colpali_acme"
        assert client_b.tenant_schema_name == "video_colpali_startup"
