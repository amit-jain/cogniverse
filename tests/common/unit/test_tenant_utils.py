"""
Unit tests for tenant utilities.

Tests tenant ID parsing and storage path generation.
"""

import tempfile
from pathlib import Path

import pytest

from cogniverse_core.common.tenant_utils import (
    SYSTEM_TENANT_ID,
    TEST_TENANT_ID,
    get_tenant_storage_path,
    parse_tenant_id,
    require_tenant_id,
    validate_tenant_id,
)


class TestParseTenantId:
    """Test tenant ID parsing"""

    @pytest.mark.ci_fast
    def test_parse_simple_format(self):
        """Test parsing simple tenant ID"""
        org_id, tenant_name = parse_tenant_id("acme")
        assert org_id == "acme"
        assert tenant_name == "acme"

    @pytest.mark.ci_fast
    def test_parse_org_tenant_format(self):
        """Test parsing org:tenant format"""
        org_id, tenant_name = parse_tenant_id("acme:production")
        assert org_id == "acme"
        assert tenant_name == "production"

    @pytest.mark.ci_fast
    def test_parse_empty_tenant_id(self):
        """Test parsing empty tenant ID raises error"""
        with pytest.raises(ValueError, match="tenant_id cannot be empty"):
            parse_tenant_id("")

    @pytest.mark.ci_fast
    def test_parse_invalid_format(self):
        """Test parsing invalid format raises error"""
        with pytest.raises(ValueError, match="Invalid tenant_id"):
            parse_tenant_id("org:tenant:extra")

    @pytest.mark.ci_fast
    def test_parse_empty_org(self):
        """Test parsing empty org raises error"""
        with pytest.raises(
            ValueError, match="both org and tenant parts must be non-empty"
        ):
            parse_tenant_id(":tenant")

    @pytest.mark.ci_fast
    def test_parse_empty_tenant(self):
        """Test parsing empty tenant raises error"""
        with pytest.raises(
            ValueError, match="both org and tenant parts must be non-empty"
        ):
            parse_tenant_id("org:")


class TestGetTenantStoragePath:
    """Test tenant storage path generation"""

    @pytest.mark.ci_fast
    def test_simple_format_path(self):
        """Test storage path for simple tenant ID"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = get_tenant_storage_path(temp_dir, "acme")
            assert path == Path(temp_dir) / "acme"

    @pytest.mark.ci_fast
    def test_org_tenant_format_path(self):
        """Test storage path for org:tenant format"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = get_tenant_storage_path(temp_dir, "acme:production")
            assert path == Path(temp_dir) / "acme" / "production"

    @pytest.mark.ci_fast
    def test_path_with_str_base_dir(self):
        """Test storage path with string base directory"""
        path = get_tenant_storage_path("data/optimization", "acme:production")
        assert path == Path("data/optimization") / "acme" / "production"

    @pytest.mark.ci_fast
    def test_path_with_path_base_dir(self):
        """Test storage path with Path base directory"""
        base_path = Path("data/optimization")
        path = get_tenant_storage_path(base_path, "acme:production")
        assert path == base_path / "acme" / "production"


class TestValidateTenantId:
    """Test tenant ID validation"""

    @pytest.mark.ci_fast
    def test_validate_simple_tenant_id(self):
        """Test validating simple tenant ID"""
        validate_tenant_id("acme")  # Should not raise

    @pytest.mark.ci_fast
    def test_validate_org_tenant_format(self):
        """Test validating org:tenant format"""
        validate_tenant_id("acme:production")  # Should not raise

    @pytest.mark.ci_fast
    def test_validate_with_underscore(self):
        """Test validating tenant ID with underscore"""
        validate_tenant_id("acme_corp")  # Should not raise
        validate_tenant_id("acme:prod_env")  # Should not raise

    @pytest.mark.ci_fast
    def test_validate_with_hyphen(self):
        """Test validating tenant ID with hyphen"""
        validate_tenant_id("acme-corp")  # Should not raise
        validate_tenant_id("acme:prod-env")  # Should not raise

    @pytest.mark.ci_fast
    def test_validate_empty_tenant_id(self):
        """Test validating empty tenant ID"""
        with pytest.raises(ValueError, match="tenant_id cannot be empty"):
            validate_tenant_id("")

    @pytest.mark.ci_fast
    def test_validate_non_string(self):
        """Test validating non-string tenant ID"""
        with pytest.raises(ValueError, match="tenant_id must be string"):
            validate_tenant_id(123)

    @pytest.mark.ci_fast
    def test_validate_invalid_characters(self):
        """Test validating tenant ID with invalid characters"""
        with pytest.raises(ValueError, match="only alphanumeric"):
            validate_tenant_id("acme@corp")

        with pytest.raises(ValueError, match="only alphanumeric"):
            validate_tenant_id("acme corp")  # Space not allowed

    @pytest.mark.ci_fast
    def test_validate_multiple_colons(self):
        """Test validating tenant ID with multiple colons"""
        with pytest.raises(ValueError, match="expected 'org:tenant' with single colon"):
            validate_tenant_id("org:tenant:extra")

    @pytest.mark.ci_fast
    def test_validate_empty_org_part(self):
        """Test validating tenant ID with empty org"""
        with pytest.raises(
            ValueError, match="both org and tenant parts must be non-empty"
        ):
            validate_tenant_id(":tenant")

    @pytest.mark.ci_fast
    def test_validate_empty_tenant_part(self):
        """Test validating tenant ID with empty tenant"""
        with pytest.raises(
            ValueError, match="both org and tenant parts must be non-empty"
        ):
            validate_tenant_id("org:")


class TestReservedPrefix:
    """Reserved __ prefix is rejected for user-registrable tenants."""

    @pytest.mark.ci_fast
    def test_validate_rejects_system_tenant_id(self):
        """SYSTEM_TENANT_ID must not be registrable as a user tenant."""
        with pytest.raises(
            ValueError, match="reserved for runtime-internal use"
        ):
            validate_tenant_id(SYSTEM_TENANT_ID)

    @pytest.mark.ci_fast
    def test_validate_rejects_any_double_underscore_prefix(self):
        """Any identifier starting with __ is reserved."""
        with pytest.raises(ValueError, match="reserved"):
            validate_tenant_id("__cluster__")
        with pytest.raises(ValueError, match="reserved"):
            validate_tenant_id("__internal")

    @pytest.mark.ci_fast
    def test_validate_allows_single_underscore_prefix(self):
        """Single-underscore prefix is a legal user tenant id."""
        validate_tenant_id("_legacy")  # must not raise

    @pytest.mark.ci_fast
    def test_validate_allows_test_tenant_id(self):
        """TEST_TENANT_ID is a normal user tenant, not reserved."""
        validate_tenant_id(TEST_TENANT_ID)


class TestRequireTenantId:
    """require_tenant_id raises on missing/invalid values."""

    @pytest.mark.ci_fast
    def test_raises_on_none(self):
        with pytest.raises(ValueError, match="tenant_id is required"):
            require_tenant_id(None, source="TestSource")

    @pytest.mark.ci_fast
    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError, match="tenant_id is required"):
            require_tenant_id("", source="TestSource")

    @pytest.mark.ci_fast
    def test_raises_on_non_string(self):
        with pytest.raises(ValueError, match="must be a string"):
            require_tenant_id(123, source="TestSource")  # type: ignore[arg-type]

    @pytest.mark.ci_fast
    def test_returns_valid_tenant_id(self):
        """On valid input, returns the value unchanged for inline use."""
        assert (
            require_tenant_id("acme:production", source="TestSource")
            == "acme:production"
        )

    @pytest.mark.ci_fast
    def test_source_label_in_error_message(self):
        """Error message should name the source to help debugging."""
        with pytest.raises(ValueError, match="SearchRequest"):
            require_tenant_id(None, source="SearchRequest")


class TestReservedIdentities:
    """SYSTEM_TENANT_ID and TEST_TENANT_ID are defined and distinct."""

    @pytest.mark.ci_fast
    def test_system_tenant_id_is_reserved_prefix(self):
        assert SYSTEM_TENANT_ID.startswith("__")

    @pytest.mark.ci_fast
    def test_test_tenant_id_is_user_space(self):
        """TEST_TENANT_ID does NOT use the reserved prefix."""
        assert not TEST_TENANT_ID.startswith("__")

    @pytest.mark.ci_fast
    def test_identities_are_distinct(self):
        assert SYSTEM_TENANT_ID != TEST_TENANT_ID


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
