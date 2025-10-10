"""
Unit tests for tenant utilities.

Tests tenant ID parsing and storage path generation.
"""

import tempfile
from pathlib import Path

import pytest

from cogniverse_core.common.tenant_utils import (
    get_tenant_storage_path,
    parse_tenant_id,
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
        with pytest.raises(ValueError, match="both org and tenant parts must be non-empty"):
            parse_tenant_id(":tenant")

    @pytest.mark.ci_fast
    def test_parse_empty_tenant(self):
        """Test parsing empty tenant raises error"""
        with pytest.raises(ValueError, match="both org and tenant parts must be non-empty"):
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
        with pytest.raises(ValueError, match="both org and tenant parts must be non-empty"):
            validate_tenant_id(":tenant")

    @pytest.mark.ci_fast
    def test_validate_empty_tenant_part(self):
        """Test validating tenant ID with empty tenant"""
        with pytest.raises(ValueError, match="both org and tenant parts must be non-empty"):
            validate_tenant_id("org:")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
