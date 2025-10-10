"""
Unit tests for tenant_manager validation functions.

Tests validation logic for organization IDs and tenant names without requiring Vespa.
"""

import pytest

from cogniverse_runtime.admin.tenant_manager import validate_org_id, validate_tenant_name


class TestValidateOrgId:
    """Test validate_org_id function"""

    def test_valid_simple_org_id(self):
        """Test valid simple org_id passes"""
        validate_org_id("acme")  # Should not raise

    def test_valid_org_id_with_underscore(self):
        """Test org_id with underscore is valid"""
        validate_org_id("acme_corp")  # Should not raise
        validate_org_id("my_organization")  # Should not raise

    def test_valid_org_id_alphanumeric(self):
        """Test org_id with numbers is valid"""
        validate_org_id("org123")  # Should not raise
        validate_org_id("acme2024")  # Should not raise

    def test_valid_org_id_mixed(self):
        """Test org_id with mixed alphanumeric and underscore"""
        validate_org_id("acme_corp_2024")  # Should not raise

    def test_empty_org_id_raises_error(self):
        """Test empty org_id raises ValueError"""
        with pytest.raises(ValueError, match="org_id cannot be empty"):
            validate_org_id("")

    def test_none_org_id_raises_error(self):
        """Test None org_id raises ValueError"""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_org_id(None)

    def test_invalid_org_id_with_hyphen(self):
        """Test org_id with hyphen raises ValueError"""
        with pytest.raises(ValueError, match="only alphanumeric and underscore"):
            validate_org_id("acme-corp")

    def test_invalid_org_id_with_special_char(self):
        """Test org_id with special characters raises ValueError"""
        with pytest.raises(ValueError, match="only alphanumeric and underscore"):
            validate_org_id("acme@corp")

        with pytest.raises(ValueError, match="only alphanumeric and underscore"):
            validate_org_id("acme.corp")

        with pytest.raises(ValueError, match="only alphanumeric and underscore"):
            validate_org_id("acme corp")  # Space

    def test_invalid_org_id_with_colon(self):
        """Test org_id with colon raises ValueError"""
        with pytest.raises(ValueError, match="only alphanumeric and underscore"):
            validate_org_id("acme:corp")

    def test_invalid_org_id_non_string(self):
        """Test non-string org_id raises ValueError"""
        with pytest.raises(ValueError, match="org_id must be string"):
            validate_org_id(123)

        with pytest.raises(ValueError, match="org_id must be string"):
            validate_org_id(["acme"])


class TestValidateTenantName:
    """Test validate_tenant_name function"""

    def test_valid_simple_tenant_name(self):
        """Test valid simple tenant name passes"""
        validate_tenant_name("production")  # Should not raise
        validate_tenant_name("dev")  # Should not raise
        validate_tenant_name("staging")  # Should not raise

    def test_valid_tenant_name_with_underscore(self):
        """Test tenant name with underscore is valid"""
        validate_tenant_name("prod_env")  # Should not raise
        validate_tenant_name("my_tenant")  # Should not raise

    def test_valid_tenant_name_with_hyphen(self):
        """Test tenant name with hyphen is valid"""
        validate_tenant_name("prod-2024")  # Should not raise
        validate_tenant_name("dev-env")  # Should not raise

    def test_valid_tenant_name_alphanumeric(self):
        """Test tenant name with numbers is valid"""
        validate_tenant_name("tenant123")  # Should not raise
        validate_tenant_name("prod2024")  # Should not raise

    def test_valid_tenant_name_mixed(self):
        """Test tenant name with mixed characters"""
        validate_tenant_name("prod_env_2024")  # Should not raise
        validate_tenant_name("dev-env-1")  # Should not raise
        validate_tenant_name("staging_v2")  # Should not raise

    def test_empty_tenant_name_raises_error(self):
        """Test empty tenant name raises ValueError"""
        with pytest.raises(ValueError, match="tenant_name cannot be empty"):
            validate_tenant_name("")

    def test_none_tenant_name_raises_error(self):
        """Test None tenant name raises ValueError"""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_tenant_name(None)

    def test_invalid_tenant_name_with_special_char(self):
        """Test tenant name with special characters raises ValueError"""
        with pytest.raises(
            ValueError, match="only alphanumeric, underscore, and hyphen"
        ):
            validate_tenant_name("prod@env")

        with pytest.raises(
            ValueError, match="only alphanumeric, underscore, and hyphen"
        ):
            validate_tenant_name("prod.env")

        with pytest.raises(
            ValueError, match="only alphanumeric, underscore, and hyphen"
        ):
            validate_tenant_name("prod env")  # Space

    def test_invalid_tenant_name_with_colon(self):
        """Test tenant name with colon raises ValueError"""
        with pytest.raises(
            ValueError, match="only alphanumeric, underscore, and hyphen"
        ):
            validate_tenant_name("prod:env")

    def test_invalid_tenant_name_non_string(self):
        """Test non-string tenant name raises ValueError"""
        with pytest.raises(ValueError, match="tenant_name must be string"):
            validate_tenant_name(123)

        with pytest.raises(ValueError, match="tenant_name must be string"):
            validate_tenant_name(["production"])


class TestValidationEdgeCases:
    """Test edge cases for validation functions"""

    def test_org_id_case_sensitive(self):
        """Test org_id validation is case sensitive"""
        validate_org_id("ACME")  # Should not raise
        validate_org_id("AcmeCorp")  # Should not raise
        validate_org_id("acme")  # Should not raise

    def test_tenant_name_case_sensitive(self):
        """Test tenant name validation is case sensitive"""
        validate_tenant_name("PRODUCTION")  # Should not raise
        validate_tenant_name("Production")  # Should not raise
        validate_tenant_name("production")  # Should not raise

    def test_org_id_numeric_only(self):
        """Test org_id can be numeric only"""
        validate_org_id("123")  # Should not raise
        validate_org_id("456789")  # Should not raise

    def test_tenant_name_numeric_only(self):
        """Test tenant name can be numeric only"""
        validate_tenant_name("123")  # Should not raise
        validate_tenant_name("456789")  # Should not raise

    def test_org_id_single_character(self):
        """Test single alphanumeric character org_id is valid"""
        validate_org_id("a")  # Should not raise
        validate_org_id("1")  # Should not raise

    def test_org_id_only_underscore_invalid(self):
        """Test org_id with only underscore is invalid"""
        with pytest.raises(ValueError, match="only alphanumeric and underscore"):
            validate_org_id("_")

    def test_tenant_name_single_character(self):
        """Test single alphanumeric character tenant name is valid"""
        validate_tenant_name("a")  # Should not raise
        validate_tenant_name("1")  # Should not raise

    def test_tenant_name_only_special_char_invalid(self):
        """Test tenant name with only underscore or hyphen is invalid"""
        with pytest.raises(
            ValueError, match="only alphanumeric, underscore, and hyphen"
        ):
            validate_tenant_name("_")

        with pytest.raises(
            ValueError, match="only alphanumeric, underscore, and hyphen"
        ):
            validate_tenant_name("-")

    def test_org_id_long_name(self):
        """Test long org_id is valid"""
        long_org = "a" * 100
        validate_org_id(long_org)  # Should not raise

    def test_tenant_name_long_name(self):
        """Test long tenant name is valid"""
        long_tenant = "a" * 100
        validate_tenant_name(long_tenant)  # Should not raise
