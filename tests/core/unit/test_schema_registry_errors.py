"""SchemaRegistry raises typed, chained errors for schema-load failures."""

from unittest.mock import MagicMock

import pytest

from cogniverse_core.registries.exceptions import SchemaLoadError
from cogniverse_core.registries.schema_registry import SchemaRegistry

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def test_deploy_schema_load_failure_raises_typed_chained_error():
    """A caller must be able to tell a missing schema file (permanent, never
    retry) from storage being down (transient) - a bare Exception without a
    cause destroyed the type and left only the message text."""
    loader = MagicMock()
    cause = FileNotFoundError("no such schema: video_missing")
    loader.load_schema.side_effect = cause
    registry = SchemaRegistry(
        config_manager=MagicMock(), backend=MagicMock(), schema_loader=loader
    )
    registry.schema_exists = MagicMock(return_value=False)

    with pytest.raises(
        SchemaLoadError, match="Failed to load base schema 'video_missing'"
    ) as exc_info:
        registry.deploy_schema("acme:acme", "video_missing")

    assert exc_info.value.__cause__ is cause
