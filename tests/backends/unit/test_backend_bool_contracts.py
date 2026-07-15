"""VespaBackend's bool-contracted surfaces return honest booleans.

``validate_schema`` used to be a validator that never validated (its try body
was a comment plus an unconditional ``return True``), and ``health_check``
(typed ``-> bool`` by the SearchBackend ABC) leaked the search backend's
status DICT — always truthy, even when degraded.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cogniverse_vespa.backend import VespaBackend

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _bare_backend():
    backend = object.__new__(VespaBackend)
    backend.schema_manager = MagicMock()
    backend._vespa_search_backend = None
    return backend


def test_validate_schema_checks_deployed_document_types():
    backend = _bare_backend()
    backend.schema_manager.list_deployed_document_types.return_value = [
        "agent_memories",
        "video_colpali_smol500_mv_frame",
    ]

    assert backend.validate_schema("agent_memories") is True
    assert backend.validate_schema("nonexistent_schema") is False


def test_validate_schema_returns_false_on_listing_failure():
    backend = _bare_backend()
    backend.schema_manager.list_deployed_document_types.side_effect = RuntimeError(
        "config server down"
    )

    assert backend.validate_schema("agent_memories") is False


def test_health_check_coerces_status_dict_to_bool():
    backend = _bare_backend()
    search = MagicMock()
    backend._vespa_search_backend = search

    search.health_check.return_value = {"status": "healthy", "components": {}}
    assert backend.health_check() is True

    search.health_check.return_value = {"status": "degraded", "components": {}}
    assert backend.health_check() is False
