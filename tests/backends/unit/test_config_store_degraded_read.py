"""Vespa soft-timeout (degraded) reads must raise, not read as absent config.

A Vespa soft-timeout is HTTP 200 with ``root.errors`` and degraded coverage
plus empty/partial hits. VespaConfigStore.get_config must raise on that shape
so the caller never mistakes a degraded backend for a genuinely-absent config
(which would trigger mass re-submit + cooldown wipe in quality_monitor_cli).
A genuinely-absent config (empty hits, no errors, coverage not degraded) must
still return None.
"""

from datetime import datetime

import pytest

from cogniverse_sdk.interfaces.config_store import ConfigEntry, ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _FakeQueryResponse:
    """Faithful stand-in for pyvespa's VespaQueryResponse.

    ``hits`` mirrors ``root.children`` and ``get_json`` exposes the raw
    JSON body the same way pyvespa does, so degraded detection sees the
    real contract (root.errors + coverage.degraded).
    """

    def __init__(self, json_body):
        self._json = json_body

    @property
    def hits(self):
        return self._json.get("root", {}).get("children", [])

    def get_json(self):
        return self._json


class _FakeVespaApp:
    def __init__(self, response):
        self._response = response
        self.url = "http://localhost:8080"

    def query(self, yql=None, **kwargs):
        return self._response


def _soft_timeout_response():
    return _FakeQueryResponse(
        {
            "root": {
                "errors": [{"code": 12, "summary": "Timeout"}],
                "coverage": {"degraded": {"timeout": True}},
                "children": [],
            }
        }
    )


def _clean_absent_response():
    return _FakeQueryResponse(
        {
            "root": {
                "coverage": {"coverage": 100, "full": True},
                "children": [],
            }
        }
    )


def _healthy_hit_response():
    now = datetime(2026, 7, 20, 12, 0, 0).isoformat()
    return _FakeQueryResponse(
        {
            "root": {
                "coverage": {"coverage": 100, "full": True},
                "children": [
                    {
                        "fields": {
                            "config_id": "acme:system:system:poll_state",
                            "tenant_id": "acme",
                            "scope": "system",
                            "service": "system",
                            "config_key": "poll_state",
                            "config_value": '{"last_run": 42}',
                            "version": 7,
                            "created_at": now,
                            "updated_at": now,
                        }
                    }
                ],
            }
        }
    )


def _store_with(response):
    return VespaConfigStore(vespa_app=_FakeVespaApp(response))


def test_get_config_raises_on_soft_timeout():
    store = _store_with(_soft_timeout_response())
    with pytest.raises(RuntimeError, match="degraded"):
        store.get_config("acme", ConfigScope.SYSTEM, "system", "poll_state")


def test_get_config_returns_none_on_clean_absence():
    store = _store_with(_clean_absent_response())
    result = store.get_config("acme", ConfigScope.SYSTEM, "system", "poll_state")
    assert result is None


def test_get_config_returns_entry_on_healthy_hit():
    store = _store_with(_healthy_hit_response())
    entry = store.get_config("acme", ConfigScope.SYSTEM, "system", "poll_state")
    assert isinstance(entry, ConfigEntry)
    assert entry.config_key == "poll_state"
    assert entry.version == 7
    assert entry.config_value == {"last_run": 42}


def test_get_config_history_raises_on_soft_timeout():
    store = _store_with(_soft_timeout_response())
    with pytest.raises(RuntimeError, match="degraded"):
        store.get_config_history("acme", ConfigScope.SYSTEM, "system", "poll_state")


def test_list_configs_raises_on_soft_timeout():
    store = _store_with(_soft_timeout_response())
    with pytest.raises(RuntimeError, match="degraded"):
        store.list_configs("acme")
