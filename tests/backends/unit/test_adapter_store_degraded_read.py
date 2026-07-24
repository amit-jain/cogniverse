"""Vespa soft-timeout (degraded) reads must raise, not read as absent adapter.

A Vespa soft-timeout is HTTP 200 with ``root.errors`` and degraded coverage
plus empty/partial hits — pyvespa does NOT raise on it. VespaAdapterStore
reads must raise on that shape: ``get_active_adapter`` returning ``None`` on
a degraded response silently reverts a tenant's finetuned LoRA to the base
model (adapter_loader treats ``None`` as "genuinely no active adapter").
A genuinely-absent adapter (empty hits, no errors, coverage not degraded)
must still return ``None``/``[]``.
"""

import pytest

from cogniverse_vespa.registry.adapter_store import VespaAdapterStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _FakeQueryResponse:
    """Faithful stand-in for pyvespa's VespaQueryResponse.

    ``hits`` mirrors ``root.children`` and ``get_json`` exposes the raw JSON
    body the same way pyvespa does, so degraded detection sees the real
    contract (root.errors + coverage.degraded).
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


def _empty_response():
    return _FakeQueryResponse({"root": {"children": []}})


def _store(response) -> VespaAdapterStore:
    return VespaAdapterStore(vespa_app=_FakeVespaApp(response))


class TestDegradedReadsRaise:
    def test_get_active_adapter_raises_on_degraded(self):
        with pytest.raises(RuntimeError, match="degraded"):
            _store(_soft_timeout_response()).get_active_adapter("acme:acme", "routing")

    def test_get_adapter_raises_on_degraded(self):
        with pytest.raises(RuntimeError, match="degraded"):
            _store(_soft_timeout_response()).get_adapter("adapter-1")

    def test_list_adapters_raises_on_degraded(self):
        with pytest.raises(RuntimeError, match="degraded"):
            _store(_soft_timeout_response()).list_adapters("acme:acme")

    def test_get_stats_raises_on_degraded(self):
        with pytest.raises(RuntimeError, match="degraded"):
            _store(_soft_timeout_response()).get_stats()


class TestGenuineAbsenceStillNone:
    def test_get_active_adapter_none_when_genuinely_absent(self):
        assert (
            _store(_empty_response()).get_active_adapter("acme:acme", "routing") is None
        )

    def test_get_adapter_none_when_genuinely_absent(self):
        assert _store(_empty_response()).get_adapter("adapter-1") is None

    def test_list_adapters_empty_when_genuinely_absent(self):
        assert _store(_empty_response()).list_adapters("acme:acme") == []

    def test_get_stats_zero_counts_when_genuinely_empty(self):
        stats = _store(_empty_response()).get_stats()
        assert stats["total_adapters"] == 0
        assert stats["total_tenants"] == 0
        assert stats["adapters_by_status"] == {}


class TestOutageRaises:
    """A hard backend failure must propagate, never flatten to no-data."""

    class _RaisingApp:
        url = "http://localhost:8080"

        def query(self, yql=None, **kwargs):
            raise ConnectionError("backend down")

    def test_get_active_adapter_raises_on_outage(self):
        store = VespaAdapterStore(vespa_app=self._RaisingApp())
        with pytest.raises(ConnectionError):
            store.get_active_adapter("acme:acme", "routing")

    def test_get_stats_raises_on_outage(self):
        store = VespaAdapterStore(vespa_app=self._RaisingApp())
        with pytest.raises(ConnectionError):
            store.get_stats()
