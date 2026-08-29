"""Config reads distinguish successful absence from backend degradation.

Public config reads use Document v1 visits and propagate transport failures.
Query-backed maintenance operations still reject HTTP-200 soft timeouts with
root errors or degraded coverage instead of reporting empty state.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import requests

import cogniverse_vespa.config.config_store as config_store_module
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
    now = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc).isoformat()
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


class _FakeVisitResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _visit_payload(fields=None):
    documents = [] if fields is None else [{"id": "id:config::entry", "fields": fields}]
    return {"documents": documents}


def _healthy_fields():
    return _healthy_hit_response().hits[0]["fields"]


class _FakeClock:
    def __init__(self, start: float = 1000.0):
        self.current = start

    def monotonic(self) -> float:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.current += seconds


def _expected_visit_failure_message(
    attempts: int, elapsed: float, error: Exception
) -> str:
    return (
        "Failed to read Vespa config visit after "
        f"{attempts} attempts over {elapsed:.3f}s: {type(error).__name__}: {error}"
    )


def test_get_config_raises_on_visit_timeout(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(
        config_store_module,
        "time",
        SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep),
        raising=False,
    )

    attempts = config_store_module._CONFIG_STORE_READ_MAX_ATTEMPTS
    failures = [requests.Timeout("visit timed out") for _ in range(attempts)]
    calls = {"count": 0}

    def timeout(*_args, **_kwargs):
        index = calls["count"]
        calls["count"] += 1
        raise failures[index] if index < len(failures) else failures[-1]

    monkeypatch.setattr(requests, "get", timeout)
    store = _store_with(_clean_absent_response())

    with pytest.raises(RuntimeError) as exc_info:
        store.get_config("acme", ConfigScope.SYSTEM, "system", "poll_state")

    assert calls["count"] == attempts
    assert exc_info.value.__cause__ is failures[-1]
    assert str(exc_info.value) == _expected_visit_failure_message(
        attempts,
        sum(
            config_store_module._config_store_visit_backoff_seconds(attempt)
            for attempt in range(1, attempts)
        ),
        failures[-1],
    )


def test_get_config_returns_none_on_clean_absence(monkeypatch):
    store = _store_with(_clean_absent_response())
    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: _FakeVisitResponse(_visit_payload()),
    )
    result = store.get_config("acme", ConfigScope.SYSTEM, "system", "poll_state")
    assert result is None


def test_get_config_returns_entry_on_healthy_hit(monkeypatch):
    store = _store_with(_healthy_hit_response())
    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: _FakeVisitResponse(_visit_payload(_healthy_fields())),
    )
    entry = store.get_config("acme", ConfigScope.SYSTEM, "system", "poll_state")
    assert isinstance(entry, ConfigEntry)
    assert entry.config_key == "poll_state"
    assert entry.version == 7
    assert entry.config_value == {"last_run": 42}
    assert entry.created_at == datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)
    assert entry.updated_at == entry.created_at


def test_get_config_rejects_obsolete_naive_timestamp(monkeypatch):
    fields = _healthy_fields()
    fields["created_at"] = "2026-07-20T12:00:00"

    store = _store_with(_clean_absent_response())
    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: _FakeVisitResponse(_visit_payload(fields)),
    )
    with pytest.raises(ValueError, match="created_at.*timezone"):
        store.get_config("acme", ConfigScope.SYSTEM, "system", "poll_state")


def test_get_config_history_raises_on_visit_timeout(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(
        config_store_module,
        "time",
        SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep),
        raising=False,
    )

    attempts = config_store_module._CONFIG_STORE_READ_MAX_ATTEMPTS
    failures = [requests.Timeout("visit timed out") for _ in range(attempts)]
    calls = {"count": 0}

    def timeout(*_args, **_kwargs):
        index = calls["count"]
        calls["count"] += 1
        raise failures[index] if index < len(failures) else failures[-1]

    monkeypatch.setattr(requests, "get", timeout)
    store = _store_with(_clean_absent_response())

    with pytest.raises(RuntimeError) as exc_info:
        store.get_config_history("acme", ConfigScope.SYSTEM, "system", "poll_state")

    assert calls["count"] == attempts
    assert exc_info.value.__cause__ is failures[-1]
    assert str(exc_info.value) == _expected_visit_failure_message(
        attempts,
        sum(
            config_store_module._config_store_visit_backoff_seconds(attempt)
            for attempt in range(1, attempts)
        ),
        failures[-1],
    )


def test_list_configs_raises_on_visit_timeout(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(
        config_store_module,
        "time",
        SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep),
        raising=False,
    )

    attempts = config_store_module._CONFIG_STORE_READ_MAX_ATTEMPTS
    failures = [requests.Timeout("visit timed out") for _ in range(attempts)]
    calls = {"count": 0}

    def timeout(*_args, **_kwargs):
        index = calls["count"]
        calls["count"] += 1
        raise failures[index] if index < len(failures) else failures[-1]

    monkeypatch.setattr(requests, "get", timeout)
    store = _store_with(_clean_absent_response())

    with pytest.raises(RuntimeError) as exc_info:
        store.list_configs("acme")

    assert calls["count"] == attempts
    assert exc_info.value.__cause__ is failures[-1]
    assert str(exc_info.value) == _expected_visit_failure_message(
        attempts,
        sum(
            config_store_module._config_store_visit_backoff_seconds(attempt)
            for attempt in range(1, attempts)
        ),
        failures[-1],
    )


def test_latest_version_read_raises_on_soft_timeout():
    """The latest-version read gates every write: a soft-timeout returns
    empty hits, which used to read as version 0 — set_config then wrote
    version 1 BELOW the real latest and the operator's change silently
    never took effect."""
    store = _store_with(_soft_timeout_response())
    with pytest.raises(RuntimeError, match="degraded"):
        store._get_latest_version("acme", ConfigScope.SYSTEM, "system", "poll_state")


def test_get_stats_raises_on_soft_timeout():
    """A degraded scan must raise, not present partial (or zero) counts as
    complete stats — a dashboard keyed off the counts would read an empty
    store during a Vespa blip. Matches the raising sibling reads."""
    store = _store_with(_soft_timeout_response())
    with pytest.raises(RuntimeError, match="degraded"):
        store.get_stats()


def test_export_configs_history_raises_on_soft_timeout():
    """export_configs(include_history=True) must not return a partial export a
    caller would persist as authoritative when the scan is degraded."""
    store = _store_with(_soft_timeout_response())
    with pytest.raises(RuntimeError, match="degraded"):
        store.export_configs("acme:acme", include_history=True)


class _RaisingVespaApp:
    url = "http://localhost:8080"

    def query(self, yql=None, **kwargs):
        raise ConnectionError("config store unreachable")


def test_get_stats_raises_on_outage():
    store = VespaConfigStore(vespa_app=_RaisingVespaApp())
    with pytest.raises(ConnectionError):
        store.get_stats()


def test_export_configs_raises_on_outage():
    store = VespaConfigStore(vespa_app=_RaisingVespaApp())
    with pytest.raises(ConnectionError):
        store.export_configs("acme:acme", include_history=True)
