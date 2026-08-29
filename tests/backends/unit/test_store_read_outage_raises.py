"""Config and adapter store reads must raise on a backend outage.

A genuinely-absent config/adapter returns None, but a Vespa read FAILURE used
to return the same None — so a transient outage silently reverted a tenant to
default config or "no adapter". The two cases must be distinguishable: absent
-> None, backend error -> raise.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import requests

import cogniverse_vespa.config.config_store as config_store_module
from cogniverse_sdk.interfaces.config_store import ConfigEntry, ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore
from cogniverse_vespa.registry.adapter_store import VespaAdapterStore


def _config_store(query_impl):
    store = object.__new__(VespaConfigStore)
    store.schema_name = "config_metadata"
    store.vespa_app = MagicMock()
    store.vespa_app.url = "http://localhost:8080"
    store.vespa_app.query = query_impl
    return store


def _adapter_store(query_impl):
    store = object.__new__(VespaAdapterStore)
    store.schema_name = "adapter_registry"
    store.vespa_app = MagicMock()
    store.vespa_app.query = query_impl
    return store


def _empty_response(*_args, **_kwargs):
    return SimpleNamespace(hits=[])


def _boom(*_args, **_kwargs):
    raise ConnectionError("vespa unreachable")


def _empty_visit_response():
    response = MagicMock()
    response.json.return_value = {"documents": [], "continuation": None}
    return response


def _visit_response(documents, continuation=None):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "documents": documents,
        "continuation": continuation,
    }
    return response


def _raise_for_status_error(
    *, status_code: int, reason: str, url: str
) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status_code
    response.reason = reason
    response.url = url
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        return exc
    raise AssertionError("response.raise_for_status() must fail")


class _ScriptedGet:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        index = self.calls
        self.calls += 1
        outcome = (
            self._outcomes[index] if index < len(self._outcomes) else self._outcomes[-1]
        )
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _expected_visit_failure_message(
    attempts: int, elapsed: float, error: Exception
) -> str:
    return (
        "Failed to read Vespa config visit after "
        f"{attempts} attempts over {elapsed:.3f}s: {type(error).__name__}: {error}"
    )


def test_config_absent_returns_none():
    store = _config_store(_empty_response)
    with patch("requests.get", return_value=_empty_visit_response()):
        result = store.get_config(
            tenant_id="acme:acme",
            scope=ConfigScope.BACKEND,
            service="backend",
            config_key="k",
        )
    assert result is None


def test_visit_reads_retry_connection_errors_until_budget_exhausted(monkeypatch):
    clock = SimpleNamespace(
        current=1000.0,
        monotonic=lambda: clock.current,
        sleep=lambda seconds: setattr(clock, "current", clock.current + seconds),
    )
    monkeypatch.setattr(
        config_store_module,
        "time",
        SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep),
        raising=False,
    )

    attempts = config_store_module._CONFIG_STORE_READ_MAX_ATTEMPTS
    failures = [requests.ConnectionError("vespa unreachable") for _ in range(attempts)]
    scripted_get = _ScriptedGet(failures)
    monkeypatch.setattr(requests, "get", scripted_get)

    store = _config_store(_boom)
    for method_name, kwargs in (
        (
            "get_config",
            {
                "tenant_id": "acme:acme",
                "scope": ConfigScope.BACKEND,
                "service": "backend",
                "config_key": "k",
            },
        ),
        (
            "get_config_history",
            {
                "tenant_id": "acme:acme",
                "scope": ConfigScope.BACKEND,
                "service": "backend",
                "config_key": "k",
            },
        ),
        ("list_configs", {"tenant_id": "acme:acme"}),
    ):
        scripted_get.calls = 0
        clock.current = 1000.0
        with pytest.raises(RuntimeError) as exc_info:
            getattr(store, method_name)(**kwargs)

        assert scripted_get.calls == attempts
        assert exc_info.value.__cause__ is failures[-1]
        assert str(exc_info.value) == _expected_visit_failure_message(
            attempts,
            sum(
                config_store_module._config_store_visit_backoff_seconds(attempt)
                for attempt in range(1, attempts)
            ),
            failures[-1],
        )


def test_adapter_absent_returns_none():
    store = _adapter_store(_empty_response)
    assert store.get_adapter("a1") is None


def test_adapter_backend_error_raises():
    store = _adapter_store(_boom)
    with pytest.raises(ConnectionError):
        store.get_adapter("a1")


def test_config_history_empty_returns_empty_list():
    store = _config_store(_empty_response)
    with patch("requests.get", return_value=_empty_visit_response()):
        assert (
            store.get_config_history(
                tenant_id="acme:acme",
                scope=ConfigScope.BACKEND,
                service="backend",
                config_key="k",
            )
            == []
        )


def test_list_configs_empty_returns_empty_list():
    store = _config_store(_empty_response)
    with patch("requests.get", return_value=_empty_visit_response()):
        assert store.list_configs(tenant_id="acme:acme") == []


def test_list_all_configs_retries_500_twice_then_succeeds(monkeypatch):
    clock = SimpleNamespace(
        current=1000.0,
        monotonic=lambda: clock.current,
        sleep=lambda seconds: setattr(clock, "current", clock.current + seconds),
    )
    monkeypatch.setattr(
        config_store_module,
        "time",
        SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep),
        raising=False,
    )

    url = "http://localhost:8080/document/v1/config_metadata/config_metadata/docid/"
    failures = [
        _raise_for_status_error(status_code=500, reason="Server Error", url=url)
        for _ in range(2)
    ]
    now = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)
    payload = _visit_response(
        [
            {
                "id": "id:config_metadata:config_metadata::entry-1",
                "fields": {
                    "config_id": "acme:backend:backend:alpha",
                    "tenant_id": "acme",
                    "scope": "backend",
                    "service": "backend",
                    "config_key": "alpha",
                    "config_value": '{"value": 1}',
                    "version": 1,
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                },
            },
            {
                "id": "id:config_metadata:config_metadata::entry-2",
                "fields": {
                    "config_id": "acme:backend:backend:beta",
                    "tenant_id": "acme",
                    "scope": "backend",
                    "service": "backend",
                    "config_key": "beta",
                    "config_value": '{"value": 2}',
                    "version": 3,
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                },
            },
        ]
    )
    scripted_get = _ScriptedGet([*failures, payload])
    monkeypatch.setattr(requests, "get", scripted_get)

    store = _config_store(_boom)
    results = store.list_all_configs()

    expected = [
        ConfigEntry(
            tenant_id="acme",
            scope=ConfigScope.BACKEND,
            service="backend",
            config_key="alpha",
            config_value={"value": 1},
            version=1,
            created_at=now,
            updated_at=now,
        ),
        ConfigEntry(
            tenant_id="acme",
            scope=ConfigScope.BACKEND,
            service="backend",
            config_key="beta",
            config_value={"value": 2},
            version=3,
            created_at=now,
            updated_at=now,
        ),
    ]

    assert scripted_get.calls == len(failures) + 1
    assert results == expected


def test_list_all_configs_raises_after_retry_budget_exhausted(monkeypatch):
    clock = SimpleNamespace(
        current=1000.0,
        monotonic=lambda: clock.current,
        sleep=lambda seconds: setattr(clock, "current", clock.current + seconds),
    )
    monkeypatch.setattr(
        config_store_module,
        "time",
        SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep),
        raising=False,
    )

    attempts = config_store_module._CONFIG_STORE_READ_MAX_ATTEMPTS
    url = "http://localhost:8080/document/v1/config_metadata/config_metadata/docid/"
    failures = [
        _raise_for_status_error(status_code=500, reason="Server Error", url=url)
        for _ in range(attempts)
    ]
    scripted_get = _ScriptedGet(failures)
    monkeypatch.setattr(requests, "get", scripted_get)

    store = _config_store(_boom)
    with pytest.raises(RuntimeError) as exc_info:
        store.list_all_configs()

    assert scripted_get.calls == attempts
    assert exc_info.value.__cause__ is failures[-1]
    assert str(exc_info.value) == _expected_visit_failure_message(
        attempts,
        sum(
            config_store_module._config_store_visit_backoff_seconds(attempt)
            for attempt in range(1, attempts)
        ),
        failures[-1],
    )


def test_list_all_configs_returns_empty_for_404_without_retry(monkeypatch):
    url = "http://localhost:8080/document/v1/config_metadata/config_metadata/docid/"
    failure = _raise_for_status_error(
        status_code=404,
        reason="Not Found",
        url=url,
    )
    scripted_get = _ScriptedGet([failure])
    monkeypatch.setattr(requests, "get", scripted_get)

    store = _config_store(_boom)
    assert store.list_all_configs() == []
    assert scripted_get.calls == len((failure,))


def test_list_all_configs_does_not_retry_non_transient_400(monkeypatch):
    url = "http://localhost:8080/document/v1/config_metadata/config_metadata/docid/"
    failure = _raise_for_status_error(
        status_code=400,
        reason="Bad Request",
        url=url,
    )
    scripted_get = _ScriptedGet([failure])
    monkeypatch.setattr(requests, "get", scripted_get)

    store = _config_store(_boom)
    with pytest.raises(requests.HTTPError) as exc_info:
        store.list_all_configs()

    assert scripted_get.calls == len((failure,))
    assert exc_info.value.response.status_code == 400


def test_list_configs_backend_error_raises():
    store = _config_store(_boom)
    with (
        patch("requests.get", side_effect=ConnectionError("vespa unreachable")),
        pytest.raises(ConnectionError, match="vespa unreachable"),
    ):
        store.list_configs(tenant_id="acme:acme")


def test_list_all_configs_empty_returns_empty_list():
    """list_all_configs reads the Document v1 visit path, not vespa_app.query."""
    store = _config_store(_empty_response)
    store.vespa_app = SimpleNamespace(url="http://localhost:8080")

    with patch("requests.get", return_value=_empty_visit_response()):
        assert store.list_all_configs() == []


def test_list_all_configs_backend_error_raises(monkeypatch):
    clock = SimpleNamespace(
        current=1000.0,
        monotonic=lambda: clock.current,
        sleep=lambda seconds: setattr(clock, "current", clock.current + seconds),
    )
    monkeypatch.setattr(
        config_store_module,
        "time",
        SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep),
        raising=False,
    )

    attempts = config_store_module._CONFIG_STORE_READ_MAX_ATTEMPTS
    failures = [requests.ConnectionError("vespa unreachable") for _ in range(attempts)]
    scripted_get = _ScriptedGet(failures)
    monkeypatch.setattr(requests, "get", scripted_get)

    store = _config_store(_boom)
    with pytest.raises(RuntimeError) as exc_info:
        store.list_all_configs()

    assert scripted_get.calls == attempts
    assert exc_info.value.__cause__ is failures[-1]
    assert str(exc_info.value) == _expected_visit_failure_message(
        attempts,
        sum(
            config_store_module._config_store_visit_backoff_seconds(attempt)
            for attempt in range(1, attempts)
        ),
        failures[-1],
    )


def test_list_adapters_empty_returns_empty_list():
    store = _adapter_store(_empty_response)
    assert store.list_adapters(tenant_id="acme:acme") == []


def test_list_adapters_backend_error_raises():
    """Finetuning resolves a tenant's LoRA through this — an outage that
    returns [] silently reverts the tenant to the base model."""
    store = _adapter_store(_boom)
    with pytest.raises(ConnectionError):
        store.list_adapters(tenant_id="acme:acme")
