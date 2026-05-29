"""``VespaConfigStore.list_all_configs`` builds a Document v1 selection
expression. The ``service`` filter is a free string; previously it was
interpolated raw (``service == "pr"obe"``), producing a malformed selection
(Vespa 400) and a selection-injection vector. It must go through yql_quote
like every other value the store interpolates.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests

from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore

SCHEMA = "config_metadata"


@pytest.fixture
def store() -> VespaConfigStore:
    # list_all_configs only reads self.schema_name and self.vespa_app.url;
    # bypass __init__ which needs a live Vespa app.
    s = object.__new__(VespaConfigStore)
    s.schema_name = SCHEMA
    s.vespa_app = SimpleNamespace(url="http://vespa:8080")
    return s


class _FakeResponse:
    @staticmethod
    def raise_for_status() -> None:
        return None

    @staticmethod
    def json() -> dict:
        return {"documents": [], "continuation": None}


@pytest.fixture
def captured(monkeypatch):
    calls = {}

    def fake_get(path, params=None, timeout=None):
        calls["params"] = params
        return _FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


def test_service_filter_escapes_embedded_quote(store, captured):
    store.list_all_configs(scope=ConfigScope.SCHEMA, service='pr"obe')
    selection = captured["params"]["selection"]
    assert selection == (
        f'{SCHEMA}.scope == "schema" and {SCHEMA}.service == "pr\\"obe"'
    )
    assert 'service == "pr"obe"' not in selection


def test_service_only_filter_is_quoted(store, captured):
    store.list_all_configs(service="probe")
    assert captured["params"]["selection"] == f'{SCHEMA}.service == "probe"'
