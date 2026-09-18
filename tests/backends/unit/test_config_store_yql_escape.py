"""get_config(version=N) must escape Document visit selections, and name the
one key it reads.

config_id derives from raw tenant_id/service/config_key (via
_create_document_id). A quote in tenant_id, scope, or service must not break
the Document v1 selection expression. The visit also carries the config_key,
so a point read does not visit and parse every other key stored under the same
service — the deployment journal keeps every tenant's intents under one.
"""

from __future__ import annotations

import pytest
import requests

from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore


class _EmptyVisitResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"documents": []}


def test_versioned_config_selection_is_escaped(monkeypatch):
    store = object.__new__(VespaConfigStore)
    store.schema_name = "config_metadata"
    captured = {}

    class _App:
        url = "http://localhost:8080"

    store.vespa_app = _App()

    def capture_get(url, *, params, timeout):
        captured["url"] = url
        captured["params"] = dict(params)
        captured["timeout"] = timeout
        return _EmptyVisitResponse()

    monkeypatch.setattr(requests, "get", capture_get)
    result = store.get_config(
        tenant_id='acme:"quoted',
        scope=ConfigScope.SCHEMA,
        service='svc"; bad',
        config_key="key",
        version=2,
    )

    # An empty visit means the requested version does not exist.
    assert result is None
    assert captured == {
        "url": (
            "http://localhost:8080/document/v1/config_metadata/config_metadata/docid/"
        ),
        "params": {
            "wantedDocumentCount": 1000,
            "selection": (
                'config_metadata.tenant_id == "acme:\\"quoted" and '
                'config_metadata.scope == "schema" and '
                'config_metadata.service == "svc\\"; bad" and '
                'config_metadata.config_key == "key"'
            ),
        },
        "timeout": 30,
    }


def test_config_key_suffix_narrows_the_selection(monkeypatch):
    """The suffix reaches the store as a glob on config_key, quoted like the
    rest of the selection, so the visit carries only the matching rows."""
    store = object.__new__(VespaConfigStore)
    store.schema_name = "config_metadata"
    captured = {}

    class _App:
        url = "http://localhost:8080"

    store.vespa_app = _App()

    def capture_get(url, *, params, timeout):
        captured["params"] = dict(params)
        return _EmptyVisitResponse()

    monkeypatch.setattr(requests, "get", capture_get)
    result = store.list_all_configs(
        scope=ConfigScope.SCHEMA,
        service="schema_deployment_intents",
        config_key_suffix='_acme"corp',
    )

    assert result == []
    assert captured["params"] == {
        "wantedDocumentCount": 1000,
        "selection": (
            'config_metadata.scope == "schema" and '
            'config_metadata.service == "schema_deployment_intents" and '
            'config_metadata.config_key = "*_acme\\"corp"'
        ),
    }


def test_a_wildcard_suffix_is_refused(monkeypatch):
    """A selection glob has no escape, so a suffix carrying one would widen the
    match to other owners' rows instead of narrowing it."""
    store = object.__new__(VespaConfigStore)
    store.schema_name = "config_metadata"

    class _App:
        url = "http://localhost:8080"

    store.vespa_app = _App()

    def refuse_get(*_args, **_kwargs):
        raise AssertionError("the store must not be read for a wildcard suffix")

    monkeypatch.setattr(requests, "get", refuse_get)
    for suffix in ("_acme*", "_acme?"):
        with pytest.raises(ValueError, match="carries a glob wildcard"):
            store.list_all_configs(
                scope=ConfigScope.SCHEMA,
                service="schema_deployment_intents",
                config_key_suffix=suffix,
            )
