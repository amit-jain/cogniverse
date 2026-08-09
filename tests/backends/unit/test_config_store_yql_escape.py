"""get_config(version=N) must escape Document visit selections.

config_id derives from raw tenant_id/service/config_key (via
_create_document_id). A quote in tenant_id, scope, or service must not break
the Document v1 selection expression.
"""

from __future__ import annotations

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
    store.get_config(
        tenant_id='acme:"quoted',
        scope=ConfigScope.SCHEMA,
        service='svc"; bad',
        config_key="key",
        version=2,
    )

    assert captured == {
        "url": (
            "http://localhost:8080/document/v1/config_metadata/config_metadata/docid/"
        ),
        "params": {
            "wantedDocumentCount": 1000,
            "selection": (
                'config_metadata.tenant_id == "acme:\\"quoted" and '
                'config_metadata.scope == "schema" and '
                'config_metadata.service == "svc\\"; bad"'
            ),
        },
        "timeout": 30,
    }
