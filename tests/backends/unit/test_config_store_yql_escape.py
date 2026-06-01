"""get_config(version=N) must escape the documentid value.

config_id derives from raw tenant_id/service/config_key (via
_create_document_id) and was interpolated into `documentid = "{doc_id}"`
unescaped — a quote in config_key broke or injected the YQL. The latest-version
branch already used yql_quote; the versioned branch must too.
"""

from __future__ import annotations

from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa._yql import yql_quote
from cogniverse_vespa.config.config_store import VespaConfigStore


class _EmptyResponse:
    hits: list = []


def test_versioned_documentid_is_escaped():
    store = object.__new__(VespaConfigStore)
    store.schema_name = "config_metadata"
    captured = {}

    class _App:
        def query(self, yql):
            captured["yql"] = yql
            return _EmptyResponse()

    store.vespa_app = _App()

    store.get_config(
        tenant_id="acme:acme",
        scope=ConfigScope.SCHEMA,
        service="svc",
        config_key='key"; bad',
        version=2,
    )

    config_id = store._create_document_id(
        "acme:acme", ConfigScope.SCHEMA, "svc", 'key"; bad'
    )
    doc_id = f"config_metadata::{config_id}::2"
    # yql_quote escapes the inner quote; pre-fix raw interpolation did not, so
    # the escaped literal only appears when the value is quoted safely.
    assert yql_quote(doc_id) in captured["yql"]
    assert 'key\\"; bad' in captured["yql"]
