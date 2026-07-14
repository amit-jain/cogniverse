"""get_config(version=N) must escape the config_id value.

config_id derives from raw tenant_id/service/config_key (via
_create_document_id) and was once interpolated unescaped — a quote in
config_key broke or injected the YQL. The versioned branch queries
``config_id contains <quoted> and version = <int>`` (Vespa has no queryable
``documentid`` field; the old ``where documentid = ...`` form fails with a
400), and the interpolated value must go through yql_quote.
"""

from __future__ import annotations

from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa._yql import yql_quote
from cogniverse_vespa.config.config_store import VespaConfigStore


class _EmptyResponse:
    hits: list = []


def test_versioned_config_id_is_escaped():
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
    # yql_quote escapes the inner quote; pre-fix raw interpolation did not, so
    # the escaped literal only appears when the value is quoted safely.
    assert (
        f"config_id contains {yql_quote(config_id)} and version = 2"
        in (captured["yql"])
    )
    assert 'key\\"; bad' in captured["yql"]
