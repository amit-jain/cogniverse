"""Regression tests for tenant_id escaping in GraphManager Vespa queries.

A tenant_id containing a double quote previously produced a malformed
Document-v1 selection (`...tenant_id=="ac"me"`) and a malformed YQL
`contains "ac"me"`, which Vespa rejects — graph traversal then silently
returned [] for that tenant. These tests capture the exact string sent to
the Vespa boundary and assert the quote is escaped via yql_quote.
"""

from types import SimpleNamespace

import pytest

from cogniverse_agents.graph import graph_manager as gm_module
from cogniverse_agents.graph.graph_manager import GraphManager

TENANT = 'ac"me'
SCHEMA = "kg_test"


def _bare_manager() -> GraphManager:
    """Build a GraphManager without running the model-loading __init__."""
    mgr = object.__new__(GraphManager)
    mgr._backend = SimpleNamespace(_url="http://vespa", _port=8080)
    mgr._tenant_id = TENANT
    mgr._schema_name = SCHEMA
    return mgr


class _FakeResponse:
    ok = True

    @staticmethod
    def json():
        return {"documents": [], "root": {"children": []}}


@pytest.fixture
def captured_get(monkeypatch):
    calls = {}

    def fake_get(url, params=None, timeout=None):
        calls["url"] = url
        calls["params"] = params
        return _FakeResponse()

    monkeypatch.setattr(gm_module.requests, "get", fake_get)
    return calls


def test_visit_escapes_tenant_quote(captured_get):
    _bare_manager()._visit(doc_type="node")
    selection = captured_get["params"]["selection"]
    assert selection == f'{SCHEMA}.doc_type=="node" and {SCHEMA}.tenant_id=="ac\\"me"'
    # the malformed (unescaped) form must never reach Vespa
    assert 'tenant_id=="ac"me"' not in selection


def test_visit_edges_escapes_tenant_and_node_ids(captured_get):
    _bare_manager()._visit_edges(source_node_id='no"de')
    selection = captured_get["params"]["selection"]
    assert f'{SCHEMA}.tenant_id=="ac\\"me"' in selection
    assert f'{SCHEMA}.source_node_id=="no\\"de"' in selection
    assert 'tenant_id=="ac"me"' not in selection


def test_search_nodes_escapes_tenant_quote_in_yql(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["body"] = json
        return _FakeResponse()

    monkeypatch.setattr(gm_module.requests, "post", fake_post)

    mgr = _bare_manager()
    monkeypatch.setattr(
        mgr, "_encode_query_blocks", lambda q: ([{"0": "x"}], [{"0": "y"}])
    )
    mgr.search_nodes("find things")

    yql = captured["body"]["yql"]
    assert 'tenant_id contains "ac\\"me"' in yql
    assert 'tenant_id contains "ac"me"' not in yql
