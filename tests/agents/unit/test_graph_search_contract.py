"""GraphManager read paths surface Vespa failures instead of flattening them.

_search_filtered logged non-200s and returned [] and never checked
root.errors, _visit/_visit_edges swallowed every failure to [],
get_edge_by_id mapped an outage to None (indistinguishable from
not-found), and the search_nodes fallback accepted name_contains but never
filtered by it — so a degraded or down Vespa read as an empty-but-healthy
graph, and the encoder-down fallback returned every tenant node for any
query. These pin the raise contract and the wired fallback filter.
"""

from types import SimpleNamespace

import pytest
import requests

from cogniverse_agents.graph.graph_manager import GraphManager
from cogniverse_agents.search.vespa_query import VespaSearchDegraded

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

TENANT = "acme"
SCHEMA = "kg_test"

DEGRADED = {"root": {"errors": [{"code": 12, "summary": "Timeout"}], "children": []}}


class _Resp:
    def __init__(self, body, ok=True, status_code=200, text="ok"):
        self._body = body
        self.ok = ok
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._body


class _Session:
    def __init__(self, resp=None, exc=None):
        self.calls = []
        self._resp = resp
        self._exc = exc

    def post(self, url, json=None, timeout=None):
        self.calls.append(json)
        if self._exc is not None:
            raise self._exc
        return self._resp


def _bare_manager(session) -> GraphManager:
    mgr = object.__new__(GraphManager)
    mgr._backend = SimpleNamespace(_url="http://vespa", _port=8080)
    mgr._tenant_id = TENANT
    mgr._schema_name = SCHEMA
    mgr._http = session
    return mgr


def test_search_filtered_raises_on_soft_timeout():
    mgr = _bare_manager(_Session(resp=_Resp(DEGRADED)))
    with pytest.raises(VespaSearchDegraded, match="errors"):
        mgr._search_filtered(['doc_type contains "node"'], top_k=5)


def test_search_filtered_raises_on_http_error():
    mgr = _bare_manager(
        _Session(resp=_Resp({}, ok=False, status_code=503, text="down"))
    )
    with pytest.raises(RuntimeError, match="503"):
        mgr._search_filtered(['doc_type contains "node"'], top_k=5)


def test_visit_propagates_outage_instead_of_empty():
    mgr = _bare_manager(_Session(exc=requests.ConnectionError("refused")))
    with pytest.raises(requests.ConnectionError):
        mgr._visit(doc_type="node")


def test_visit_edges_propagates_outage():
    mgr = _bare_manager(_Session(exc=requests.ConnectionError("refused")))
    with pytest.raises(requests.ConnectionError):
        mgr._visit_edges(source_node_id="n1")


def test_get_stats_propagates_outage():
    """A down Vespa must not read as an empty graph (node_count=0)."""
    mgr = _bare_manager(_Session(exc=requests.ConnectionError("refused")))
    with pytest.raises(requests.ConnectionError):
        mgr.get_stats()


def test_visit_name_contains_filters_yql():
    sess = _Session(resp=_Resp({"root": {"children": [{"fields": {"name": "V"}}]}}))
    mgr = _bare_manager(sess)
    mgr._visit(doc_type="node", name_contains='Ves"pa')
    yql = sess.calls[0]["yql"]
    assert 'name contains "Ves\\"pa"' in yql


def test_search_nodes_encoder_down_fallback_filters_by_query():
    """Encoder failure degrades to the filtered visit WITH the query text."""
    sess = _Session(resp=_Resp({"root": {"children": [{"fields": {"name": "Vespa"}}]}}))
    mgr = _bare_manager(sess)

    def boom(q):
        raise RuntimeError("encoder sidecar down")

    mgr._encode_query_blocks = boom
    hits = mgr.search_nodes("Vespa", top_k=5)
    assert hits == [{"name": "Vespa"}]
    assert 'name contains "Vespa"' in sess.calls[0]["yql"]


def test_search_nodes_query_failure_raises_not_fallback():
    """A Vespa-side failure surfaces; it must not cascade into a second
    (fallback) query that would mask the degradation as thin results."""
    sess = _Session(resp=_Resp(DEGRADED))
    mgr = _bare_manager(sess)
    mgr._encode_query_blocks = lambda q: ([], [])
    with pytest.raises(VespaSearchDegraded):
        mgr.search_nodes("anything", top_k=5)
    assert len(sess.calls) == 1


def test_get_edge_by_id_outage_raises_not_none():
    mgr = _bare_manager(_Session())
    mgr._backend = SimpleNamespace(
        _url="http://vespa",
        _port=8080,
        get_document_fields=lambda *a, **k: (_ for _ in ()).throw(
            ConnectionError("refused")
        ),
    )
    with pytest.raises(ConnectionError):
        mgr.get_edge_by_id("e1")
