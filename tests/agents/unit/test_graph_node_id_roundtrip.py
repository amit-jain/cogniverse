"""node_id_from_doc_id must invert the real Node.doc_id format.

Regression: cross_tenant / federated agents recovered a
node_id with ``doc_id.split('_', 3)[-1]``. The doc_id is
``kg_node_{safe_tenant}_{node_id}`` where safe_tenant = tenant_id.replace(':','_')
— so for a canonical ``org:tenant`` (and node_ids that contain underscores)
the positional split glued the tenant's trailing segment onto the node_id.
These round-trip through the REAL Node.doc_id producer.
"""

import pytest

from cogniverse_agents.graph.graph_schema import Node, node_id_from_doc_id


@pytest.mark.parametrize(
    "tenant_id, name",
    [
        ("acme", "Marie Curie"),  # single-segment tenant, multi-word name
        ("acme:cell_a", "Marie Curie"),  # colon-form tenant WITH underscore
        ("my_org:cell_b", "Niels Bohr Institute"),  # underscores in both segments + multiword
        ("acme:prod", "X"),  # short node_id
    ],
)
def test_node_id_roundtrips_through_real_doc_id(tenant_id, name):
    node = Node(tenant_id=tenant_id, name=name, mentions=[])
    # Encode with production, decode with the helper — must recover exactly.
    recovered = node_id_from_doc_id(node.doc_id, tenant_id)
    assert recovered == node.node_id


def test_colon_tenant_does_not_leak_into_node_id():
    """The exact failure the old split produced."""
    node = Node(tenant_id="acme:cell_a", name="Marie Curie", mentions=[])
    # doc_id is kg_node_acme_cell_a_marie_curie
    assert node.doc_id == "kg_node_acme_cell_a_marie_curie"
    # old code: split('_', 3)[-1] == "cell_a_marie_curie" (WRONG)
    assert node.doc_id.split("_", 3)[-1] == "cell_a_marie_curie"
    # fixed: recovers the true node_id
    assert node_id_from_doc_id(node.doc_id, "acme:cell_a") == "marie_curie"


def test_non_matching_doc_id_returns_empty():
    assert node_id_from_doc_id("kg_edge_acme_e1", "acme") == ""
    assert node_id_from_doc_id("kg_node_other_x", "acme") == ""
    assert node_id_from_doc_id("", "acme") == ""
