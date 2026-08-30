from __future__ import annotations

import pytest

from cogniverse_agents.graph.graph_schema import node_id_from_doc_id
from tests.utils.kg_lookup import kg_node_doc_id, resolve_persisted_kg_nodes

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class RecordingBackend:
    def __init__(self, *, tenant_id: str, entity_ids: list[str]) -> None:
        self.tenant_id = tenant_id
        self.search_calls: list[tuple[str, int]] = []
        self.get_doc_calls: list[str] = []
        self.docs = {
            kg_node_doc_id(entity_id, tenant_id): {
                "doc_id": kg_node_doc_id(entity_id, tenant_id),
                "tenant_id": tenant_id,
                "doc_type": "node",
                "name": entity_id,
            }
            for entity_id in entity_ids
        }

    def search_nodes(self, *, tenant_id: str, hits: int):
        self.search_calls.append((tenant_id, hits))
        return list(self.docs.values())[:400]

    def get_node_doc(self, doc_id: str):
        self.get_doc_calls.append(doc_id)
        return self.docs.get(doc_id)


def test_resolve_persisted_kg_nodes_is_exhaustive_and_per_id():
    tenant_id = "acme:tenant"
    expected_ids = [f"entity_{i:03d}" for i in range(501)]
    backend = RecordingBackend(tenant_id=tenant_id, entity_ids=expected_ids)

    resolved, dangling = resolve_persisted_kg_nodes(
        expected_ids,
        tenant_id=tenant_id,
        backend=backend,
    )

    expected_doc_ids = [
        kg_node_doc_id(entity_id, tenant_id) for entity_id in expected_ids
    ]

    assert dangling == []
    assert list(resolved) == expected_ids
    assert {entity_id: doc["doc_id"] for entity_id, doc in resolved.items()} == {
        entity_id: doc_id for entity_id, doc_id in zip(expected_ids, expected_doc_ids)
    }
    assert backend.search_calls == []
    assert backend.get_doc_calls == expected_doc_ids
    assert {
        entity_id: node_id_from_doc_id(doc["doc_id"], tenant_id)
        for entity_id, doc in resolved.items()
    } == {entity_id: entity_id for entity_id in expected_ids}
