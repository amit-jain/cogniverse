"""Helpers for resolving persisted KG nodes in tests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from cogniverse_agents.graph.graph_schema import Node


def kg_node_doc_id(entity_id: str, tenant_id: str) -> str:
    """Return the production doc id for a KG node id."""
    return Node(tenant_id=tenant_id, name=entity_id, mentions=[]).doc_id


def resolve_persisted_kg_nodes(
    expected_entity_ids: Iterable[str],
    *,
    tenant_id: str,
    backend: Any,
) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    """Resolve each expected KG node id to its persisted Vespa doc.

    The backend supplies the Vespa access primitive. The returned mapping is
    keyed by node id; missing ids are reported with the doc id they were
    resolved against.
    """
    resolved: dict[str, Mapping[str, Any]] = {}
    dangling: list[str] = []
    for entity_id in sorted({str(entity_id) for entity_id in expected_entity_ids}):
        doc_id = kg_node_doc_id(entity_id, tenant_id)
        doc = backend.get_node_doc(doc_id)
        if doc is None:
            dangling.append(f"{entity_id} (doc_id={doc_id})")
            continue
        resolved[entity_id] = doc
    return resolved, dangling
