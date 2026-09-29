"""Merge knowledge-graph nodes whose id is an article-prefixed twin's.

Claim extraction stores an endpoint the model writes with a leading article
("the Sorbonne") under the entity hint it names ("Sorbonne"). A graph
ingested before that holds ``the_sorbonne`` where new ingests write
``sorbonne``, so one entity is split across two ids. Within one tenant,
``merge_article_nodes`` folds ``<article>_<id>`` into ``<id>`` when ``<id>``
is a node of that tenant:

1. every edge with the article id as an endpoint is re-pointed onto the twin
   and moved to its canonical edge id. An edge that then matches an
   existing one (same endpoints, relation, segment and span) collapses into
   it when both carry the same grounding (source document, evidence span,
   modality), keeping the stronger provenance, the higher confidence and the
   earlier creation time; one with its own grounding is re-pointed in place
   so neither grounding is lost;
2. the tenant's content docs have their ``entity_ids`` / ``relation_ids`` /
   ``claim_ids`` back-refs rewritten to the surviving ids;
3. the retired edge documents are deleted;
4. the article node's mentions are unioned into the twin's, which keeps its
   own name, description, kind and label (the upsert merge semantics);
5. the article node is deleted.

Each step is repeatable and every step's input is re-derived from Vespa, so
an interrupted run resumes by running again, and a completed run finds
nothing to do. Writes go through the backend document API on a backend
leased per operation, like ``GraphManager`` feeds; no step deploys a schema.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import requests

from cogniverse_agents.graph.claim_extractor import _LEADING_ARTICLES
from cogniverse_agents.graph.graph_manager import _GRAPH_BASE_SCHEMA, _GRAPH_NAMESPACE
from cogniverse_agents.graph.graph_schema import (
    Edge,
    Mention,
    _safe_tenant,
    merge_mentions,
    node_id_from_doc_id,
    normalize_name,
)
from cogniverse_agents.search.vespa_query import vespa_search_children
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_core.registries.backend_registry import leased_backend
from cogniverse_vespa._yql import yql_quote

logger = logging.getLogger(__name__)

# Node-id forms of the articles claim extraction drops ("the " -> "the_").
ARTICLE_ID_PREFIXES: Tuple[str, ...] = tuple(
    f"{normalize_name(article)}_" for article in _LEADING_ARTICLES
)

_CONTENT_NAMESPACE = "content"
_BACKREF_FIELDS = ("entity_ids", "relation_ids", "claim_ids")
_NODE_FIELDS = (
    "doc_id",
    "tenant_id",
    "doc_type",
    "name",
    "description",
    "kind",
    "label",
    "mentions",
    "degree",
    "created_at",
    "updated_at",
)
_EDGE_FIELDS = (
    "doc_id",
    "tenant_id",
    "doc_type",
    "source_node_id",
    "target_node_id",
    "relation",
    "evidence_span",
    "segment_id",
    "ts_start",
    "ts_end",
    "modality",
    "provenance",
    "source_doc_id",
    "confidence",
    "created_at",
)
_QUERY_PAGE = 400
_IDS_PER_QUERY = 50
_HTTP_TIMEOUT = 60


# --------------------------------------------------------------------- #
# The merge rule                                                         #
# --------------------------------------------------------------------- #


def article_twin_id(node_id: str) -> Optional[str]:
    """The id ``node_id`` names once a leading article is dropped, or None."""
    for prefix in ARTICLE_ID_PREFIXES:
        if node_id.startswith(prefix) and len(node_id) > len(prefix):
            return node_id[len(prefix) :]
    return None


@dataclass(frozen=True)
class ArticleMergePlan:
    """``merges`` maps each article id to the node it folds into;
    ``skipped`` lists article ids whose bare twin is not a node."""

    merges: Dict[str, str]
    skipped: List[str]


def plan_article_merges(
    node_ids: Iterable[str], endpoint_ids: Iterable[str]
) -> ArticleMergePlan:
    """Decide which article ids merge into which node, within one tenant.

    An article id (a node, or an edge endpoint with no node of its own)
    merges only into a bare twin that is a node. A twin that is itself an
    article id with a twin merges on, so every merge lands on its final node.
    """
    nodes = set(node_ids)
    direct: Dict[str, str] = {}
    skipped = set()
    for candidate in nodes | set(endpoint_ids):
        twin = article_twin_id(candidate)
        if twin is None:
            continue
        if twin in nodes:
            direct[candidate] = twin
        else:
            skipped.add(candidate)
    merges = {}
    for article_id, target in direct.items():
        while target in direct:
            target = direct[target]
        merges[article_id] = target
    return ArticleMergePlan(
        merges=dict(sorted(merges.items())), skipped=sorted(skipped)
    )


def carries_same_grounding(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Whether two edges cite the same source document, span and modality."""
    return all(
        (a.get(name) or "") == (b.get(name) or "")
        for name in ("source_doc_id", "evidence_span", "modality")
    )


def merge_edge_provenance(
    survivor: Dict[str, Any], duplicate: Dict[str, Any]
) -> Dict[str, Any]:
    """Provenance fields of ``survivor`` once ``duplicate`` collapses into it:
    EXTRACTED over INFERRED, the higher confidence, the earlier creation."""
    labels = (survivor.get("provenance"), duplicate.get("provenance"))
    created = [
        c for c in (survivor.get("created_at"), duplicate.get("created_at")) if c
    ]
    return {
        "provenance": "EXTRACTED"
        if "EXTRACTED" in labels
        else survivor.get("provenance"),
        "confidence": max(
            float(survivor.get("confidence") or 0.0),
            float(duplicate.get("confidence") or 0.0),
        ),
        "created_at": min(created) if created else survivor.get("created_at"),
    }


# --------------------------------------------------------------------- #
# Reports                                                                #
# --------------------------------------------------------------------- #


@dataclass
class ArticleMerge:
    source_id: str
    target_id: str
    node_doc: bool
    edges_repointed: int = 0
    edges_deduped: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from": self.source_id,
            "into": self.target_id,
            "node_doc": self.node_doc,
            "edges_repointed": self.edges_repointed,
            "edges_deduped": self.edges_deduped,
        }


@dataclass
class TenantArticleMergeReport:
    """What one tenant's run merged (or, dry, would merge).

    Per-merge edge counts count an edge under each article endpoint it has;
    the tenant totals count each edge once."""

    tenant_id: str
    applied: bool
    merges: List[ArticleMerge] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    edges_repointed: int = 0
    edges_deduped: int = 0
    mentions_added: int = 0
    content_docs_updated: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "applied": self.applied,
            "merges": [m.to_dict() for m in self.merges],
            "skipped": list(self.skipped),
            "edges_repointed": self.edges_repointed,
            "edges_deduped": self.edges_deduped,
            "mentions_added": self.mentions_added,
            "content_docs_updated": self.content_docs_updated,
        }


class UnknownGraphTenantError(LookupError):
    """A requested tenant has no deployed knowledge_graph schema."""


# --------------------------------------------------------------------- #
# Entry point                                                            #
# --------------------------------------------------------------------- #


def merge_article_nodes(
    backend_resolver: Callable[[], Any],
    *,
    tenant_ids: Optional[List[str]] = None,
    apply: bool = False,
) -> List[TenantArticleMergeReport]:
    """Merge article-prefixed node ids into their twins, per tenant.

    ``tenant_ids`` scopes the run; None runs every tenant with a deployed
    knowledge_graph schema. Without ``apply`` nothing is written and the
    reports say what a run would do.
    """
    targets = _graph_targets(backend_resolver, tenant_ids)
    return [
        _TenantMigration(backend_resolver, tenant, graph_schema, content).run(apply)
        for tenant, graph_schema, content in targets
    ]


def _graph_targets(
    backend_resolver: Callable[[], Any], tenant_ids: Optional[List[str]]
) -> List[Tuple[str, str, Dict[str, List[str]]]]:
    """(tenant, graph schema, {content schema: back-ref fields}) per tenant,
    from the schema registry, restricted to schemas deployed in Vespa."""
    with leased_backend(backend_resolver) as backend:
        # Strict: a stale cached registry could hide a tenant's schemas.
        infos = list(backend.schema_registry._get_all_schemas(strict=True))
        deployed = set(
            backend.schema_manager.list_deployed_document_types(raise_on_failure=True)
        )
    infos = [info for info in infos if info.full_schema_name in deployed]
    graphs = {
        info.tenant_id: info.full_schema_name
        for info in infos
        if info.base_schema_name == _GRAPH_BASE_SCHEMA
    }
    if tenant_ids is None:
        tenants = sorted(graphs)
    else:
        tenants = [canonical_tenant_id(t) for t in tenant_ids]
        missing = [t for t in tenants if t not in graphs]
        if missing:
            raise UnknownGraphTenantError(
                f"no deployed {_GRAPH_BASE_SCHEMA} schema for tenant(s) {missing}"
            )
    targets = []
    for tenant in tenants:
        content: Dict[str, List[str]] = {}
        for info in infos:
            if info.tenant_id != tenant or info.base_schema_name == _GRAPH_BASE_SCHEMA:
                continue
            present = [
                name
                for name in _BACKREF_FIELDS
                if re.search(rf"\b{name}\b", info.schema_definition or "")
            ]
            if present:
                content[info.full_schema_name] = present
        targets.append((tenant, graphs[tenant], dict(sorted(content.items()))))
    return targets


# --------------------------------------------------------------------- #
# One tenant                                                             #
# --------------------------------------------------------------------- #


def _edge_key(edge: Dict[str, Any]) -> Tuple:
    return (
        edge.get("source_node_id"),
        edge.get("relation"),
        edge.get("target_node_id"),
        edge.get("segment_id"),
        float(edge.get("ts_start") or 0.0),
        float(edge.get("ts_end") or 0.0),
    )


def _mentions(raw: Optional[str]) -> List[Mention]:
    names = {f.name for f in fields(Mention)}
    return [
        Mention(**{k: v for k, v in m.items() if k in names})
        for m in json.loads(raw or "[]")
    ]


class _TenantMigration:
    def __init__(
        self,
        backend_resolver: Callable[[], Any],
        tenant_id: str,
        graph_schema: str,
        content_schemas: Dict[str, List[str]],
    ) -> None:
        self._resolve = backend_resolver
        self._tenant = tenant_id
        self._graph = graph_schema
        self._content = content_schemas
        self._edge_prefix = f"kg_edge_{_safe_tenant(tenant_id)}_"
        self._http = requests.Session()
        with leased_backend(backend_resolver) as backend:
            self._base = f"{backend._url}:{backend._port}"

    def run(self, apply: bool) -> TenantArticleMergeReport:
        nodes = {
            node_id_from_doc_id(doc_id, self._tenant): {**doc, "doc_id": doc_id}
            for doc_id, doc in self._read_graph("node", _NODE_FIELDS).items()
        }
        nodes.pop("", None)
        edges = {
            doc_id: {**doc, "doc_id": doc_id}
            for doc_id, doc in self._read_graph("edge", _EDGE_FIELDS).items()
        }
        endpoints = {
            node_id
            for edge in edges.values()
            for node_id in (edge.get("source_node_id"), edge.get("target_node_id"))
            if node_id
        }
        plan = plan_article_merges(nodes, endpoints)
        report = TenantArticleMergeReport(
            tenant_id=self._tenant,
            applied=apply,
            merges=[
                ArticleMerge(source_id=a, target_id=b, node_doc=a in nodes)
                for a, b in plan.merges.items()
            ],
            skipped=plan.skipped,
        )
        if not plan.merges:
            return report

        renamed, retired = self._repoint_edges(edges, plan.merges, report, apply)
        self._rewrite_backrefs(plan.merges, renamed, report, apply)
        for doc_id in retired:
            self._write(apply, "delete", doc_id)
        self._merge_node_docs(nodes, plan.merges, report, apply)
        logger.info(
            "Article-node merge for %s (%s): %s",
            self._tenant,
            "applied" if apply else "dry run",
            report.to_dict(),
        )
        return report

    # -- edges -----------------------------------------------------------

    def _repoint_edges(
        self,
        edges: Dict[str, Dict[str, Any]],
        merges: Dict[str, str],
        report: TenantArticleMergeReport,
        apply: bool,
    ) -> Tuple[Dict[str, str], List[str]]:
        """Re-point every edge touching an article id; return the retired
        edges' old->surviving edge ids and their document ids."""
        per_article = {m.source_id: m for m in report.merges}
        index: Dict[Tuple, Tuple[str, Dict[str, Any]]] = {}
        moving = []
        for doc_id in sorted(edges):
            edge = edges[doc_id]
            touched = {
                edge.get("source_node_id"),
                edge.get("target_node_id"),
            } & merges.keys()
            if touched:
                moving.append((doc_id, edge, touched))
            else:
                index[_edge_key(edge)] = (doc_id, edge)

        renamed: Dict[str, str] = {}
        retired: List[str] = []
        for doc_id, edge, touched in moving:
            source = edge.get("source_node_id")
            target = edge.get("target_node_id")
            moved = {
                **edge,
                "source_node_id": merges.get(source, source),
                "target_node_id": merges.get(target, target),
            }
            report.edges_repointed += 1
            for article_id in touched:
                per_article[article_id].edges_repointed += 1

            existing = index.get(_edge_key(moved))
            if existing is None:
                new_doc_id = self._canonical_edge_doc_id(moved)
                moved["doc_id"] = new_doc_id
                self._write(
                    apply,
                    "put",
                    new_doc_id,
                    {k: moved[k] for k in _EDGE_FIELDS if k in moved},
                )
                index[_edge_key(moved)] = (new_doc_id, moved)
                renamed[self._bare(doc_id)] = self._bare(new_doc_id)
                retired.append(doc_id)
            elif carries_same_grounding(existing[1], edge):
                survivor_id, survivor = existing
                merged = merge_edge_provenance(survivor, edge)
                changes = {k: v for k, v in merged.items() if survivor.get(k) != v}
                if changes:
                    self._write(apply, "update", survivor_id, changes)
                    survivor.update(changes)
                report.edges_deduped += 1
                for article_id in touched:
                    per_article[article_id].edges_deduped += 1
                renamed[self._bare(doc_id)] = self._bare(survivor_id)
                retired.append(doc_id)
            else:
                # Its own grounding: re-point it where it stands, so the
                # grounding and every citation of its id survive.
                self._write(
                    apply,
                    "update",
                    doc_id,
                    {
                        "source_node_id": moved["source_node_id"],
                        "target_node_id": moved["target_node_id"],
                    },
                )
        return renamed, retired

    def _canonical_edge_doc_id(self, edge: Dict[str, Any]) -> str:
        return Edge(
            tenant_id=self._tenant,
            source=edge["source_node_id"],
            target=edge["target_node_id"],
            relation=edge.get("relation", ""),
            evidence_span=edge.get("evidence_span", ""),
            segment_id=edge.get("segment_id", ""),
            ts_start=float(edge.get("ts_start") or 0.0),
            ts_end=float(edge.get("ts_end") or 0.0),
            modality=edge.get("modality", ""),
        ).doc_id

    def _bare(self, edge_doc_id: str) -> str:
        return edge_doc_id[len(self._edge_prefix) :]

    # -- content back-refs ----------------------------------------------

    def _rewrite_backrefs(
        self,
        merges: Dict[str, str],
        renamed: Dict[str, str],
        report: TenantArticleMergeReport,
        apply: bool,
    ) -> None:
        mapping = {
            "entity_ids": dict(merges),
            "relation_ids": {old: new for old, new in renamed.items() if old != new},
            "claim_ids": {old: new for old, new in renamed.items() if old != new},
        }
        for schema, present in self._content.items():
            for doc_id, doc in self._citing_docs(schema, present, mapping).items():
                changes = {}
                for name in present:
                    current = list(doc.get(name) or [])
                    rewritten: List[str] = []
                    for value in current:
                        value = mapping[name].get(value, value)
                        if value not in rewritten:
                            rewritten.append(value)
                    if rewritten != current:
                        changes[name] = rewritten
                if not changes:
                    continue
                report.content_docs_updated += 1
                if apply:
                    with leased_backend(self._resolve) as backend:
                        backend.update_document_fields(
                            doc_id,
                            changes,
                            schema_name=schema,
                            namespace=_CONTENT_NAMESPACE,
                            create=False,
                        )

    def _citing_docs(
        self,
        schema: str,
        present: List[str],
        mapping: Dict[str, Dict[str, str]],
    ) -> Dict[str, Dict[str, Any]]:
        terms = [(name, old) for name in present for old in sorted(mapping[name])]
        docs: Dict[str, Dict[str, Any]] = {}
        for start in range(0, len(terms), _IDS_PER_QUERY):
            chunk = terms[start : start + _IDS_PER_QUERY]
            where = " or ".join(
                f"{name} contains {yql_quote(old)}" for name, old in chunk
            )
            for hit in self._query_all(
                f"select documentid, {', '.join(present)} from {schema} where {where}"
            ):
                docs[hit["id"].split("::", 1)[1]] = hit.get("fields", {})
        return docs

    # -- nodes -----------------------------------------------------------

    def _merge_node_docs(
        self,
        nodes: Dict[str, Dict[str, Any]],
        merges: Dict[str, str],
        report: TenantArticleMergeReport,
        apply: bool,
    ) -> None:
        gathered: Dict[str, List[Mention]] = {}
        added: Dict[str, int] = {}
        for article_id, target_id in merges.items():
            if article_id not in nodes:
                continue
            mentions = gathered.setdefault(
                target_id, _mentions(nodes[target_id].get("mentions"))
            )
            count = merge_mentions(
                mentions, _mentions(nodes[article_id].get("mentions"))
            )
            added[target_id] = added.get(target_id, 0) + count
        for target_id, count in added.items():
            if not count:
                continue
            report.mentions_added += count
            self._write(
                apply,
                "update",
                nodes[target_id]["doc_id"],
                {
                    "mentions": json.dumps(
                        [
                            {f.name: getattr(m, f.name) for f in fields(Mention)}
                            for m in gathered[target_id]
                        ]
                    ),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        for article_id in merges:
            if article_id in nodes:
                self._write(apply, "delete", nodes[article_id]["doc_id"])

    # -- Vespa I/O -------------------------------------------------------

    def _write(
        self, apply: bool, op: str, doc_id: str, doc_fields: Optional[Dict] = None
    ) -> None:
        if not apply:
            return
        with leased_backend(self._resolve) as backend:
            if op == "put":
                backend.put_document_fields(
                    doc_id,
                    doc_fields,
                    schema_name=self._graph,
                    namespace=_GRAPH_NAMESPACE,
                )
            elif op == "update":
                backend.update_document_fields(
                    doc_id,
                    doc_fields,
                    schema_name=self._graph,
                    namespace=_GRAPH_NAMESPACE,
                )
            else:
                backend.delete_document_fields(
                    doc_id, schema_name=self._graph, namespace=_GRAPH_NAMESPACE
                )

    def _read_graph(
        self, doc_type: str, wanted: Tuple[str, ...]
    ) -> Dict[str, Dict[str, Any]]:
        """Every ``doc_type`` document of this tenant's graph, by document id."""
        hits = self._query_all(
            f"select {', '.join(wanted)} from {self._graph} "
            f"where tenant_id contains {yql_quote(self._tenant)} "
            f"and doc_type contains {yql_quote(doc_type)} order by doc_id"
        )
        return {hit["fields"]["doc_id"]: hit["fields"] for hit in hits}

    def _query_all(self, yql: str) -> List[Dict[str, Any]]:
        """Every hit of ``yql``, paged; raises unless the pages add up to
        the query's total, so a partial read never passes for a whole one."""
        hits: Dict[str, Dict[str, Any]] = {}
        offset = 0
        while True:
            resp = self._http.post(
                f"{self._base}/search/",
                json={
                    "yql": yql,
                    "hits": _QUERY_PAGE,
                    "offset": offset,
                    "maxHits": _QUERY_PAGE,
                    "maxOffset": offset + _QUERY_PAGE,
                    "ranking": "unranked",
                },
                timeout=_HTTP_TIMEOUT,
            )
            if not resp.ok:
                raise RuntimeError(
                    f"Graph migration query failed ({resp.status_code}): "
                    f"{resp.text[:300]}"
                )
            body = resp.json()
            page = vespa_search_children(body)
            for hit in page:
                hits[hit["id"]] = hit
            total = body.get("root", {}).get("fields", {}).get("totalCount", 0)
            if len(page) < _QUERY_PAGE:
                break
            offset += _QUERY_PAGE
        if len(hits) != total:
            raise RuntimeError(
                f"Graph migration read {len(hits)} of {total} documents for {yql!r}"
            )
        return list(hits.values())
