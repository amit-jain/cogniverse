"""14-G1 regression: wiki search returns fed pages against real Vespa.

wiki_manager.search passed ``type:"wiki"`` to backend.search, but no "wiki"
profile was registered, so profile-by-type resolution raised and the manager
swallowed it → search always returned []. The fix registers a durable
``wiki_semantic`` profile (type wiki, schema wiki_pages, single-vector). This
deploys the wiki schema the production way (registry, canonical tenant scoping)
and asserts a fed page comes back — the search path the old wiki integration
fixture stubbed away.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

pytestmark = [pytest.mark.integration]


class _FixedEmbedder:
    """Deterministic stand-in for the DenseOn SemanticEmbedder — one fixed
    768-d unit vector, so closeness is constant and BM25 ranks the text."""

    def encode(self, text, is_query=False):
        v = np.ones(768, dtype=np.float32)
        return v / np.linalg.norm(v)


@pytest.fixture()
def wiki_manager(vespa_instance, monkeypatch):
    from cogniverse_agents.wiki import wiki_manager as wm_mod
    from cogniverse_agents.wiki.wiki_manager import WikiManager
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import SystemConfig
    from cogniverse_vespa.config.config_store import VespaConfigStore

    monkeypatch.setattr(
        "cogniverse_core.common.models.semantic_embedder.get_semantic_embedder",
        lambda *a, **k: _FixedEmbedder(),
    )

    http_port = vespa_instance["http_port"]
    config_port = vespa_instance["config_port"]
    tenant_id = f"wiki_search_{uuid.uuid4().hex[:8]}"

    store = VespaConfigStore(backend_url="http://localhost", backend_port=http_port)
    cm = ConfigManager(store=store)
    cm.set_system_config(
        SystemConfig(backend_url="http://localhost", backend_port=http_port)
    )

    BackendRegistry._backend_instances.clear()
    backend = BackendRegistry.get_instance().get_ingestion_backend(
        name="vespa",
        tenant_id=tenant_id,
        config={
            "backend": {
                "url": "http://localhost",
                "config_port": config_port,
                "port": http_port,
            }
        },
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
    )
    backend.schema_registry.deploy_schema(
        tenant_id=tenant_id, base_schema_name="wiki_pages"
    )
    tenant_schema = backend.get_tenant_schema_name(tenant_id, "wiki_pages")

    import requests

    url = f"http://localhost:{http_port}"
    for _ in range(30):
        r = requests.get(
            f"{url}/search/",
            params={"yql": f"select * from {tenant_schema} where true limit 0"},
            timeout=5,
        )
        if r.status_code == 200 and "errors" not in r.json().get("root", {}):
            break
        time.sleep(1)
    else:
        pytest.fail(f"Vespa never activated {tenant_schema}")

    mgr = WikiManager(backend=backend, tenant_id=tenant_id, schema_name=tenant_schema)
    yield mgr, wm_mod
    BackendRegistry._backend_instances.clear()


def test_wiki_search_returns_a_fed_page(wiki_manager):
    mgr, _ = wiki_manager

    mgr.save_session(
        query="what is colpali",
        response="ColPali is a late-interaction visual document retrieval model.",
        entities=["ColPali"],
        agent_name="search_agent",
    )
    mgr.save_session(
        query="what is whisper",
        response="Whisper is an automatic speech recognition model by OpenAI.",
        entities=["Whisper"],
        agent_name="search_agent",
    )
    time.sleep(2)

    results = mgr.search("ColPali visual document retrieval", top_k=5)

    assert results, "wiki search returned [] — the wiki profile did not resolve"
    blob = " ".join(
        f"{r.get('title', '')} {r.get('content', '')}" for r in results
    ).lower()
    assert "colpali" in blob, (
        f"search did not surface the ColPali page; got {results!r}"
    )

    # Semantic recall: the hybrid profile emits nearestNeighbor, so a page is
    # retrievable on embedding closeness alone (the fixed embedder makes every
    # page maximally close). With the dead-closeness ranking this query had
    # zero lexical overlap -> zero hits and 0.0 scores.
    semantic_hits = mgr.search("zzz gibberish nonlexical", top_k=5)
    assert semantic_hits, "nearestNeighbor did not fire — no semantic recall"
    assert semantic_hits[0]["score"] >= 0.5, (
        f"closeness term is dead; hybrid score {semantic_hits[0]['score']}"
    )

    # Index + lint enumerate the fed pages. The rebuild inside save_session
    # ran before Vespa made the feeds searchable, so rebuild again now that
    # visibility has settled — this pins the enumerate -> render -> feed loop
    # (which previously failed encoder resolution and reported zero pages).
    mgr._rebuild_index()
    index_md = mgr.get_index()
    assert index_md is not None
    assert "- **ColPali**" in index_md
    assert "- **Whisper**" in index_md
    assert "_No topics yet._" not in index_md

    time.sleep(2)
    report = mgr.lint()
    assert report["total_pages"] == 5  # 2 topics + 2 sessions + 1 index
    assert report["orphan_pages"] == []  # both topics are cross-referenced
    assert report["issues_found"] == 0
