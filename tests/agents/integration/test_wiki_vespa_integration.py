"""
Integration tests for WikiManager against a real Vespa Docker container.

Starts its own Vespa container, deploys the wiki_pages_test_tenant schema,
exercises the full save→feed→retrieve round-trip, then tears down.

These tests verify that WikiManager correctly writes documents to Vespa
and that the stored content is retrievable via the Document v1 HTTP API.
"""

import json
import platform
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from cogniverse_agents.wiki.wiki_manager import WikiManager
from cogniverse_agents.wiki.wiki_schema import generate_slug
from tests.utils.docker_utils import generate_unique_ports

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TENANT_ID = "test_tenant"
WIKI_SCHEMA = "wiki_pages_test_tenant"
CONTAINER_NAME = "vespa-wiki-integration-tests"

_HTTP_PORT, _CONFIG_PORT = generate_unique_ports(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc_url(port: int, doc_id: str) -> str:
    return (
        f"http://localhost:{port}/document/v1"
        f"/wiki_content/{WIKI_SCHEMA}/docid/{doc_id}"
    )


def _get_vespa_doc(port: int, doc_id: str, retries: int = 15) -> dict | None:
    """Fetch a document from Vespa, retrying to allow for indexing latency."""
    for _ in range(retries):
        try:
            resp = requests.get(_doc_url(port, doc_id), timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        time.sleep(1)
    return None


def _wait_for_config_port(config_port: int, timeout: int = 120) -> bool:
    for _ in range(timeout):
        try:
            resp = requests.get(
                f"http://localhost:{config_port}/ApplicationStatus", timeout=2
            )
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _wait_for_data_port(http_port: int, timeout: int = 120) -> bool:
    for _ in range(timeout):
        try:
            resp = requests.get(
                f"http://localhost:{http_port}/ApplicationStatus", timeout=5
            )
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _wait_for_schema_ready(http_port: int, schema_name: str, timeout: int = 120) -> bool:
    """Feed a minimal probe document to confirm the schema is accepting writes."""
    try:
        resp = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "nomic-embed-text", "input": "readiness"},
            timeout=10,
        )
        embedding = resp.json()["embeddings"][0] if resp.status_code == 200 else [0.01] * 768
    except Exception:
        embedding = [0.01] * 768

    probe = {
        "fields": {
            "doc_id": "readiness_check",
            "tenant_id": "test",
            "page_type": "topic",
            "title": "readiness check",
            "content": "test",
            "slug": "readiness_check",
            "entities": "[]",
            "sources": "[]",
            "cross_references": "[]",
            "update_count": 1,
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
            "embedding": embedding,
        }
    }
    url = f"http://localhost:{http_port}/document/v1/wiki_content/{schema_name}/docid/readiness_check"

    for i in range(timeout):
        try:
            resp = requests.post(url, json=probe, timeout=5)
            if resp.status_code in (200, 201):
                requests.delete(url, timeout=5)
                return True
            if i % 10 == 0:
                print(f"   readiness attempt {i + 1}: {resp.status_code} {resp.text[:80]}")
        except Exception as exc:
            if i % 10 == 0:
                print(f"   readiness attempt {i + 1}: {exc}")
        time.sleep(1)
    return False


def _deploy_wiki_schema(config_port: int, http_port: int) -> None:
    """Deploy the wiki_pages_test_tenant schema via ApplicationPackage."""
    from vespa.package import ApplicationPackage

    from cogniverse_vespa.json_schema_parser import JsonSchemaParser
    from cogniverse_vespa.metadata_schemas import (
        create_adapter_registry_schema,
        create_config_metadata_schema,
        create_organization_metadata_schema,
        create_tenant_metadata_schema,
    )
    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

    metadata_schemas = [
        create_organization_metadata_schema(),
        create_tenant_metadata_schema(),
        create_config_metadata_schema(),
        create_adapter_registry_schema(),
    ]

    parser = JsonSchemaParser()
    schema_file = Path("configs/schemas/wiki_pages_schema.json")
    with open(schema_file) as f:
        schema_json = json.load(f)
    schema_json["name"] = WIKI_SCHEMA
    schema_json["document"]["name"] = WIKI_SCHEMA
    wiki_schema = parser.parse_schema(schema_json)

    app_package = ApplicationPackage(name="cogniverse", schema=metadata_schemas + [wiki_schema])
    mgr = VespaSchemaManager(
        backend_endpoint="http://localhost",
        backend_port=config_port,
    )
    mgr._deploy_package(app_package)


# ---------------------------------------------------------------------------
# Session-scoped fixture: start Vespa, deploy wiki schema, yield port info
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wiki_vespa():
    """Module-scoped real Vespa instance with the wiki_pages schema deployed."""
    http_port = _HTTP_PORT
    config_port = _CONFIG_PORT

    machine = platform.machine().lower()
    docker_platform = "linux/arm64" if machine in ("arm64", "aarch64") else "linux/amd64"

    subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True)
    subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True)

    result = subprocess.run(
        [
            "docker", "run", "-d",
            "--name", CONTAINER_NAME,
            "-p", f"{http_port}:8080",
            "-p", f"{config_port}:19071",
            "--platform", docker_platform,
            "vespaengine/vespa",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to start Vespa container: {result.stderr}")

    print(f"\nVespa container started on http={http_port}, config={config_port}")

    if not _wait_for_config_port(config_port, timeout=120):
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True)
        pytest.fail("Vespa config port did not come up within 120 s")

    time.sleep(10)  # Additional settle time before schema deployment

    try:
        _deploy_wiki_schema(config_port, http_port)
        print("Wiki schema deployed successfully")
    except Exception as exc:
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True)
        pytest.fail(f"Schema deployment failed: {exc}")

    if not _wait_for_data_port(http_port, timeout=120):
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True)
        pytest.fail("Vespa data port did not come up within 120 s after deployment")

    if not _wait_for_schema_ready(http_port, WIKI_SCHEMA, timeout=120):
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True)
        pytest.fail(f"Schema {WIKI_SCHEMA} not ready within 120 s")

    yield {"http_port": http_port, "config_port": config_port}

    subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True)
    subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True)


@pytest.fixture(scope="module")
def wiki_manager(wiki_vespa):
    """WikiManager wired to the real test Vespa instance.

    backend._url/_port are set so _get_document_http and _rebuild_index can
    construct the Vespa Document v1 URL.  backend.search returns [] so
    _rebuild_index gracefully skips index population (full-text search over
    wiki_content is not the focus of these tests).

    _generate_embedding uses the real Ollama nomic-embed-text model when
    available, or the built-in zero-vector fallback otherwise — no mock
    needed.
    """
    http_port = wiki_vespa["http_port"]

    backend = MagicMock()
    backend._url = "http://localhost"
    backend._port = http_port
    backend.search.return_value = []

    manager = WikiManager(
        backend=backend,
        tenant_id=TENANT_ID,
        schema_name=WIKI_SCHEMA,
    )

    yield manager, http_port


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestWikiVespaIntegration:
    def test_save_session_feeds_to_vespa(self, wiki_manager):
        """save_session feeds a session document retrievable via Vespa Document API."""
        manager, port = wiki_manager

        page = manager.save_session(
            query="What is reinforcement learning?",
            response="Reinforcement learning is a type of machine learning.",
            entities=["reinforcement_learning"],
            agent_name="search_agent",
        )

        doc = _get_vespa_doc(port, page.doc_id)

        assert doc is not None, (
            f"Document {page.doc_id} not found in Vespa after save_session. "
            f"GET {_doc_url(port, page.doc_id)}"
        )
        fields = doc.get("fields", {})
        assert fields.get("page_type") == "session"
        assert fields.get("tenant_id") == TENANT_ID
        assert "reinforcement learning" in fields.get("content", "").lower()

    def test_topic_pages_created_for_entities(self, wiki_manager):
        """save_session creates one topic page in Vespa for each entity."""
        manager, port = wiki_manager

        entities = ["neural_network", "gradient_descent"]
        manager.save_session(
            query="Explain neural networks",
            response="Neural networks use gradient descent for training.",
            entities=entities,
            agent_name="search_agent",
        )

        safe = TENANT_ID.replace(":", "_")
        for entity in entities:
            slug = generate_slug(entity)
            doc_id = f"wiki_topic_{safe}_{slug}"
            doc = _get_vespa_doc(port, doc_id)

            assert doc is not None, (
                f"Topic page for entity '{entity}' (doc_id={doc_id}) not found in Vespa."
            )
            fields = doc.get("fields", {})
            assert fields.get("page_type") == "topic"
            assert fields.get("tenant_id") == TENANT_ID

    def test_topic_update_merges_content(self, wiki_manager):
        """Saving two sessions with the same entity merges content on the topic page.

        _get_or_create_topic uses _get_document_http (real Vespa HTTP GET) to
        detect the existing topic — no mock needed for the merge path.
        """
        manager, port = wiki_manager

        entity = "transformer_architecture"
        safe = TENANT_ID.replace(":", "_")
        slug = generate_slug(entity)
        doc_id = f"wiki_topic_{safe}_{slug}"

        # First session — topic page is created fresh.
        manager.save_session(
            query="What is a transformer?",
            response="Transformers use self-attention mechanisms.",
            entities=[entity],
            agent_name="search_agent",
        )

        first_doc = _get_vespa_doc(port, doc_id)
        assert first_doc is not None, f"Topic page {doc_id} not found after first save."
        first_update_count = first_doc.get("fields", {}).get("update_count", 0)

        # Second session — _get_or_create_topic fetches the existing doc via HTTP,
        # then merges and re-feeds.  No mock needed here.
        manager.save_session(
            query="How do transformers handle long sequences?",
            response="Transformers use positional encoding for sequence order.",
            entities=[entity],
            agent_name="search_agent",
        )

        updated_doc = _get_vespa_doc(port, doc_id)
        assert updated_doc is not None, f"Topic page {doc_id} not found after second save."
        updated_content = updated_doc.get("fields", {}).get("content", "")
        updated_update_count = updated_doc.get("fields", {}).get("update_count", 0)

        assert "self-attention" in updated_content, (
            "Merged content missing first session text."
        )
        assert "positional encoding" in updated_content, (
            "Merged content missing second session text."
        )
        assert updated_update_count > first_update_count, (
            f"update_count did not increment: was {first_update_count}, "
            f"now {updated_update_count}"
        )

    def test_delete_page_removes_from_vespa(self, wiki_manager):
        """delete_page removes the document from Vespa so it is no longer retrievable."""
        manager, port = wiki_manager

        manager.save_session(
            query="What is attention mechanism?",
            response="Attention mechanism weights token importance.",
            entities=["attention_mechanism_delete_test"],
            agent_name="search_agent",
        )

        safe = TENANT_ID.replace(":", "_")
        slug = generate_slug("attention_mechanism_delete_test")
        topic_doc_id = f"wiki_topic_{safe}_{slug}"

        # Confirm the topic page exists before deletion.
        doc_before = _get_vespa_doc(port, topic_doc_id)
        assert doc_before is not None, (
            f"Topic page {topic_doc_id} not found in Vespa before delete."
        )

        manager.delete_page(topic_doc_id)

        # After deletion, Vespa should return 404 — poll briefly for convergence.
        for _ in range(10):
            try:
                resp = requests.get(_doc_url(port, topic_doc_id), timeout=5)
                if resp.status_code == 404:
                    return
            except Exception:
                pass
            time.sleep(1)

        pytest.fail(
            f"Document {topic_doc_id} still retrievable from Vespa after delete_page()."
        )
