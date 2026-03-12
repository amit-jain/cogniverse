"""
Integration tests for multi-modal content schema deployment, feeding, and retrieval.

Tests the full round-trip against a real Vespa Docker instance:
1. Deploy document_text and audio_content schemas (built programmatically via pyvespa)
2. Construct embeddings with numpy (matching ColBERT 128-dim multi-vector format)
3. Convert via VespaEmbeddingProcessor to hex bfloat16 / binary
4. Feed documents to Vespa via pyvespa feed_data_point
5. Query with MaxSim, BM25, hamming, and hybrid rank profiles
6. Verify retrieval correctness
"""

import time

import numpy as np
import pytest
from vespa.application import Vespa
from vespa.package import (
    HNSW,
    ApplicationPackage,
    Document,
    Field,
    FieldSet,
    Function,
    RankProfile,
    Schema,
    SecondPhaseRanking,
)

from cogniverse_vespa.embedding_processor import VespaEmbeddingProcessor
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
from tests.system.vespa_test_manager import VespaTestManager
from tests.utils.docker_utils import generate_unique_ports

MULTIMODAL_HTTP_PORT, MULTIMODAL_CONFIG_PORT = generate_unique_ports(__name__)


def _build_document_text_schema() -> Schema:
    """Build document_text schema matching configs/schemas/document_text_schema.json."""
    return Schema(
        name="document_text",
        document=Document(
            fields=[
                Field(name="document_id", type="string",
                      indexing=["summary", "attribute"], attribute=["fast-search"]),
                Field(name="document_title", type="string",
                      indexing=["summary", "index"], index="enable-bm25"),
                Field(name="creation_timestamp", type="long",
                      indexing=["summary", "attribute"], attribute=["fast-search"]),
                Field(name="document_type", type="string",
                      indexing=["summary", "attribute"], attribute=["fast-search"]),
                Field(name="document_path", type="string",
                      indexing=["summary", "attribute"]),
                Field(name="page_count", type="int",
                      indexing=["summary", "attribute"]),
                Field(name="full_text", type="string",
                      indexing=["summary", "index"], index="enable-bm25"),
                Field(name="section_headings", type="string",
                      indexing=["summary", "index"], index="enable-bm25"),
                Field(name="embedding", type="tensor<bfloat16>(token{}, v[128])",
                      indexing=["attribute"]),
                Field(name="embedding_binary", type="tensor<int8>(token{}, v[16])",
                      indexing=["attribute", "index"]),
            ]
        ),
        fieldsets=[FieldSet(name="default", fields=["document_title", "full_text", "section_headings"])],
        rank_profiles=[
            RankProfile(
                name="default",
                inputs=[("query(qtb)", "tensor<int8>(querytoken{}, v[16])")],
                functions=[
                    Function(name="max_sim_hamming",
                             expression="sum(reduce(1/(1+ sum(hamming(query(qtb), attribute(embedding_binary)), v)), max, token), querytoken)")
                ],
                first_phase="max_sim_hamming",
            ),
            RankProfile(
                name="bm25_only",
                first_phase="bm25(document_title) + bm25(full_text) + bm25(section_headings)",
            ),
            RankProfile(
                name="float_float",
                inputs=[("query(qt)", "tensor<float>(querytoken{}, v[128])")],
                functions=[
                    Function(name="max_sim",
                             expression="sum(reduce(sum(query(qt) * cell_cast(attribute(embedding), float), v), max, token), querytoken)")
                ],
                first_phase="max_sim",
            ),
            RankProfile(
                name="binary_binary",
                inputs=[("query(qtb)", "tensor<int8>(querytoken{}, v[16])")],
                functions=[
                    Function(name="max_sim_hamming",
                             expression="sum(reduce(1/(1+ sum(hamming(query(qtb), attribute(embedding_binary)), v)), max, token), querytoken)")
                ],
                first_phase="max_sim_hamming",
            ),
            RankProfile(
                name="phased",
                inputs=[
                    ("query(qtb)", "tensor<int8>(querytoken{}, v[16])"),
                    ("query(qt)", "tensor<float>(querytoken{}, v[128])"),
                ],
                functions=[
                    Function(name="max_sim_hamming",
                             expression="sum(reduce(1/(1+ sum(hamming(query(qtb), attribute(embedding_binary)), v)), max, token), querytoken)"),
                    Function(name="max_sim",
                             expression="sum(reduce(sum(query(qt) * cell_cast(attribute(embedding), float), v), max, token), querytoken)"),
                ],
                first_phase="max_sim_hamming",
                second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=100),
            ),
            RankProfile(
                name="hybrid_float_bm25",
                inputs=[("query(qt)", "tensor<float>(querytoken{}, v[128])")],
                functions=[
                    Function(name="colbert_sim",
                             expression="sum(reduce(sum(query(qt) * cell_cast(attribute(embedding), float), v), max, token), querytoken)"),
                    Function(name="text_score",
                             expression="bm25(document_title) + bm25(full_text) + bm25(section_headings)"),
                ],
                first_phase="colbert_sim",
                second_phase=SecondPhaseRanking(expression="text_score", rerank_count=100),
            ),
        ],
    )


def _build_audio_content_schema() -> Schema:
    """Build audio_content schema matching configs/schemas/audio_content_schema.json."""
    return Schema(
        name="audio_content",
        document=Document(
            fields=[
                Field(name="audio_id", type="string",
                      indexing=["summary", "attribute"], attribute=["fast-search"]),
                Field(name="audio_title", type="string",
                      indexing=["summary", "index"], index="enable-bm25"),
                Field(name="creation_timestamp", type="long",
                      indexing=["summary", "attribute"], attribute=["fast-search"]),
                Field(name="audio_transcript", type="string",
                      indexing=["summary", "index"], index="enable-bm25"),
                Field(name="audio_path", type="string",
                      indexing=["summary", "attribute"]),
                Field(name="audio_duration", type="double",
                      indexing=["summary", "attribute"]),
                Field(name="audio_language", type="string",
                      indexing=["summary", "attribute"]),
                Field(name="acoustic_embedding", type="tensor<float>(v[512])",
                      indexing=["attribute", "index"],
                      ann=HNSW(max_links_per_node=16, neighbors_to_explore_at_insert=200)),
                Field(name="semantic_embedding", type="tensor<bfloat16>(token{}, v[128])",
                      indexing=["attribute"]),
                Field(name="semantic_embedding_binary", type="tensor<int8>(token{}, v[16])",
                      indexing=["attribute", "index"]),
            ]
        ),
        fieldsets=[FieldSet(name="default", fields=["audio_title", "audio_transcript"])],
        rank_profiles=[
            RankProfile(
                name="default",
                inputs=[("query(qtb)", "tensor<int8>(querytoken{}, v[16])")],
                functions=[
                    Function(name="max_sim_hamming",
                             expression="sum(reduce(1/(1+ sum(hamming(query(qtb), attribute(semantic_embedding_binary)), v)), max, token), querytoken)")
                ],
                first_phase="max_sim_hamming",
            ),
            RankProfile(
                name="transcript_search",
                first_phase="bm25(audio_title) + bm25(audio_transcript)",
            ),
            RankProfile(
                name="acoustic_similarity",
                inputs=[("query(acoustic_query)", "tensor<float>(v[512])")],
                first_phase="closeness(field, acoustic_embedding)",
            ),
            RankProfile(
                name="semantic_float",
                inputs=[("query(qt)", "tensor<float>(querytoken{}, v[128])")],
                functions=[
                    Function(name="max_sim",
                             expression="sum(reduce(sum(query(qt) * cell_cast(attribute(semantic_embedding), float), v), max, token), querytoken)")
                ],
                first_phase="max_sim",
            ),
            RankProfile(
                name="semantic_binary",
                inputs=[("query(qtb)", "tensor<int8>(querytoken{}, v[16])")],
                functions=[
                    Function(name="max_sim_hamming",
                             expression="sum(reduce(1/(1+ sum(hamming(query(qtb), attribute(semantic_embedding_binary)), v)), max, token), querytoken)")
                ],
                first_phase="max_sim_hamming",
            ),
            RankProfile(
                name="phased_semantic",
                inputs=[
                    ("query(qtb)", "tensor<int8>(querytoken{}, v[16])"),
                    ("query(qt)", "tensor<float>(querytoken{}, v[128])"),
                ],
                functions=[
                    Function(name="max_sim_hamming",
                             expression="sum(reduce(1/(1+ sum(hamming(query(qtb), attribute(semantic_embedding_binary)), v)), max, token), querytoken)"),
                    Function(name="max_sim",
                             expression="sum(reduce(sum(query(qt) * cell_cast(attribute(semantic_embedding), float), v), max, token), querytoken)"),
                ],
                first_phase="max_sim_hamming",
                second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=100),
            ),
            RankProfile(
                name="hybrid_semantic_bm25",
                inputs=[("query(qtb)", "tensor<int8>(querytoken{}, v[16])")],
                functions=[
                    Function(name="semantic_sim",
                             expression="sum(reduce(1/(1+ sum(hamming(query(qtb), attribute(semantic_embedding_binary)), v)), max, token), querytoken)"),
                    Function(name="text_score",
                             expression="bm25(audio_title) + bm25(audio_transcript)"),
                ],
                first_phase="semantic_sim",
                second_phase=SecondPhaseRanking(expression="text_score", rerank_count=100),
            ),
        ],
    )


@pytest.fixture(scope="module")
def vespa_with_schemas():
    """Module-scoped Vespa instance with document_text + audio_content schemas deployed."""
    manager = VespaTestManager(
        app_name="test-multimodal",
        http_port=MULTIMODAL_HTTP_PORT,
        config_port=MULTIMODAL_CONFIG_PORT,
    )

    if not manager.setup_application_directory():
        raise RuntimeError("Failed to setup application directory — check VespaTestManager logs")

    if not manager.deploy_test_application():
        raise RuntimeError("Failed to deploy Vespa test application — check Docker/Vespa logs")

    # Deploy both content schemas alongside existing metadata schemas
    schema_manager = VespaSchemaManager(
        backend_endpoint="http://localhost",
        backend_port=manager.config_port,
    )

    doc_schema = _build_document_text_schema()
    audio_schema = _build_audio_content_schema()

    app_package = ApplicationPackage(
        name="cogniverse", schema=[doc_schema, audio_schema]
    )
    schema_manager._deploy_package(
        app_package, allow_schema_removal=True
    )

    time.sleep(8)

    app = Vespa(url=f"http://localhost:{manager.http_port}")

    yield {
        "app": app,
        "http_port": manager.http_port,
        "config_port": manager.config_port,
    }

    manager.cleanup()


def make_colbert_embeddings(num_tokens: int, dim: int = 128, seed: int = 42) -> np.ndarray:
    """Construct deterministic ColBERT-style multi-vector embeddings.

    Returns shape (num_tokens, dim) — same format as PyLate ColBERT output.
    """
    rng = np.random.RandomState(seed)
    emb = rng.randn(num_tokens, dim).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


def make_acoustic_embedding(dim: int = 512, seed: int = 99) -> list:
    """Construct a CLAP-style single-vector acoustic embedding as a float list."""
    rng = np.random.RandomState(seed)
    emb = rng.randn(dim).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    return emb.tolist()


@pytest.mark.integration
@pytest.mark.requires_vespa
class TestDocumentSchemaDeployAndFeed:
    """Deploy document_text schema, feed documents with ColBERT embeddings, query."""

    def test_feed_document_with_colbert_embeddings(self, vespa_with_schemas):
        """Feed a document with 128-dim multi-vector embeddings and verify it's stored."""
        app = vespa_with_schemas["app"]
        schema = "document_text"

        raw_emb = make_colbert_embeddings(num_tokens=5, dim=128, seed=10)
        processor = VespaEmbeddingProcessor(schema_name=schema)
        float_dict = processor._convert_to_float_dict(raw_emb)
        binary_dict = processor._convert_to_binary_dict(raw_emb)

        assert isinstance(float_dict, dict)
        assert len(float_dict) == 5
        assert isinstance(binary_dict, dict)
        assert len(binary_dict) == 5

        fields = {
            "document_id": "test_doc_001",
            "document_title": "Quarterly Earnings Report Q3 2025",
            "creation_timestamp": int(time.time()),
            "document_type": "txt",
            "document_path": "/test/quarterly_report.txt",
            "page_count": 1,
            "full_text": "Revenue grew 15% year-over-year driven by enterprise adoption and cloud services expansion.",
            "section_headings": "Executive Summary Financial Highlights",
            "embedding": float_dict,
            "embedding_binary": binary_dict,
        }

        response = app.feed_data_point(schema=schema, data_id="test_doc_001", fields=fields)
        assert response.status_code == 200

    def test_feed_multiple_documents(self, vespa_with_schemas):
        """Feed multiple documents with different embeddings."""
        app = vespa_with_schemas["app"]
        schema = "document_text"
        processor = VespaEmbeddingProcessor(schema_name=schema)

        docs = [
            {
                "id": "doc_design",
                "title": "System Architecture Design",
                "text": "The architecture uses event-driven microservices with message queues for decoupling.",
                "headings": "Architecture Overview Message Queues",
                "seed": 20, "tokens": 8,
            },
            {
                "id": "doc_security",
                "title": "Security Audit Report",
                "text": "All endpoints require TLS 1.3 and OAuth2 bearer tokens for authentication.",
                "headings": "Authentication Encryption",
                "seed": 30, "tokens": 6,
            },
        ]

        for doc in docs:
            raw_emb = make_colbert_embeddings(num_tokens=doc["tokens"], dim=128, seed=doc["seed"])
            fields = {
                "document_id": doc["id"],
                "document_title": doc["title"],
                "creation_timestamp": int(time.time()),
                "document_type": "md",
                "document_path": f"/test/{doc['id']}.md",
                "page_count": 1,
                "full_text": doc["text"],
                "section_headings": doc["headings"],
                "embedding": processor._convert_to_float_dict(raw_emb),
                "embedding_binary": processor._convert_to_binary_dict(raw_emb),
            }
            response = app.feed_data_point(schema=schema, data_id=doc["id"], fields=fields)
            assert response.status_code == 200

    def test_bm25_query_retrieves_documents(self, vespa_with_schemas):
        """BM25 text search retrieves fed documents by content."""
        app = vespa_with_schemas["app"]
        schema = "document_text"
        time.sleep(2)

        body = {
            "yql": f"select document_id, document_title, full_text from {schema} where userQuery()",
            "query": "revenue enterprise",
            "hits": 10,
            "ranking": "bm25_only",
            "model.restrict": schema,
        }

        response = app.query(body=body)
        assert response.is_successful()
        hits = response.hits
        assert len(hits) > 0
        doc_ids = [hit["fields"]["document_id"] for hit in hits]
        assert "test_doc_001" in doc_ids

    def test_colbert_float_query(self, vespa_with_schemas):
        """MaxSim float-float rank profile returns all documents with relevance scores."""
        app = vespa_with_schemas["app"]
        schema = "document_text"

        query_emb = make_colbert_embeddings(num_tokens=3, dim=128, seed=10)
        processor = VespaEmbeddingProcessor(schema_name=schema)

        body = {
            "yql": f"select document_id, document_title from {schema} where true",
            "hits": 10,
            "ranking": "float_float",
            "model.restrict": schema,
            "input.query(qt)": processor._convert_to_float_dict(query_emb),
        }

        response = app.query(body=body)
        assert response.is_successful()
        hits = response.hits
        assert len(hits) == 3
        doc_ids = {hit["fields"]["document_id"] for hit in hits}
        assert doc_ids == {"test_doc_001", "doc_design", "doc_security"}
        assert all(hit["relevance"] > 0 for hit in hits)

    def test_colbert_binary_query(self, vespa_with_schemas):
        """Hamming distance binary-binary rank profile returns results."""
        app = vespa_with_schemas["app"]
        schema = "document_text"

        query_emb = make_colbert_embeddings(num_tokens=3, dim=128, seed=10)
        processor = VespaEmbeddingProcessor(schema_name=schema)

        body = {
            "yql": f"select document_id from {schema} where true",
            "hits": 10,
            "ranking": "binary_binary",
            "model.restrict": schema,
            "input.query(qtb)": processor._convert_to_binary_dict(query_emb),
        }

        response = app.query(body=body)
        assert response.is_successful()
        assert len(response.hits) > 0

    def test_phased_ranking_binary_then_float(self, vespa_with_schemas):
        """Phased ranking: hamming first-phase, MaxSim second-phase rerank."""
        app = vespa_with_schemas["app"]
        schema = "document_text"

        query_emb = make_colbert_embeddings(num_tokens=3, dim=128, seed=10)
        processor = VespaEmbeddingProcessor(schema_name=schema)

        body = {
            "yql": f"select document_id from {schema} where true",
            "hits": 10,
            "ranking": "phased",
            "model.restrict": schema,
            "input.query(qt)": processor._convert_to_float_dict(query_emb),
            "input.query(qtb)": processor._convert_to_binary_dict(query_emb),
        }

        response = app.query(body=body)
        assert response.is_successful()
        assert len(response.hits) > 0

    def test_hybrid_float_bm25_query(self, vespa_with_schemas):
        """Hybrid ColBERT + BM25 rank profile combines both signals."""
        app = vespa_with_schemas["app"]
        schema = "document_text"

        query_emb = make_colbert_embeddings(num_tokens=3, dim=128, seed=10)
        processor = VespaEmbeddingProcessor(schema_name=schema)

        body = {
            "yql": f"select document_id, document_title from {schema} where userQuery()",
            "query": "revenue",
            "hits": 10,
            "ranking": "hybrid_float_bm25",
            "model.restrict": schema,
            "input.query(qt)": processor._convert_to_float_dict(query_emb),
        }

        response = app.query(body=body)
        assert response.is_successful()


@pytest.mark.integration
@pytest.mark.requires_vespa
class TestAudioSchemaDeployAndFeed:
    """Deploy audio_content schema, feed audio docs with CLAP + ColBERT embeddings, query."""

    def test_feed_audio_with_acoustic_and_semantic_embeddings(self, vespa_with_schemas):
        """Feed an audio document with CLAP (512-dim) + ColBERT (128-dim multi-vector)."""
        app = vespa_with_schemas["app"]
        schema = "audio_content"

        acoustic_emb = make_acoustic_embedding(dim=512, seed=50)
        semantic_emb = make_colbert_embeddings(num_tokens=10, dim=128, seed=51)

        processor = VespaEmbeddingProcessor(schema_name=schema)

        fields = {
            "audio_id": "podcast_ep42",
            "audio_title": "Deep Learning in Production Systems",
            "creation_timestamp": int(time.time()),
            "audio_transcript": "Today we discuss deploying deep learning models at scale with GPU orchestration and model serving.",
            "audio_path": "/test/podcast_ep42.mp3",
            "audio_duration": 1847.5,
            "audio_language": "en",
            "acoustic_embedding": acoustic_emb,
            "semantic_embedding": processor._convert_to_float_dict(semantic_emb),
            "semantic_embedding_binary": processor._convert_to_binary_dict(semantic_emb),
        }

        response = app.feed_data_point(schema=schema, data_id="podcast_ep42", fields=fields)
        assert response.status_code == 200

    def test_feed_multiple_audio_documents(self, vespa_with_schemas):
        """Feed multiple audio documents with different content."""
        app = vespa_with_schemas["app"]
        schema = "audio_content"
        processor = VespaEmbeddingProcessor(schema_name=schema)

        audios = [
            {"id": "lecture_ml101", "title": "Introduction to Machine Learning",
             "transcript": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
             "duration": 3600.0, "language": "en", "a_seed": 60, "s_seed": 61, "tokens": 12},
            {"id": "interview_cto", "title": "CTO Interview on Cloud Infrastructure",
             "transcript": "Our cloud infrastructure handles millions of requests using Kubernetes and auto-scaling.",
             "duration": 2100.0, "language": "en", "a_seed": 70, "s_seed": 71, "tokens": 8},
        ]

        for audio in audios:
            acoustic = make_acoustic_embedding(dim=512, seed=audio["a_seed"])
            semantic = make_colbert_embeddings(num_tokens=audio["tokens"], dim=128, seed=audio["s_seed"])
            fields = {
                "audio_id": audio["id"],
                "audio_title": audio["title"],
                "creation_timestamp": int(time.time()),
                "audio_transcript": audio["transcript"],
                "audio_path": f"/test/{audio['id']}.mp3",
                "audio_duration": audio["duration"],
                "audio_language": audio["language"],
                "acoustic_embedding": acoustic,
                "semantic_embedding": processor._convert_to_float_dict(semantic),
                "semantic_embedding_binary": processor._convert_to_binary_dict(semantic),
            }
            response = app.feed_data_point(schema=schema, data_id=audio["id"], fields=fields)
            assert response.status_code == 200

    def test_transcript_bm25_search(self, vespa_with_schemas):
        """BM25 search on audio transcript retrieves matching documents."""
        app = vespa_with_schemas["app"]
        schema = "audio_content"
        time.sleep(2)

        body = {
            "yql": f"select audio_id, audio_title from {schema} where userQuery()",
            "query": "deep learning production GPU",
            "hits": 10,
            "ranking": "transcript_search",
            "model.restrict": schema,
        }

        response = app.query(body=body)
        assert response.is_successful()
        hits = response.hits
        assert len(hits) > 0
        audio_ids = [hit["fields"]["audio_id"] for hit in hits]
        assert "podcast_ep42" in audio_ids

    def test_semantic_float_maxsim_query(self, vespa_with_schemas):
        """Semantic MaxSim float query returns all audio documents with relevance scores."""
        app = vespa_with_schemas["app"]
        schema = "audio_content"

        query_emb = make_colbert_embeddings(num_tokens=4, dim=128, seed=51)
        processor = VespaEmbeddingProcessor(schema_name=schema)

        body = {
            "yql": f"select audio_id, audio_title from {schema} where true",
            "hits": 10,
            "ranking": "semantic_float",
            "model.restrict": schema,
            "input.query(qt)": processor._convert_to_float_dict(query_emb),
        }

        response = app.query(body=body)
        assert response.is_successful()
        hits = response.hits
        assert len(hits) == 3
        audio_ids = {hit["fields"]["audio_id"] for hit in hits}
        assert audio_ids == {"podcast_ep42", "lecture_ml101", "interview_cto"}
        assert all(hit["relevance"] > 0 for hit in hits)

    def test_semantic_binary_hamming_query(self, vespa_with_schemas):
        """Hamming distance query on binary ColBERT embeddings."""
        app = vespa_with_schemas["app"]
        schema = "audio_content"

        query_emb = make_colbert_embeddings(num_tokens=4, dim=128, seed=51)
        processor = VespaEmbeddingProcessor(schema_name=schema)

        body = {
            "yql": f"select audio_id from {schema} where true",
            "hits": 10,
            "ranking": "semantic_binary",
            "model.restrict": schema,
            "input.query(qtb)": processor._convert_to_binary_dict(query_emb),
        }

        response = app.query(body=body)
        assert response.is_successful()
        assert len(response.hits) > 0

    def test_acoustic_similarity_query(self, vespa_with_schemas):
        """HNSW nearest-neighbor query on CLAP acoustic embeddings."""
        app = vespa_with_schemas["app"]
        schema = "audio_content"

        query_acoustic = make_acoustic_embedding(dim=512, seed=50)

        body = {
            "yql": f"select audio_id, audio_title from {schema} where "
                   f"({{targetHits:10}}nearestNeighbor(acoustic_embedding, acoustic_query))",
            "hits": 10,
            "ranking": "acoustic_similarity",
            "model.restrict": schema,
            "input.query(acoustic_query)": query_acoustic,
        }

        response = app.query(body=body)
        assert response.is_successful()
        hits = response.hits
        assert len(hits) > 0
        assert hits[0]["fields"]["audio_id"] == "podcast_ep42"

    def test_phased_semantic_ranking(self, vespa_with_schemas):
        """Phased ranking: hamming first-phase, MaxSim rerank."""
        app = vespa_with_schemas["app"]
        schema = "audio_content"

        query_emb = make_colbert_embeddings(num_tokens=4, dim=128, seed=51)
        processor = VespaEmbeddingProcessor(schema_name=schema)

        body = {
            "yql": f"select audio_id from {schema} where true",
            "hits": 10,
            "ranking": "phased_semantic",
            "model.restrict": schema,
            "input.query(qt)": processor._convert_to_float_dict(query_emb),
            "input.query(qtb)": processor._convert_to_binary_dict(query_emb),
        }

        response = app.query(body=body)
        assert response.is_successful()
        assert len(response.hits) > 0

    def test_hybrid_semantic_bm25_query(self, vespa_with_schemas):
        """Hybrid ColBERT + BM25 combines embedding similarity with text matching."""
        app = vespa_with_schemas["app"]
        schema = "audio_content"

        query_emb = make_colbert_embeddings(num_tokens=4, dim=128, seed=51)
        processor = VespaEmbeddingProcessor(schema_name=schema)

        body = {
            "yql": f"select audio_id, audio_title from {schema} where userQuery()",
            "query": "deep learning",
            "hits": 10,
            "ranking": "hybrid_semantic_bm25",
            "model.restrict": schema,
            "input.query(qtb)": processor._convert_to_binary_dict(query_emb),
        }

        response = app.query(body=body)
        assert response.is_successful()


@pytest.mark.integration
@pytest.mark.requires_vespa
class TestEmbeddingFormatConsistency:
    """Verify that VespaEmbeddingProcessor output format matches schema expectations."""

    def test_colbert_float_dict_dimensions(self):
        """Float dict has correct number of tokens with correct hex length per token."""
        raw = make_colbert_embeddings(num_tokens=7, dim=128, seed=1)
        processor = VespaEmbeddingProcessor(schema_name="document_text")
        result = processor._convert_to_float_dict(raw)

        assert len(result) == 7
        for idx in range(7):
            hex_str = result[idx]
            # 128 dims x 4 hex chars per bfloat16 = 512 hex chars
            assert len(hex_str) == 128 * 4, f"Token {idx}: expected 512 hex chars, got {len(hex_str)}"

    def test_colbert_binary_dict_dimensions(self):
        """Binary dict has correct number of tokens with 16 bytes (128 bits packed) each."""
        raw = make_colbert_embeddings(num_tokens=7, dim=128, seed=1)
        processor = VespaEmbeddingProcessor(schema_name="document_text")
        result = processor._convert_to_binary_dict(raw)

        assert len(result) == 7
        for idx in range(7):
            hex_str = result[idx]
            # 128 dims -> 16 bytes -> 32 hex chars
            assert len(hex_str) == 32, f"Token {idx}: expected 32 hex chars, got {len(hex_str)}"

    def test_acoustic_single_vector_format(self):
        """CLAP 512-dim embedding is a plain float list, not hex dict."""
        emb = make_acoustic_embedding(dim=512, seed=1)
        assert isinstance(emb, list)
        assert len(emb) == 512
        assert all(isinstance(v, float) for v in emb)

    def test_binarization_preserves_sign_information(self):
        """Binary quantization maps positive -> 1, negative -> 0 correctly."""
        padded = np.zeros((2, 128), dtype=np.float32)
        padded[0, :8] = [1.0, -0.5, 0.3, -0.8, 0.0, 0.1, -0.2, 0.9]
        padded[0, 8:] = np.random.RandomState(0).randn(120).astype(np.float32)
        padded[1, :] = np.random.RandomState(1).randn(128).astype(np.float32)

        processor = VespaEmbeddingProcessor(schema_name="document_text")
        binary = processor._convert_to_binary_dict(padded)

        assert isinstance(binary, dict)
        assert len(binary) == 2

        # First byte of token 0: bits for first 8 values
        # positive(1), negative(0), positive(1), negative(0), zero(0), positive(1), negative(0), positive(1)
        # = 10100101 = 0xa5
        first_byte_hex = binary[0][:2]
        assert first_byte_hex == "a5", f"Expected 'a5', got '{first_byte_hex}'"
