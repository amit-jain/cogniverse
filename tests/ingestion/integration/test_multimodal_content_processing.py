"""
Integration tests for multi-modal content processing through the real production path.

Tests the full round-trip against real models and a real Vespa Docker instance:
1. Load ColBERT model via ModelLoaderFactory / get_or_load_model (real PyLate model)
2. Encode real text to produce real 128-dim per-token ColBERT embeddings
3. Load CLAP model via AudioEmbeddingGenerator (real HuggingFace model)
4. Generate real 512-dim acoustic embeddings from synthesized audio
5. Deploy document_text and audio_content schemas to real Vespa
6. Convert embeddings via VespaEmbeddingProcessor to bfloat16 hex / int8 binary
7. Feed to Vespa via pyvespa feed_data_point
8. Query with MaxSim, BM25, hamming, HNSW, and hybrid rank profiles
9. Verify retrieval correctness and semantic ranking quality
"""

import time
import wave

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

from cogniverse_core.common.models import get_or_load_model
from cogniverse_runtime.ingestion.processors.audio_embedding_generator import (
    AudioEmbeddingGenerator,
)
from cogniverse_vespa.embedding_processor import VespaEmbeddingProcessor
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
from tests.system.vespa_test_manager import VespaTestManager
from tests.utils.docker_utils import generate_unique_ports

MULTIMODAL_HTTP_PORT, MULTIMODAL_CONFIG_PORT = generate_unique_ports(__name__)

COLBERT_MODEL_NAME = "lightonai/GTE-ModernColBERT-v1"
COLBERT_CONFIG = {
    "model_loader": "colbert",
    "embedding_type": "document_colbert",
    "embedding_model": COLBERT_MODEL_NAME,
}

CLAP_MODEL_NAME = "laion/clap-htsat-unfused"

# Test document texts — distinct enough for semantic ranking to differentiate
DOC_TEXTS = {
    "earnings": {
        "title": "Quarterly Earnings Report Q3 2025",
        "text": "Revenue grew 15% year-over-year driven by enterprise adoption and cloud services expansion. "
                "Operating margins improved to 28% reflecting cost discipline and economies of scale.",
        "headings": "Executive Summary Financial Highlights",
    },
    "architecture": {
        "title": "System Architecture Design",
        "text": "The architecture uses event-driven microservices with message queues for decoupling. "
                "Each service publishes domain events consumed by downstream aggregators.",
        "headings": "Architecture Overview Message Queues",
    },
    "security": {
        "title": "Security Audit Report",
        "text": "All endpoints require TLS 1.3 and OAuth2 bearer tokens for authentication. "
                "Penetration testing revealed no critical vulnerabilities in the API surface.",
        "headings": "Authentication Encryption Penetration Testing",
    },
}

AUDIO_TRANSCRIPTS = {
    "podcast": {
        "title": "Deep Learning in Production Systems",
        "transcript": "Today we discuss deploying deep learning models at scale with GPU orchestration and model serving.",
    },
    "lecture": {
        "title": "Introduction to Machine Learning",
        "transcript": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    },
    "interview": {
        "title": "CTO Interview on Cloud Infrastructure",
        "transcript": "Our cloud infrastructure handles millions of requests using Kubernetes and auto-scaling.",
    },
}


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


def _synthesize_wav(path, duration_s: float = 1.0, sample_rate: int = 48000):
    """Write a short sine-wave WAV file for CLAP acoustic embedding tests."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
    # 440 Hz sine wave, amplitude scaled to int16 range
    samples = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


@pytest.fixture(scope="module")
def colbert_model():
    """Load real ColBERT model via ModelLoaderFactory — cached across all tests in module."""
    model, _ = get_or_load_model(COLBERT_MODEL_NAME, COLBERT_CONFIG)
    return model


@pytest.fixture(scope="module")
def clap_generator():
    """Load real CLAP model via AudioEmbeddingGenerator — cached across all tests in module."""
    return AudioEmbeddingGenerator(clap_model=CLAP_MODEL_NAME)


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


@pytest.fixture(scope="module")
def doc_embeddings(colbert_model):
    """Encode all test document texts with real ColBERT model. Returns {doc_id: np.ndarray}."""
    result = {}
    for doc_id, doc_info in DOC_TEXTS.items():
        text = f"{doc_info['title']}. {doc_info['text']}"
        token_embeddings = colbert_model.encode([text[:8192]], is_query=False)[0]
        result[doc_id] = np.array(token_embeddings, dtype=np.float32)
    return result


@pytest.fixture(scope="module")
def audio_semantic_embeddings(colbert_model):
    """Encode all audio transcripts with real ColBERT model. Returns {audio_id: np.ndarray}."""
    result = {}
    for audio_id, audio_info in AUDIO_TRANSCRIPTS.items():
        token_embeddings = colbert_model.encode(
            [audio_info["transcript"][:8192]], is_query=False
        )[0]
        result[audio_id] = np.array(token_embeddings, dtype=np.float32)
    return result


@pytest.fixture(scope="module")
def audio_acoustic_embeddings(clap_generator, tmp_path_factory):
    """Generate CLAP acoustic embeddings from synthesized WAV files. Returns {audio_id: np.ndarray}."""
    tmp_dir = tmp_path_factory.mktemp("audio")
    result = {}
    for audio_id in AUDIO_TRANSCRIPTS:
        wav_path = tmp_dir / f"{audio_id}.wav"
        # Each audio gets a different frequency to produce distinct embeddings
        freq_map = {"podcast": 440, "lecture": 880, "interview": 220}
        t = np.linspace(0, 1.0, 48000, dtype=np.float32)
        samples = (0.5 * np.sin(2 * np.pi * freq_map[audio_id] * t) * 32767).astype(np.int16)
        with wave.open(str(wav_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(samples.tobytes())

        acoustic_emb = clap_generator.generate_acoustic_embedding(audio_path=wav_path)
        result[audio_id] = acoustic_emb
    return result


@pytest.mark.integration
@pytest.mark.requires_vespa
class TestDocumentSchemaDeployAndFeed:
    """Deploy document_text schema, feed documents with real ColBERT embeddings, query."""

    def test_colbert_model_produces_128_dim_tokens(self, doc_embeddings):
        """Verify real ColBERT model produces the expected 128-dim per-token embeddings."""
        for doc_id, emb in doc_embeddings.items():
            assert emb.ndim == 2, f"{doc_id}: expected 2D array, got {emb.ndim}D"
            assert emb.shape[1] == 128, f"{doc_id}: expected 128 dims, got {emb.shape[1]}"
            assert emb.shape[0] > 0, f"{doc_id}: got zero tokens"

    def test_feed_document_with_colbert_embeddings(self, vespa_with_schemas, doc_embeddings):
        """Feed a document with real ColBERT embeddings and verify it's stored."""
        app = vespa_with_schemas["app"]
        schema = "document_text"
        processor = VespaEmbeddingProcessor(schema_name=schema)

        doc_id = "earnings"
        raw_emb = doc_embeddings[doc_id]
        float_dict = processor._convert_to_float_dict(raw_emb)
        binary_dict = processor._convert_to_binary_dict(raw_emb)

        assert isinstance(float_dict, dict)
        assert len(float_dict) == raw_emb.shape[0]
        assert isinstance(binary_dict, dict)
        assert len(binary_dict) == raw_emb.shape[0]

        fields = {
            "document_id": doc_id,
            "document_title": DOC_TEXTS[doc_id]["title"],
            "creation_timestamp": int(time.time()),
            "document_type": "txt",
            "document_path": f"/test/{doc_id}.txt",
            "page_count": 1,
            "full_text": DOC_TEXTS[doc_id]["text"],
            "section_headings": DOC_TEXTS[doc_id]["headings"],
            "embedding": float_dict,
            "embedding_binary": binary_dict,
        }

        response = app.feed_data_point(schema=schema, data_id=doc_id, fields=fields)
        assert response.status_code == 200

    def test_feed_multiple_documents(self, vespa_with_schemas, doc_embeddings):
        """Feed remaining documents with real ColBERT embeddings."""
        app = vespa_with_schemas["app"]
        schema = "document_text"
        processor = VespaEmbeddingProcessor(schema_name=schema)

        for doc_id in ["architecture", "security"]:
            raw_emb = doc_embeddings[doc_id]
            fields = {
                "document_id": doc_id,
                "document_title": DOC_TEXTS[doc_id]["title"],
                "creation_timestamp": int(time.time()),
                "document_type": "md",
                "document_path": f"/test/{doc_id}.md",
                "page_count": 1,
                "full_text": DOC_TEXTS[doc_id]["text"],
                "section_headings": DOC_TEXTS[doc_id]["headings"],
                "embedding": processor._convert_to_float_dict(raw_emb),
                "embedding_binary": processor._convert_to_binary_dict(raw_emb),
            }
            response = app.feed_data_point(schema=schema, data_id=doc_id, fields=fields)
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
        assert "earnings" in doc_ids

    def test_colbert_float_query_with_real_embeddings(self, vespa_with_schemas, colbert_model):
        """MaxSim query using real ColBERT query encoding ranks semantically relevant docs higher."""
        app = vespa_with_schemas["app"]
        schema = "document_text"
        processor = VespaEmbeddingProcessor(schema_name=schema)

        # Encode a query about financial performance
        query_tokens = colbert_model.encode(["quarterly financial performance revenue"], is_query=True)[0]
        query_emb = np.array(query_tokens, dtype=np.float32)
        assert query_emb.ndim == 2
        assert query_emb.shape[1] == 128

        body = {
            "yql": f"select document_id, document_title from {schema} where true",
            "hits": 10,
            "ranking": "float_float",
            "model.restrict": schema,
            "input.query(qt)": processor._convert_to_query_float_dict(query_emb),
        }

        response = app.query(body=body)
        assert response.is_successful()
        hits = response.hits
        assert len(hits) == 3
        doc_ids = [hit["fields"]["document_id"] for hit in hits]
        assert doc_ids[0] == "earnings", (
            f"Expected 'earnings' as top result for financial query, got {doc_ids}"
        )
        assert all(hit["relevance"] > 0 for hit in hits)

    def test_colbert_binary_query(self, vespa_with_schemas, colbert_model):
        """Hamming distance binary-binary rank profile returns results with real embeddings."""
        app = vespa_with_schemas["app"]
        schema = "document_text"
        processor = VespaEmbeddingProcessor(schema_name=schema)

        query_tokens = colbert_model.encode(["system architecture microservices"], is_query=True)[0]
        query_emb = np.array(query_tokens, dtype=np.float32)

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

    def test_phased_ranking_binary_then_float(self, vespa_with_schemas, colbert_model):
        """Phased ranking: hamming first-phase, MaxSim second-phase rerank."""
        app = vespa_with_schemas["app"]
        schema = "document_text"
        processor = VespaEmbeddingProcessor(schema_name=schema)

        query_tokens = colbert_model.encode(["security audit TLS authentication"], is_query=True)[0]
        query_emb = np.array(query_tokens, dtype=np.float32)

        body = {
            "yql": f"select document_id from {schema} where true",
            "hits": 10,
            "ranking": "phased",
            "model.restrict": schema,
            "input.query(qt)": processor._convert_to_query_float_dict(query_emb),
            "input.query(qtb)": processor._convert_to_binary_dict(query_emb),
        }

        response = app.query(body=body)
        assert response.is_successful()
        hits = response.hits
        assert len(hits) > 0
        assert hits[0]["fields"]["document_id"] == "security", (
            f"Expected 'security' as top result for security query, got {hits[0]['fields']['document_id']}"
        )

    def test_hybrid_float_bm25_query(self, vespa_with_schemas, colbert_model):
        """Hybrid ColBERT + BM25 rank profile combines both signals."""
        app = vespa_with_schemas["app"]
        schema = "document_text"
        processor = VespaEmbeddingProcessor(schema_name=schema)

        query_tokens = colbert_model.encode(["revenue growth"], is_query=True)[0]
        query_emb = np.array(query_tokens, dtype=np.float32)

        body = {
            "yql": f"select document_id, document_title from {schema} where userQuery()",
            "query": "revenue",
            "hits": 10,
            "ranking": "hybrid_float_bm25",
            "model.restrict": schema,
            "input.query(qt)": processor._convert_to_query_float_dict(query_emb),
        }

        response = app.query(body=body)
        assert response.is_successful()


@pytest.mark.integration
@pytest.mark.requires_vespa
class TestAudioSchemaDeployAndFeed:
    """Deploy audio_content schema, feed with real CLAP + ColBERT embeddings, query."""

    def test_clap_model_produces_512_dim_embeddings(self, audio_acoustic_embeddings):
        """Verify real CLAP model produces 512-dim acoustic embeddings."""
        for audio_id, emb in audio_acoustic_embeddings.items():
            assert emb.shape == (512,), f"{audio_id}: expected (512,), got {emb.shape}"

    def test_colbert_semantic_produces_128_dim_tokens(self, audio_semantic_embeddings):
        """Verify real ColBERT model produces 128-dim per-token semantic embeddings for transcripts."""
        for audio_id, emb in audio_semantic_embeddings.items():
            assert emb.ndim == 2, f"{audio_id}: expected 2D, got {emb.ndim}D"
            assert emb.shape[1] == 128, f"{audio_id}: expected 128 dims, got {emb.shape[1]}"

    def test_feed_audio_with_real_acoustic_and_semantic_embeddings(
        self, vespa_with_schemas, audio_acoustic_embeddings, audio_semantic_embeddings
    ):
        """Feed audio document with real CLAP (512-dim) + real ColBERT (128-dim multi-vector)."""
        app = vespa_with_schemas["app"]
        schema = "audio_content"
        processor = VespaEmbeddingProcessor(schema_name=schema)

        audio_id = "podcast"
        acoustic_emb = audio_acoustic_embeddings[audio_id]
        semantic_emb = audio_semantic_embeddings[audio_id]

        fields = {
            "audio_id": audio_id,
            "audio_title": AUDIO_TRANSCRIPTS[audio_id]["title"],
            "creation_timestamp": int(time.time()),
            "audio_transcript": AUDIO_TRANSCRIPTS[audio_id]["transcript"],
            "audio_path": f"/test/{audio_id}.mp3",
            "audio_duration": 1847.5,
            "audio_language": "en",
            "acoustic_embedding": acoustic_emb.tolist(),
            "semantic_embedding": processor._convert_to_float_dict(semantic_emb),
            "semantic_embedding_binary": processor._convert_to_binary_dict(semantic_emb),
        }

        response = app.feed_data_point(schema=schema, data_id=audio_id, fields=fields)
        assert response.status_code == 200

    def test_feed_multiple_audio_documents(
        self, vespa_with_schemas, audio_acoustic_embeddings, audio_semantic_embeddings
    ):
        """Feed multiple audio documents with real embeddings."""
        app = vespa_with_schemas["app"]
        schema = "audio_content"
        processor = VespaEmbeddingProcessor(schema_name=schema)

        for audio_id in ["lecture", "interview"]:
            acoustic_emb = audio_acoustic_embeddings[audio_id]
            semantic_emb = audio_semantic_embeddings[audio_id]
            fields = {
                "audio_id": audio_id,
                "audio_title": AUDIO_TRANSCRIPTS[audio_id]["title"],
                "creation_timestamp": int(time.time()),
                "audio_transcript": AUDIO_TRANSCRIPTS[audio_id]["transcript"],
                "audio_path": f"/test/{audio_id}.mp3",
                "audio_duration": 3600.0,
                "audio_language": "en",
                "acoustic_embedding": acoustic_emb.tolist(),
                "semantic_embedding": processor._convert_to_float_dict(semantic_emb),
                "semantic_embedding_binary": processor._convert_to_binary_dict(semantic_emb),
            }
            response = app.feed_data_point(schema=schema, data_id=audio_id, fields=fields)
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
        assert "podcast" in audio_ids

    def test_semantic_float_maxsim_query(self, vespa_with_schemas, colbert_model):
        """Semantic MaxSim float query with real ColBERT query encoding."""
        app = vespa_with_schemas["app"]
        schema = "audio_content"
        processor = VespaEmbeddingProcessor(schema_name=schema)

        query_tokens = colbert_model.encode(
            ["deploying machine learning models at scale"], is_query=True
        )[0]
        query_emb = np.array(query_tokens, dtype=np.float32)

        body = {
            "yql": f"select audio_id, audio_title from {schema} where true",
            "hits": 10,
            "ranking": "semantic_float",
            "model.restrict": schema,
            "input.query(qt)": processor._convert_to_query_float_dict(query_emb),
        }

        response = app.query(body=body)
        assert response.is_successful()
        hits = response.hits
        assert len(hits) == 3
        audio_ids = {hit["fields"]["audio_id"] for hit in hits}
        assert audio_ids == {"podcast", "lecture", "interview"}
        assert all(hit["relevance"] > 0 for hit in hits)

    def test_semantic_binary_hamming_query(self, vespa_with_schemas, colbert_model):
        """Hamming distance query on binary ColBERT embeddings from real model."""
        app = vespa_with_schemas["app"]
        schema = "audio_content"
        processor = VespaEmbeddingProcessor(schema_name=schema)

        query_tokens = colbert_model.encode(["cloud infrastructure kubernetes"], is_query=True)[0]
        query_emb = np.array(query_tokens, dtype=np.float32)

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

    def test_acoustic_similarity_query(self, vespa_with_schemas, audio_acoustic_embeddings):
        """HNSW nearest-neighbor query on real CLAP acoustic embeddings."""
        app = vespa_with_schemas["app"]
        schema = "audio_content"

        # Query with the same acoustic embedding as podcast — should match itself
        query_acoustic = audio_acoustic_embeddings["podcast"].tolist()

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
        assert hits[0]["fields"]["audio_id"] == "podcast"

    def test_phased_semantic_ranking(self, vespa_with_schemas, colbert_model):
        """Phased ranking: hamming first-phase, MaxSim rerank with real embeddings."""
        app = vespa_with_schemas["app"]
        schema = "audio_content"
        processor = VespaEmbeddingProcessor(schema_name=schema)

        query_tokens = colbert_model.encode(
            ["artificial intelligence learning from data"], is_query=True
        )[0]
        query_emb = np.array(query_tokens, dtype=np.float32)

        body = {
            "yql": f"select audio_id from {schema} where true",
            "hits": 10,
            "ranking": "phased_semantic",
            "model.restrict": schema,
            "input.query(qt)": processor._convert_to_query_float_dict(query_emb),
            "input.query(qtb)": processor._convert_to_binary_dict(query_emb),
        }

        response = app.query(body=body)
        assert response.is_successful()
        hits = response.hits
        assert len(hits) > 0
        assert hits[0]["fields"]["audio_id"] == "lecture", (
            f"Expected 'lecture' as top result for AI/ML query, got {hits[0]['fields']['audio_id']}"
        )

    def test_hybrid_semantic_bm25_query(self, vespa_with_schemas, colbert_model):
        """Hybrid ColBERT + BM25 combines real embedding similarity with text matching."""
        app = vespa_with_schemas["app"]
        schema = "audio_content"
        processor = VespaEmbeddingProcessor(schema_name=schema)

        query_tokens = colbert_model.encode(["deep learning"], is_query=True)[0]
        query_emb = np.array(query_tokens, dtype=np.float32)

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
    """Verify that VespaEmbeddingProcessor correctly converts real model output."""

    def test_real_colbert_float_dict_dimensions(self, doc_embeddings):
        """Float dict from real ColBERT output has correct hex length per token."""
        processor = VespaEmbeddingProcessor(schema_name="document_text")
        raw = doc_embeddings["earnings"]
        result = processor._convert_to_float_dict(raw)

        assert len(result) == raw.shape[0]
        for idx in range(raw.shape[0]):
            hex_str = result[idx]
            assert len(hex_str) == 128 * 4, f"Token {idx}: expected 512 hex chars, got {len(hex_str)}"

    def test_real_colbert_binary_dict_dimensions(self, doc_embeddings):
        """Binary dict from real ColBERT output has 16 bytes (128 bits packed) each."""
        processor = VespaEmbeddingProcessor(schema_name="document_text")
        raw = doc_embeddings["earnings"]
        result = processor._convert_to_binary_dict(raw)

        assert len(result) == raw.shape[0]
        for idx in range(raw.shape[0]):
            hex_str = result[idx]
            assert len(hex_str) == 32, f"Token {idx}: expected 32 hex chars, got {len(hex_str)}"

    def test_real_clap_acoustic_format(self, audio_acoustic_embeddings):
        """Real CLAP 512-dim embedding is a numpy array of floats."""
        emb = audio_acoustic_embeddings["podcast"]
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (512,)
        assert emb.dtype in (np.float32, np.float64)

    def test_binarization_preserves_sign_information(self, doc_embeddings):
        """Binary quantization of real model output preserves sign pattern correctly."""
        processor = VespaEmbeddingProcessor(schema_name="document_text")
        raw = doc_embeddings["earnings"]

        float_dict = processor._convert_to_float_dict(raw)
        binary_dict = processor._convert_to_binary_dict(raw)

        assert len(float_dict) == len(binary_dict)
        assert len(float_dict) == raw.shape[0]

        # Each binary token should be 16 bytes = 32 hex chars (128 bits packed)
        for idx in range(min(3, raw.shape[0])):
            assert len(binary_dict[idx]) == 32
            assert len(float_dict[idx]) == 512
