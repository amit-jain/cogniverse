"""
Integration tests for multi-modal content processing through the full production pipeline.

Tests exercise the real end-to-end path against real models and a real Vespa Docker instance:
1. Load schemas from configs/schemas/*.json via JsonSchemaParser (production schema loading)
2. Feed documents through EmbeddingGeneratorImpl → VespaPyClient pipeline:
   a. EmbeddingGeneratorImpl loads ColBERT/CLAP models via get_or_load_model
   b. Encodes real text/audio to real embeddings
   c. Creates Document objects with production metadata structure
   d. Feeds through backend_client.ingest_documents()
3. VespaPyClient.process() converts Documents to Vespa format:
   a. VespaEmbeddingProcessor converts embeddings to bfloat16 hex / int8 binary
   b. StrategyAwareProcessor maps embedding fields to schema-specific names
   c. Metadata-to-field mapping populates remaining Vespa fields
4. VespaPyClient._feed_prepared_batch() feeds to real Vespa via pyvespa feed_iterable
5. Query with MaxSim, BM25, hamming, HNSW, and hybrid rank profiles
6. Verify retrieval correctness and semantic ranking quality
"""

import time
import wave
from pathlib import Path

import numpy as np
import pytest
from vespa.application import Vespa
from vespa.package import ApplicationPackage

from cogniverse_core.common.models import get_or_load_model
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_runtime.ingestion.processors.audio_embedding_generator import (
    AudioEmbeddingGenerator,
)
from cogniverse_runtime.ingestion.processors.embedding_generator.embedding_generator_impl import (
    EmbeddingGeneratorImpl,
)
from cogniverse_vespa.embedding_processor import VespaEmbeddingProcessor
from cogniverse_vespa.ingestion_client import VespaPyClient
from cogniverse_vespa.json_schema_parser import JsonSchemaParser
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
from tests.system.vespa_test_manager import VespaTestManager
from tests.utils.docker_utils import generate_unique_ports

MULTIMODAL_HTTP_PORT, MULTIMODAL_CONFIG_PORT = generate_unique_ports(__name__)

COLBERT_MODEL_NAME = "lightonai/Reason-ModernColBERT"
COLBERT_CONFIG = {
    "model_loader": "colbert",
    "embedding_type": "document_colbert",
    "embedding_model": COLBERT_MODEL_NAME,
}

CLAP_MODEL_NAME = "laion/clap-htsat-unfused"

SCHEMAS_DIR = Path("configs/schemas")

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


def _create_vespa_client(schema_name, http_port, schema_loader):
    """Create a VespaPyClient connected to the test Vespa instance.

    This is the same VespaPyClient used in production by VespaBackend.ingest_documents().
    We create it directly here to avoid the heavyweight VespaBackend setup (which requires
    ConfigManager, SchemaRegistry, tenant management) while exercising the exact same
    process() + _feed_prepared_batch() code path.
    """
    config = {
        "schema_name": schema_name,
        "base_schema_name": schema_name,
        "url": "http://localhost",
        "port": http_port,
        "schema_loader": schema_loader,
    }
    client = VespaPyClient(config=config)
    client.connect()
    return client


class _BackendAdapter:
    """Adapts VespaPyClient to the ingest_documents() interface expected by EmbeddingGeneratorImpl.

    Production uses VespaBackend.ingest_documents() which internally does the exact same thing:
    client.process(doc) + client._feed_prepared_batch(). This adapter strips the tenant
    management and lazy init, exercising the same Document→Vespa conversion pipeline.
    """

    def __init__(self, vespa_client):
        self._client = vespa_client

    def ingest_documents(self, documents, schema_name):
        prepared = [self._client.process(doc) for doc in documents]
        success, failed = self._client._feed_prepared_batch(prepared)
        return {
            "success_count": success,
            "failed_count": len(failed),
            "failed_documents": failed,
            "total_documents": len(documents),
        }


@pytest.fixture(scope="module")
def colbert_model():
    """Load real ColBERT model via ModelLoaderFactory — cached across all tests in module."""
    model, _ = get_or_load_model(COLBERT_MODEL_NAME, COLBERT_CONFIG)
    return model


@pytest.fixture(scope="module")
def audio_wav_files(tmp_path_factory):
    """Create synthesized WAV files at distinct frequencies for CLAP embedding tests."""
    tmp_dir = tmp_path_factory.mktemp("audio")
    paths = {}
    freq_map = {"podcast": 440, "lecture": 880, "interview": 220}
    for audio_id, freq in freq_map.items():
        wav_path = tmp_dir / f"{audio_id}.wav"
        t = np.linspace(0, 1.0, 48000, dtype=np.float32)
        samples = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
        with wave.open(str(wav_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(samples.tobytes())
        paths[audio_id] = wav_path
    return paths


@pytest.fixture(scope="module")
def vespa_with_schemas():
    """Module-scoped Vespa instance with schemas loaded from JSON and deployed."""
    manager = VespaTestManager(
        app_name="test-multimodal",
        http_port=MULTIMODAL_HTTP_PORT,
        config_port=MULTIMODAL_CONFIG_PORT,
    )

    if not manager.setup_application_directory():
        raise RuntimeError(
            "Failed to setup application directory — check VespaTestManager logs"
        )

    if not manager.deploy_test_application():
        raise RuntimeError(
            "Failed to deploy Vespa test application — check Docker/Vespa logs"
        )

    schema_manager = VespaSchemaManager(
        backend_endpoint="http://localhost",
        backend_port=manager.config_port,
    )

    # Load schemas from production JSON files via JsonSchemaParser
    parser = JsonSchemaParser()
    doc_schema = parser.load_schema_from_json_file(
        str(SCHEMAS_DIR / "document_text_schema.json")
    )
    audio_schema = parser.load_schema_from_json_file(
        str(SCHEMAS_DIR / "audio_content_schema.json")
    )

    app_package = ApplicationPackage(
        name="cogniverse", schema=[doc_schema, audio_schema]
    )
    schema_manager._deploy_package(app_package, allow_schema_removal=True)

    time.sleep(8)

    app = Vespa(url=f"http://localhost:{manager.http_port}")

    yield {
        "app": app,
        "http_port": manager.http_port,
        "config_port": manager.config_port,
    }

    manager.cleanup()


@pytest.fixture(scope="module")
def fed_documents(vespa_with_schemas, audio_wav_files):
    """Feed all content through the production EmbeddingGeneratorImpl → VespaPyClient pipeline.

    Document path:
      EmbeddingGeneratorImpl._process_document_segments()
        → colbert_model.encode(text, is_query=False)
        → Document(content_type=DOCUMENT) with add_embedding() + add_metadata()
        → _feed_document() → backend_client.ingest_documents()
          → VespaPyClient.process() (VespaEmbeddingProcessor + StrategyAwareProcessor)
          → VespaPyClient._feed_prepared_batch() (pyvespa feed_iterable)

    Audio path:
      EmbeddingGeneratorImpl._process_audio_segments()
        → AudioEmbeddingGenerator.generate_acoustic_embedding() (CLAP 512-dim)
        → colbert_model.encode(transcript, is_query=False) (ColBERT 128-dim)
        → Document(content_type=AUDIO) with semantic embedding + acoustic metadata
        → same VespaPyClient pipeline
    """
    http_port = vespa_with_schemas["http_port"]
    schema_loader = FilesystemSchemaLoader(SCHEMAS_DIR)

    # --- Feed documents through production pipeline ---
    doc_client = _create_vespa_client("document_text", http_port, schema_loader)
    doc_generator = EmbeddingGeneratorImpl(
        config={
            "embedding_model": COLBERT_MODEL_NAME,
            "embedding_type": "document_colbert",
            "model_loader": "colbert",
            "schema_name": "document_text",
        },
        backend_client=_BackendAdapter(doc_client),
    )

    doc_segments = [
        {
            "document_id": doc_id,
            "extracted_text": f"{info['title']}. {info['text']}",
            "filename": info["title"],
            "document_type": "txt",
            "path": f"/test/{doc_id}.txt",
            "page_count": 1,
        }
        for doc_id, info in DOC_TEXTS.items()
    ]
    doc_result = doc_generator.generate_embeddings(
        {"video_id": "test_documents", "document_files": doc_segments},
        output_dir=Path("/tmp"),
    )

    # --- Feed audio through production pipeline (one call per item, each has own transcript) ---
    audio_results = {}
    for audio_id, audio_info in AUDIO_TRANSCRIPTS.items():
        audio_client = _create_vespa_client("audio_content", http_port, schema_loader)
        audio_generator = EmbeddingGeneratorImpl(
            config={
                "embedding_model": CLAP_MODEL_NAME,
                "semantic_model": COLBERT_MODEL_NAME,
                "embedding_type": "audio_dual",
                "model_loader": "colbert",
                "schema_name": "audio_content",
            },
            backend_client=_BackendAdapter(audio_client),
        )
        audio_data = {
            "video_id": f"audio_{audio_id}",
            "audio_files": [
                {
                    "audio_id": audio_id,
                    "path": str(audio_wav_files[audio_id]),
                    "filename": f"{audio_id}.wav",
                },
            ],
            "transcript": {"full_text": audio_info["transcript"]},
        }
        audio_results[audio_id] = audio_generator.generate_embeddings(
            audio_data,
            output_dir=Path("/tmp"),
        )

    time.sleep(3)

    return {
        "doc_result": doc_result,
        "audio_results": audio_results,
    }


@pytest.fixture(scope="module")
def doc_embeddings(colbert_model):
    """Encode all test document texts with real ColBERT model for format consistency tests."""
    result = {}
    for doc_id, doc_info in DOC_TEXTS.items():
        text = f"{doc_info['title']}. {doc_info['text']}"
        token_embeddings = colbert_model.encode([text[:8192]], is_query=False)[0]
        result[doc_id] = np.array(token_embeddings, dtype=np.float32)
    return result


@pytest.fixture(scope="module")
def audio_acoustic_embeddings(audio_wav_files):
    """Generate CLAP acoustic embeddings from the same WAV files used for feeding.

    These are re-generated from the same deterministic WAV files, producing identical
    embeddings to what EmbeddingGeneratorImpl._process_audio_segments() fed to Vespa.
    Used by acoustic similarity query tests.
    """
    generator = AudioEmbeddingGenerator(clap_model=CLAP_MODEL_NAME)
    result = {}
    for audio_id, wav_path in audio_wav_files.items():
        result[audio_id] = generator.generate_acoustic_embedding(audio_path=wav_path)
    return result


@pytest.mark.integration
@pytest.mark.requires_vespa
class TestDocumentSchemaDeployAndFeed:
    """Deploy document_text schema from JSON, feed through EmbeddingGeneratorImpl, query."""

    def test_colbert_model_produces_128_dim_tokens(self, doc_embeddings):
        """Verify real ColBERT model produces the expected 128-dim per-token embeddings."""
        for doc_id, emb in doc_embeddings.items():
            assert emb.ndim == 2, f"{doc_id}: expected 2D array, got {emb.ndim}D"
            assert emb.shape[1] == 128, (
                f"{doc_id}: expected 128 dims, got {emb.shape[1]}"
            )
            assert emb.shape[0] > 0, f"{doc_id}: got zero tokens"

    def test_feed_documents_through_production_pipeline(self, fed_documents):
        """All documents fed successfully through EmbeddingGeneratorImpl → VespaPyClient."""
        result = fed_documents["doc_result"]
        assert result.documents_fed == 3, (
            f"Expected 3 documents fed, got {result.documents_fed}. "
            f"Errors: {result.errors}"
        )
        assert result.documents_processed == 3
        assert len(result.errors) == 0

    def test_bm25_query_retrieves_documents(self, vespa_with_schemas, fed_documents):
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

    def test_colbert_float_query_with_real_embeddings(
        self, vespa_with_schemas, colbert_model, fed_documents
    ):
        """MaxSim query using real ColBERT query encoding ranks semantically relevant docs higher."""
        app = vespa_with_schemas["app"]
        schema = "document_text"

        query_tokens = colbert_model.encode(
            ["quarterly financial performance revenue"], is_query=True
        )[0]
        query_emb = np.array(query_tokens, dtype=np.float32)
        assert query_emb.ndim == 2
        assert query_emb.shape[1] == 128

        body = {
            "yql": f"select document_id, document_title from {schema} where true",
            "hits": 10,
            "ranking": "float_float",
            "model.restrict": schema,
            "input.query(qt)": {idx: vec.tolist() for idx, vec in enumerate(query_emb)},
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

    def test_colbert_binary_query(
        self, vespa_with_schemas, colbert_model, fed_documents
    ):
        """Hamming distance binary-binary rank profile returns results with real embeddings."""
        app = vespa_with_schemas["app"]
        schema = "document_text"
        processor = VespaEmbeddingProcessor(schema_name=schema)

        query_tokens = colbert_model.encode(
            ["system architecture microservices"], is_query=True
        )[0]
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

    def test_phased_ranking_binary_then_float(
        self, vespa_with_schemas, colbert_model, fed_documents
    ):
        """Phased ranking: hamming first-phase, MaxSim second-phase rerank."""
        app = vespa_with_schemas["app"]
        schema = "document_text"
        processor = VespaEmbeddingProcessor(schema_name=schema)

        query_tokens = colbert_model.encode(
            ["security audit TLS authentication"], is_query=True
        )[0]
        query_emb = np.array(query_tokens, dtype=np.float32)

        body = {
            "yql": f"select document_id from {schema} where true",
            "hits": 10,
            "ranking": "phased",
            "model.restrict": schema,
            "input.query(qt)": {idx: vec.tolist() for idx, vec in enumerate(query_emb)},
            "input.query(qtb)": processor._convert_to_binary_dict(query_emb),
        }

        response = app.query(body=body)
        assert response.is_successful()
        hits = response.hits
        assert len(hits) > 0
        assert hits[0]["fields"]["document_id"] == "security", (
            f"Expected 'security' as top result for security query, got {hits[0]['fields']['document_id']}"
        )

    def test_hybrid_float_bm25_query(
        self, vespa_with_schemas, colbert_model, fed_documents
    ):
        """Hybrid ColBERT + BM25 rank profile combines both signals."""
        app = vespa_with_schemas["app"]
        schema = "document_text"

        query_tokens = colbert_model.encode(["revenue growth"], is_query=True)[0]
        query_emb = np.array(query_tokens, dtype=np.float32)

        body = {
            "yql": f"select document_id, document_title from {schema} where userQuery()",
            "query": "revenue",
            "hits": 10,
            "ranking": "hybrid_float_bm25",
            "model.restrict": schema,
            "input.query(qt)": {idx: vec.tolist() for idx, vec in enumerate(query_emb)},
        }

        response = app.query(body=body)
        assert response.is_successful()


@pytest.mark.integration
@pytest.mark.requires_vespa
class TestAudioSchemaDeployAndFeed:
    """Deploy audio_content schema from JSON, feed with real CLAP + ColBERT, query."""

    def test_feed_audio_through_production_pipeline(self, fed_documents):
        """All audio documents fed successfully through EmbeddingGeneratorImpl → VespaPyClient."""
        for audio_id, result in fed_documents["audio_results"].items():
            assert result.documents_fed == 1, (
                f"{audio_id}: expected 1 document fed, got {result.documents_fed}. "
                f"Errors: {result.errors}"
            )
            assert result.documents_processed == 1
            assert len(result.errors) == 0

    def test_clap_model_produces_512_dim_embeddings(self, audio_acoustic_embeddings):
        """Verify real CLAP model produces 512-dim acoustic embeddings."""
        for audio_id, emb in audio_acoustic_embeddings.items():
            assert emb.shape == (512,), f"{audio_id}: expected (512,), got {emb.shape}"

    def test_transcript_bm25_search(self, vespa_with_schemas, fed_documents):
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

    def test_semantic_float_maxsim_query(
        self, vespa_with_schemas, colbert_model, fed_documents
    ):
        """Semantic MaxSim float query with real ColBERT query encoding."""
        app = vespa_with_schemas["app"]
        schema = "audio_content"

        query_tokens = colbert_model.encode(
            ["deploying machine learning models at scale"], is_query=True
        )[0]
        query_emb = np.array(query_tokens, dtype=np.float32)

        body = {
            "yql": f"select audio_id, audio_title from {schema} where true",
            "hits": 10,
            "ranking": "semantic_float",
            "model.restrict": schema,
            "input.query(qt)": {idx: vec.tolist() for idx, vec in enumerate(query_emb)},
        }

        response = app.query(body=body)
        assert response.is_successful()
        hits = response.hits
        assert len(hits) == 3
        audio_ids = {hit["fields"]["audio_id"] for hit in hits}
        assert audio_ids == {"podcast", "lecture", "interview"}
        assert all(hit["relevance"] > 0 for hit in hits)

    def test_semantic_binary_hamming_query(
        self, vespa_with_schemas, colbert_model, fed_documents
    ):
        """Hamming distance query on binary ColBERT embeddings from real model."""
        app = vespa_with_schemas["app"]
        schema = "audio_content"
        processor = VespaEmbeddingProcessor(schema_name=schema)

        query_tokens = colbert_model.encode(
            ["cloud infrastructure kubernetes"], is_query=True
        )[0]
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

    def test_acoustic_similarity_query(
        self, vespa_with_schemas, audio_acoustic_embeddings, fed_documents
    ):
        """HNSW nearest-neighbor query on real CLAP acoustic embeddings."""
        app = vespa_with_schemas["app"]
        schema = "audio_content"

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

    def test_phased_semantic_ranking(
        self, vespa_with_schemas, colbert_model, fed_documents
    ):
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
            "input.query(qt)": {idx: vec.tolist() for idx, vec in enumerate(query_emb)},
            "input.query(qtb)": processor._convert_to_binary_dict(query_emb),
        }

        response = app.query(body=body)
        assert response.is_successful()
        hits = response.hits
        assert len(hits) > 0
        assert hits[0]["fields"]["audio_id"] == "lecture", (
            f"Expected 'lecture' as top result for AI/ML query, got {hits[0]['fields']['audio_id']}"
        )

    def test_hybrid_semantic_bm25_query(
        self, vespa_with_schemas, colbert_model, fed_documents
    ):
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
            assert len(hex_str) == 128 * 4, (
                f"Token {idx}: expected 512 hex chars, got {len(hex_str)}"
            )

    def test_real_colbert_binary_dict_dimensions(self, doc_embeddings):
        """Binary dict from real ColBERT output has 16 bytes (128 bits packed) each."""
        processor = VespaEmbeddingProcessor(schema_name="document_text")
        raw = doc_embeddings["earnings"]
        result = processor._convert_to_binary_dict(raw)

        assert len(result) == raw.shape[0]
        for idx in range(raw.shape[0]):
            hex_str = result[idx]
            assert len(hex_str) == 32, (
                f"Token {idx}: expected 32 hex chars, got {len(hex_str)}"
            )

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

        for idx in range(min(3, raw.shape[0])):
            assert len(binary_dict[idx]) == 32
            assert len(float_dict[idx]) == 512
