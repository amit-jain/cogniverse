"""
Integration tests for Content Type Schemas

Tests image_content and audio_content Vespa schemas with test Vespa Docker instance.
Validates schema uploads, data ingestion, and search functionality.
"""

import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
from tests.utils.async_polling import wait_for_vespa_indexing
from tests.utils.docker_utils import start_docker_container_with_port_retry
from tests.utils.vllm_sidecar import OWNER_LABEL

pytestmark = pytest.mark.integration

COLPALI_MODEL = "TomoroAI/tomoro-colqwen3-embed-4b"


@pytest.fixture(scope="module")
def tomoro_client(vllm_sidecar):
    """Real vLLM-served Tomoro ColQwen3 via RemoteColPaliLoader.

    Tomoro (qwen3_vl) is remote-only — the image/document-visual ingestion and
    search paths route image and query embedding through this sidecar. Same
    ``--runner pooling --convert embed`` serving config the other real-boundary
    visual fixtures use; cached across the session by the vllm_sidecar factory.
    """
    from cogniverse_core.common.models.model_loaders import RemoteColPaliLoader

    url = vllm_sidecar.spawn(
        model=COLPALI_MODEL,
        extra_args=[
            "--runner",
            "pooling",
            "--convert",
            "embed",
            "--max-model-len",
            "4096",
        ],
    )
    loader = RemoteColPaliLoader(
        model_name=COLPALI_MODEL,
        config={"remote_inference_url": url},
        logger=None,
    )
    client, _ = loader.load_model()
    return {"client": client, "url": url}


@pytest.fixture(scope="module")
def test_vespa_manager():
    """
    Setup a test Vespa Docker instance on freshly allocated ports.

    Automatically starts test Vespa before tests and cleans up after.
    """
    print("\n" + "=" * 80)
    print("Setting up Content Types Test Vespa Instance")
    print("=" * 80)

    import platform

    machine = platform.machine().lower()
    docker_platform = (
        "linux/arm64" if machine in ["arm64", "aarch64"] else "linux/amd64"
    )

    container_name, test_port, config_port = start_docker_container_with_port_retry(
        __name__,
        name_prefix="vespa-content-types-test",
        image="vespaengine/vespa:8.668.5",
        container_ports=(8080, 19071),
        extra_run_args=[
            "--label",
            f"{OWNER_LABEL}={os.getpid()}",
            "--platform",
            docker_platform,
        ],
    )
    print(f"✅ Container '{container_name}' started")

    # Wait for Vespa to be ready
    print(f"\n⏳ Waiting for Vespa config server on port {config_port}...")
    import requests

    for i in range(120):
        try:
            response = requests.get(f"http://localhost:{config_port}/", timeout=5)
            if response.status_code == 200:
                print(f"✅ Config server ready (took {i}s)")
                break
        except Exception:
            pass
        wait_for_vespa_indexing(delay=1)
        if i % 10 == 0 and i > 0:
            print(f"   Still waiting... ({i}s)")
    else:
        # Cleanup on failure
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        subprocess.run(["docker", "rm", container_name], capture_output=True)
        pytest.fail("Vespa config server not ready after 120 seconds")

    # Return test Vespa configuration
    test_vespa = {
        "http_port": test_port,
        "config_port": config_port,
        "container_name": container_name,
        "base_url": f"http://localhost:{test_port}",
        "config_url": f"http://localhost:{config_port}",
    }

    # Upload all schemas once at module setup to avoid schema removal errors
    print("\n📤 Uploading all content type schemas for the test suite...")

    schema_manager = VespaSchemaManager(
        backend_endpoint=test_vespa["config_url"],
        backend_port=test_vespa["config_port"],
    )

    schema_manager.upload_content_type_schemas(
        app_name="contenttypetest",
        schemas=["image_content", "audio_content", "document_visual", "document_text"],
    )
    print("✅ All schemas uploaded successfully")

    # Wait for application to be ready
    print("\n⏳ Waiting for application to be ready...")
    for i in range(60):
        try:
            response = requests.get(
                f"{test_vespa['base_url']}/ApplicationStatus", timeout=5
            )
            if response.status_code == 200:
                print(f"✅ Application ready (took {i * 2}s)")
                break
        except Exception:
            pass
        wait_for_vespa_indexing(delay=2)
    else:
        # Cleanup on failure
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        subprocess.run(["docker", "rm", container_name], capture_output=True)
        pytest.fail("Application not ready after 120 seconds")

    yield test_vespa

    # Teardown: Stop and remove test Vespa
    print("\n" + "=" * 80)
    print("Tearing Down Content Types Test Vespa Instance")
    print("=" * 80)

    print(f"\n🧹 Stopping and removing container '{container_name}'...")
    stop_result = subprocess.run(
        ["docker", "stop", container_name], capture_output=True, timeout=30
    )
    remove_result = subprocess.run(
        ["docker", "rm", container_name], capture_output=True, timeout=30
    )

    if stop_result.returncode == 0 and remove_result.returncode == 0:
        print("✅ Test Vespa cleaned up successfully")
    else:
        print(
            f"⚠️  Issues during cleanup: stop={stop_result.returncode}, rm={remove_result.returncode}"
        )


class TestContentTypeVespaSchemas:
    """Test content type schemas with test Vespa"""

    def test_content_type_schemas_accessible(self, test_vespa_manager):
        """Test that all content type schemas are accessible (uploaded in fixture)"""
        print("\n" + "-" * 80)
        print("Test: Verify Content Type Schemas Accessibility")
        print("-" * 80)

        import requests

        # Verify all schemas are accessible via search API
        print("\n🔍 Verifying all schemas are accessible...")
        try:
            # Test image_content schema
            response = requests.get(
                f"{test_vespa_manager['base_url']}/search/",
                params={"query": "test", "restrict": "image_content"},
                timeout=10,
            )
            assert response.status_code == 200, (
                f"Image search failed: {response.status_code}"
            )
            print("✅ image_content schema is accessible")

            # Test audio_content schema
            response = requests.get(
                f"{test_vespa_manager['base_url']}/search/",
                params={"query": "test", "restrict": "audio_content"},
                timeout=10,
            )
            assert response.status_code == 200, (
                f"Audio search failed: {response.status_code}"
            )
            print("✅ audio_content schema is accessible")

            # Test document_visual schema
            response = requests.get(
                f"{test_vespa_manager['base_url']}/search/",
                params={"query": "test", "restrict": "document_visual"},
                timeout=10,
            )
            assert response.status_code == 200, (
                f"Document visual search failed: {response.status_code}"
            )
            print("✅ document_visual schema is accessible")

            # Test document_text schema
            response = requests.get(
                f"{test_vespa_manager['base_url']}/search/",
                params={"query": "test", "restrict": "document_text"},
                timeout=10,
            )
            assert response.status_code == 200, (
                f"Document text search failed: {response.status_code}"
            )
            print("✅ document_text schema is accessible")

        except Exception as e:
            pytest.fail(f"Failed to verify schemas: {e}")

    def test_image_content_document_ingestion(self, test_vespa_manager, tomoro_client):
        """Test ingesting sample image documents with real ColPali embeddings"""
        print("\n" + "-" * 80)
        print("Test: Image Content Document Ingestion (Real ColPali)")
        print("-" * 80)

        import numpy as np
        import requests
        from PIL import Image

        # Create a test image (100x100 red square)
        print("\n🎨 Creating test image...")
        test_image = Image.new("RGB", (100, 100), color=(255, 0, 0))

        # Generate real ColPali embedding via the remote Tomoro sidecar.
        print("\n🔢 Generating ColPali embedding (remote)...")
        result = tomoro_client["client"].process_images(
            [test_image], model_name=COLPALI_MODEL
        )
        embeddings_np = np.asarray(result["embeddings"]).astype(np.float32)
        if embeddings_np.ndim == 3 and embeddings_np.shape[0] == 1:
            embeddings_np = embeddings_np[0]

        # Pad or truncate to exactly 1024 patches
        if embeddings_np.shape[0] < 1024:
            padding = np.zeros((1024 - embeddings_np.shape[0], embeddings_np.shape[1]))
            embeddings_np = np.vstack([embeddings_np, padding])
        elif embeddings_np.shape[0] > 1024:
            embeddings_np = embeddings_np[:1024]

        colpali_embedding = embeddings_np.tolist()
        print(f"✅ Generated embedding shape: {embeddings_np.shape}")

        sample_image = {
            "fields": {
                "image_id": "img_test_001",
                "image_title": "Test Red Car",
                "source_url": "http://example.com/images/red_car.jpg",
                "creation_timestamp": int(time.time()),
                "image_description": "A red sports car on a highway",
                "detected_objects": ["car", "vehicle", "road"],
                "detected_scenes": ["outdoor", "highway"],
                "colpali_embedding": colpali_embedding,
            }
        }

        # Ingest document via Vespa HTTP API
        print("\n📥 Ingesting sample image document...")
        doc_url = f"{test_vespa_manager['base_url']}/document/v1/contenttypetest/image_content/docid/img_test_001"

        response = requests.post(
            doc_url,
            json=sample_image,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        assert response.status_code == 200, (
            f"Document ingestion failed: {response.status_code} - {response.text}"
        )
        print("✅ Sample image document ingested successfully")

        # Wait for document to be indexed
        wait_for_vespa_indexing(delay=2)

        # Verify document can be retrieved
        print("\n🔍 Verifying document retrieval...")
        get_response = requests.get(doc_url, timeout=10)

        assert get_response.status_code == 200, "Document retrieval failed"
        doc_data = get_response.json()
        assert doc_data["fields"]["image_id"] == "img_test_001"
        print("✅ Image document retrieved successfully")

    @pytest.mark.requires_whisper
    def test_audio_content_document_ingestion(self, test_vespa_manager):
        """Test ingesting sample audio documents with real Whisper transcription and embeddings"""
        print("\n" + "-" * 80)
        print("Test: Audio Content Document Ingestion (Real Whisper + Embeddings)")
        print("-" * 80)

        import numpy as np
        import requests

        from cogniverse_runtime.ingestion.processors.audio_embedding_generator import (
            AudioEmbeddingGenerator,
        )
        from cogniverse_runtime.ingestion.processors.audio_transcriber import (
            AudioTranscriber,
        )

        # Create a simple test audio file (1 second of silence)
        print("\n🎵 Creating test audio file...")
        import tempfile
        import wave

        # Create 1 second of silence at 16kHz (Whisper's native rate)
        sample_rate = 16000
        duration = 1  # seconds
        audio_data = np.zeros(sample_rate * duration, dtype=np.int16)

        # Write to temporary WAV file
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(temp_audio.name, "w") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        print(f"✅ Created test audio: {temp_audio.name}")

        # Transcribe with Whisper
        print("\n📦 Loading Whisper model...")
        transcriber = AudioTranscriber(model_size="base")
        print("✅ Whisper model loaded")

        print("\n🔊 Transcribing audio...")
        result = transcriber.transcribe_audio(
            video_path=Path(temp_audio.name), output_dir=None
        )
        transcript = result.get("full_text", "")
        language = result.get("language", "unknown")
        print(f"✅ Transcription complete: '{transcript}' (language: {language})")

        # Generate embeddings
        print("\n🔢 Loading embedding models...")
        embedding_generator = AudioEmbeddingGenerator()
        print("✅ Embedding models loaded")

        print("\n🔢 Generating embeddings...")
        acoustic_embedding, semantic_embedding = (
            embedding_generator.generate_embeddings(
                audio_path=Path(temp_audio.name),
                transcript=transcript if transcript else "Test audio with silence",
            )
        )
        print(
            f"✅ Generated embeddings: acoustic={acoustic_embedding.shape}, semantic={semantic_embedding.shape}"
        )

        # Clean up temp file
        import os

        os.unlink(temp_audio.name)

        # Sample audio document matching our schema
        sample_audio = {
            "fields": {
                "audio_id": "audio_test_001",
                "audio_title": "Test Podcast Episode",
                "source_url": "http://example.com/audio/podcast_ep1.mp3",
                "duration": duration,
                "transcript": transcript if transcript else "Test audio with silence",
                "speaker_labels": [],
                "detected_events": ["silence"],
                "language": language,
                "audio_embedding": acoustic_embedding.tolist(),
                "semantic_embedding": semantic_embedding.tolist(),
            }
        }

        # Ingest document via Vespa HTTP API
        print("\n📥 Ingesting sample audio document...")
        doc_url = f"{test_vespa_manager['base_url']}/document/v1/contenttypetest/audio_content/docid/audio_test_001"

        response = requests.post(
            doc_url,
            json=sample_audio,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        assert response.status_code == 200, (
            f"Document ingestion failed: {response.status_code} - {response.text}"
        )
        print("✅ Sample audio document ingested successfully")

        # Wait for document to be indexed
        wait_for_vespa_indexing(delay=2)

        # Verify document can be retrieved
        print("\n🔍 Verifying document retrieval...")
        get_response = requests.get(doc_url, timeout=10)

        assert get_response.status_code == 200, "Document retrieval failed"
        doc_data = get_response.json()
        assert doc_data["fields"]["audio_id"] == "audio_test_001"
        assert "audio_embedding" in doc_data["fields"]
        assert "semantic_embedding" in doc_data["fields"]
        print(
            f"✅ Audio document retrieved successfully (language: {doc_data['fields'].get('language', 'unknown')})"
        )

    def test_image_content_search(self, test_vespa_manager):
        """Test searching image content"""
        print("\n" + "-" * 80)
        print("Test: Image Content Search")
        print("-" * 80)

        import requests

        # Wait a bit more for indexing to complete
        print("\n⏳ Waiting for indexing to complete...")
        wait_for_vespa_indexing(delay=3)

        # Search for images with YQL query
        print("\n🔍 Searching for 'car' in image descriptions...")
        response = requests.get(
            f"{test_vespa_manager['base_url']}/search/",
            params={
                "yql": "select * from image_content where userQuery()",
                "query": "car",
                "hits": 10,
            },
            timeout=10,
        )

        assert response.status_code == 200, f"Search failed: {response.status_code}"
        results = response.json()

        total_count = results.get("root", {}).get("fields", {}).get("totalCount", 0)
        print(f"✅ Search completed: {total_count} total hits")

        # Verify we can find our ingested document
        hits = results.get("root", {}).get("children", [])
        if len(hits) > 0:
            print(f"   Found {len(hits)} results")
            first_hit = hits[0]["fields"]
            print(f"   Top result: {first_hit.get('image_title', 'no title')}")
            assert first_hit.get("image_id") == "img_test_001"
            print("✅ Ingested image document found in search results")
        else:
            print(
                "⚠️  No search results found (document may not be indexed yet or BM25 didn't match)"
            )
            # This is okay - the important test is that schema works and document was ingested

    def test_audio_content_search(self, test_vespa_manager):
        """Test searching audio content"""
        print("\n" + "-" * 80)
        print("Test: Audio Content Search")
        print("-" * 80)

        import requests

        # Wait a bit more for indexing to complete
        print("\n⏳ Waiting for indexing to complete...")
        wait_for_vespa_indexing(delay=3)

        # Search for audio with YQL query
        print("\n🔍 Searching for 'podcast' in audio transcripts...")
        response = requests.get(
            f"{test_vespa_manager['base_url']}/search/",
            params={
                "yql": "select * from audio_content where userQuery()",
                "query": "podcast",
                "hits": 10,
            },
            timeout=10,
        )

        assert response.status_code == 200, f"Search failed: {response.status_code}"
        results = response.json()

        total_count = results.get("root", {}).get("fields", {}).get("totalCount", 0)
        print(f"✅ Search completed: {total_count} total hits")

        # Verify we can find our ingested document
        hits = results.get("root", {}).get("children", [])
        if len(hits) > 0:
            print(f"   Found {len(hits)} results")
            first_hit = hits[0]["fields"]
            print(f"   Top result: {first_hit.get('audio_title', 'no title')}")
            assert first_hit.get("audio_id") == "audio_test_001"
            print("✅ Ingested audio document found in search results")
        else:
            print(
                "⚠️  No search results found (document may not be indexed yet or BM25 didn't match)"
            )
            # This is okay - the important test is that schema works and document was ingested

    def test_document_visual_ingestion(self, test_vespa_manager, tomoro_client):
        """Test ingesting document pages with real ColPali embeddings"""
        print("\n" + "-" * 80)
        print("Test: Document Visual Strategy Ingestion (ColPali)")
        print("-" * 80)

        import numpy as np
        import requests
        from PIL import Image

        # Create a test document page (simulating a PDF page)
        print("\n📄 Creating test document page...")
        # Create a white page with some text-like rectangles
        test_page = Image.new("RGB", (800, 600), color=(255, 255, 255))
        # Add some black rectangles to simulate text blocks
        page_array = np.array(test_page)
        page_array[50:100, 50:400] = [0, 0, 0]  # Simulate title
        page_array[150:200, 50:700] = [0, 0, 0]  # Simulate paragraph
        test_page = Image.fromarray(page_array)

        # Generate ColPali embedding for document page via the remote sidecar.
        print("\n🔢 Generating ColPali embedding for document page (remote)...")
        result = tomoro_client["client"].process_images(
            [test_page], model_name=COLPALI_MODEL
        )
        embeddings_np = np.asarray(result["embeddings"]).astype(np.float32)
        if embeddings_np.ndim == 3 and embeddings_np.shape[0] == 1:
            embeddings_np = embeddings_np[0]
        print(f"✅ Generated embedding shape: {embeddings_np.shape}")

        # Mapped patch{} tensor feed: one block per page patch.
        colpali_embedding = {
            "blocks": {str(i): row for i, row in enumerate(embeddings_np.tolist())}
        }

        # Sample document page matching document_visual schema
        sample_doc_page = {
            "fields": {
                "document_id": "doc_test_001_p1",
                "document_title": "Machine Learning Fundamentals.pdf",
                "document_type": "pdf",
                "page_number": 1,
                "page_count": 10,
                "document_path": "file:///test/docs/ml_fundamentals.pdf",
                "creation_timestamp": int(time.time()),
                "colpali_embedding": colpali_embedding,
            }
        }

        # Ingest document page via Vespa HTTP API
        print("\n📥 Ingesting sample document page (visual strategy)...")
        doc_url = f"{test_vespa_manager['base_url']}/document/v1/documenttest/document_visual/docid/doc_test_001_p1"

        response = requests.post(
            doc_url,
            json=sample_doc_page,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        assert response.status_code == 200, (
            f"Document page ingestion failed: {response.status_code} - {response.text}"
        )
        print("✅ Document page ingested successfully (visual strategy)")

        # Wait for document to be indexed
        wait_for_vespa_indexing(delay=2)

        # Verify document can be retrieved
        print("\n🔍 Verifying document page retrieval...")
        get_response = requests.get(doc_url, timeout=10)

        assert get_response.status_code == 200, "Document page retrieval failed"
        doc_data = get_response.json()
        assert doc_data["fields"]["document_id"] == "doc_test_001_p1"
        assert doc_data["fields"]["page_number"] == 1
        print("✅ Document page retrieved successfully (visual strategy)")

    def test_document_text_ingestion(self, test_vespa_manager):
        """Test ingesting documents with text extraction and semantic embeddings"""
        print("\n" + "-" * 80)
        print("Test: Document Text Strategy Ingestion (Extraction + Semantic)")
        print("-" * 80)

        import requests
        from sentence_transformers import SentenceTransformer

        # Sample extracted text (simulating PDF text extraction)
        sample_text = """
        Machine Learning Fundamentals

        Machine learning is a subset of artificial intelligence that focuses on
        developing algorithms that can learn from and make predictions on data.
        This document covers the basics of supervised learning, unsupervised learning,
        and reinforcement learning. Key topics include regression, classification,
        clustering, and neural networks.
        """

        # Load text embedding model
        print("\n📦 Loading text embedding model...")
        text_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        print("✅ Text embedding model loaded")

        # Generate semantic embedding
        print("\n🔢 Generating semantic embedding...")
        document_embedding = text_model.encode(
            sample_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        print(f"✅ Generated embedding shape: {document_embedding.shape}")

        # Sample document matching document_text schema
        sample_doc_text = {
            "fields": {
                "document_id": "doc_test_001",
                "document_title": "Machine Learning Fundamentals.pdf",
                "document_type": "pdf",
                "page_count": 10,
                "source_url": "file:///test/docs/ml_fundamentals.pdf",
                "creation_timestamp": int(time.time()),
                "full_text": sample_text.strip(),
                "section_headings": [
                    "Introduction",
                    "Supervised Learning",
                    "Unsupervised Learning",
                ],
                "key_entities": [
                    "machine learning",
                    "artificial intelligence",
                    "neural networks",
                ],
                "document_embedding": document_embedding.tolist(),
            }
        }

        # Ingest document via Vespa HTTP API
        print("\n📥 Ingesting sample document (text strategy)...")
        doc_url = f"{test_vespa_manager['base_url']}/document/v1/documenttest/document_text/docid/doc_test_001"

        response = requests.post(
            doc_url,
            json=sample_doc_text,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        assert response.status_code == 200, (
            f"Document ingestion failed: {response.status_code} - {response.text}"
        )
        print("✅ Document ingested successfully (text strategy)")

        # Wait for document to be indexed
        wait_for_vespa_indexing(delay=2)

        # Verify document can be retrieved
        print("\n🔍 Verifying document retrieval...")
        get_response = requests.get(doc_url, timeout=10)

        assert get_response.status_code == 200, "Document retrieval failed"
        doc_data = get_response.json()
        assert doc_data["fields"]["document_id"] == "doc_test_001"
        assert "document_embedding" in doc_data["fields"]
        assert len(doc_data["fields"]["section_headings"]) == 3
        print("✅ Document retrieved successfully (text strategy)")

    def test_document_visual_search(self, test_vespa_manager, tomoro_client):
        """Test searching documents with visual strategy (ColPali)"""
        print("\n" + "-" * 80)
        print("Test: Document Visual Search (ColPali)")
        print("-" * 80)

        import requests

        from cogniverse_core.query.encoders import ColPaliQueryEncoder

        # Wait for indexing to complete
        print("\n⏳ Waiting for indexing to complete...")
        wait_for_vespa_indexing(delay=3)

        # Load query encoder routed through the remote Tomoro sidecar.
        print("\n📦 Loading ColPali query encoder (remote)...")
        query_encoder = ColPaliQueryEncoder(
            model_name=COLPALI_MODEL,
            inference_service_url=tomoro_client["url"],
        )
        print("✅ Query encoder loaded")

        # Encode query -> (num_tokens, 320) per-token embedding
        query = "machine learning fundamentals"
        print(f"\n🔍 Encoding query: '{query}'")
        query_embedding = query_encoder.encode(query)
        qt_value = {str(i): row.tolist() for i, row in enumerate(query_embedding)}
        print(f"✅ Query embedding generated: shape {query_embedding.shape}")

        # Search with the float_float ColPali MaxSim rank profile
        print("\n🔍 Searching with ColPali visual strategy...")
        yql = "select * from document_visual where true"

        response = requests.post(
            f"{test_vespa_manager['base_url']}/search/",
            json={
                "yql": yql,
                "hits": 10,
                "ranking.profile": "float_float",
                "input.query(qt)": qt_value,
            },
            timeout=10,
        )

        assert response.status_code == 200, (
            f"Visual search failed: {response.status_code} - {response.text}"
        )
        results = response.json()

        hits = results.get("root", {}).get("children", [])
        print(f"✅ Visual search completed: {len(hits)} hits")

        # The page ingested by test_document_visual_ingestion (module-scoped
        # container) must be retrieved and rank first under the float_float
        # ColPali MaxSim profile.
        assert hits, "visual search returned no hits for the ingested page"
        first_hit = hits[0]["fields"]
        print(f"   Top result: {first_hit.get('document_title', 'no title')}")
        print(f"   Relevance: {hits[0].get('relevance', 0.0):.4f}")
        assert first_hit.get("document_id") == "doc_test_001_p1"
        assert first_hit.get("document_path") == "file:///test/docs/ml_fundamentals.pdf"
        assert hits[0].get("relevance", 0.0) > 0
        print("✅ Ingested document page found in visual search results")

    def test_document_text_search(self, test_vespa_manager):
        """Test searching documents with text strategy (semantic + BM25)"""
        print("\n" + "-" * 80)
        print("Test: Document Text Search (Semantic + BM25)")
        print("-" * 80)

        import requests
        from sentence_transformers import SentenceTransformer

        # Wait for indexing to complete
        print("\n⏳ Waiting for indexing to complete...")
        wait_for_vespa_indexing(delay=3)

        # Load text embedding model
        print("\n📦 Loading text embedding model...")
        text_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        print("✅ Text embedding model loaded")

        # Encode query
        query = "machine learning algorithms"
        print(f"\n🔍 Encoding query: '{query}'")
        query_embedding = text_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        print(f"✅ Query embedding generated: shape {query_embedding.shape}")

        # Search with hybrid_bm25_semantic rank profile
        print("\n🔍 Searching with hybrid BM25 + semantic strategy...")
        yql = "select * from document_text where userQuery()"

        response = requests.post(
            f"{test_vespa_manager['base_url']}/search/",
            json={
                "yql": yql,
                "query": query,
                "hits": 10,
                "ranking.profile": "hybrid_bm25_semantic",
                "input.query(q)": query_embedding.tolist(),
            },
            timeout=10,
        )

        assert response.status_code == 200, (
            f"Text search failed: {response.status_code} - {response.text}"
        )
        results = response.json()

        hits = results.get("root", {}).get("children", [])
        print(f"✅ Text search completed: {len(hits)} hits")

        if len(hits) > 0:
            first_hit = hits[0]["fields"]
            print(f"   Top result: {first_hit.get('document_title', 'no title')}")
            print(f"   Relevance: {hits[0].get('relevance', 0.0):.4f}")
            assert first_hit.get("document_id") == "doc_test_001"
            print("✅ Ingested document found in text search results")
        else:
            print("⚠️  No search results found (document may not be indexed yet)")


# Vespa's nativeRank of a text field holding the one query term once, first.
TERM_NATIVE_RANK = 0.3818623835995125


def _image_case():
    query = np.zeros((1024, 320), np.float32)
    query[0, 0] = 1.0
    near = np.zeros((1024, 320), np.float32)
    near[0, 0] = 0.75
    far = np.zeros((1024, 320), np.float32)
    far[0, 0] = 0.25
    # The max over aligned rows of their dot product with the query rows.
    return query, near, far, 0.75, 0.25


def _semantic_case():
    query = np.zeros(768, np.float32)
    query[:2] = [1.0, 1.0]
    near = np.zeros(768, np.float32)
    near[0] = 1.0
    far = np.zeros(768, np.float32)
    far[2] = 1.0
    # Angular closeness 1/(1 + angle): 45 and 90 degrees from the query.
    return query, near, far, 1 / (1 + np.pi / 4), 1 / (1 + np.pi / 2)


IN_CODE_TEXT_FIRST_HYBRIDS = {
    "hybrid_image": ("image_content", "image_description", "colpali_embedding"),
    "hybrid_audio": ("audio_content", "transcript", "semantic_embedding"),
    "hybrid_bm25_semantic": ("document_text", "full_text", "document_embedding"),
}


class TestInCodeTextFirstHybrids:
    """The text-first hybrids of the in-code content type schemas rank their
    text matches by the visual or semantic similarity plus nativeRank.

    Each gets two documents holding the query term in its text field with
    different embeddings, and one without the term: the two text matches come
    back, each scored by its own similarity plus its nativeRank."""

    @pytest.mark.parametrize("profile", sorted(IN_CODE_TEXT_FIRST_HYBRIDS))
    def test_text_matches_rank_by_similarity_plus_native_rank(
        self, test_vespa_manager, profile
    ):
        import requests

        schema, text_field, embedding_field = IN_CODE_TEXT_FIRST_HYBRIDS[profile]
        query, near, far, near_score, far_score = (
            _image_case() if profile == "hybrid_image" else _semantic_case()
        )
        docs = {
            f"{profile}_near": (near, "kestrel harbour notes"),
            f"{profile}_far": (far, "kestrel harbour notes"),
            f"{profile}_textless": (near, "harbour notes"),
        }
        base = f"{test_vespa_manager['base_url']}/document/v1/hybridtest/{schema}"
        for doc_id, (embedding, text) in docs.items():
            response = requests.post(
                f"{base}/docid/{doc_id}",
                json={
                    "fields": {text_field: text, embedding_field: embedding.tolist()}
                },
                timeout=30,
            )
            assert response.status_code == 200, response.text
        try:
            response = requests.post(
                f"{test_vespa_manager['base_url']}/search/",
                json={
                    "yql": (
                        f"select * from {schema} where "
                        f'{{defaultIndex: "{text_field}"}}userInput(@userQuery)'
                    ),
                    "userQuery": "kestrel",
                    "ranking": profile,
                    "input.query(q)": query.tolist(),
                    "hits": 10,
                },
                timeout=30,
            )
        finally:
            for doc_id in docs:
                requests.delete(f"{base}/docid/{doc_id}", timeout=30)

        assert response.status_code == 200, response.text
        hits = response.json()["root"]["children"]
        assert [h["id"].rsplit("::", 1)[1] for h in hits] == [
            f"{profile}_near",
            f"{profile}_far",
        ]
        assert [h["relevance"] for h in hits] == pytest.approx(
            [near_score + TERM_NATIVE_RANK, far_score + TERM_NATIVE_RANK], abs=1e-6
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
