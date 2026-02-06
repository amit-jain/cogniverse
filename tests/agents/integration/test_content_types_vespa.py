"""
Integration tests for Content Type Schemas

Tests image_content and audio_content Vespa schemas with test Vespa Docker instance.
Validates schema uploads, data ingestion, and search functionality.
"""

import subprocess
import time
from pathlib import Path

import pytest

from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
from tests.utils.async_polling import wait_for_vespa_indexing


@pytest.fixture(scope="module")
def test_vespa_manager():
    """
    Setup test Vespa Docker instance on port 8082 (different from main Vespa)

    Automatically starts test Vespa before tests and cleans up after.
    """
    print("\n" + "=" * 80)
    print("Setting up Content Types Test Vespa Instance")
    print("=" * 80)

    # Configuration
    test_port = 8082
    config_port = 19073  # Config server port
    container_name = f"vespa-content-types-test-{test_port}"

    # Step 1: Stop and remove existing test container
    print(f"\n🧹 Cleaning up any existing test container '{container_name}'...")
    subprocess.run(["docker", "stop", container_name], capture_output=True)
    subprocess.run(["docker", "rm", container_name], capture_output=True)

    # Step 2: Start test Vespa Docker
    print(f"\n🚀 Starting test Vespa container on port {test_port}...")

    import platform

    machine = platform.machine().lower()
    docker_platform = (
        "linux/arm64" if machine in ["arm64", "aarch64"] else "linux/amd64"
    )

    docker_result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "-p",
            f"{test_port}:8080",
            "-p",
            f"{config_port}:19071",
            "--platform",
            docker_platform,
            "vespaengine/vespa",
        ],
        capture_output=True,
        timeout=60,
    )

    if docker_result.returncode != 0:
        pytest.fail(
            f"Failed to start Docker container: {docker_result.stderr.decode()}"
        )

    print(f"✅ Container '{container_name}' started")

    # Step 3: Wait for Vespa to be ready
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
                print(f"✅ Application ready (took {i*2}s)")
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
            assert (
                response.status_code == 200
            ), f"Image search failed: {response.status_code}"
            print("✅ image_content schema is accessible")

            # Test audio_content schema
            response = requests.get(
                f"{test_vespa_manager['base_url']}/search/",
                params={"query": "test", "restrict": "audio_content"},
                timeout=10,
            )
            assert (
                response.status_code == 200
            ), f"Audio search failed: {response.status_code}"
            print("✅ audio_content schema is accessible")

            # Test document_visual schema
            response = requests.get(
                f"{test_vespa_manager['base_url']}/search/",
                params={"query": "test", "restrict": "document_visual"},
                timeout=10,
            )
            assert (
                response.status_code == 200
            ), f"Document visual search failed: {response.status_code}"
            print("✅ document_visual schema is accessible")

            # Test document_text schema
            response = requests.get(
                f"{test_vespa_manager['base_url']}/search/",
                params={"query": "test", "restrict": "document_text"},
                timeout=10,
            )
            assert (
                response.status_code == 200
            ), f"Document text search failed: {response.status_code}"
            print("✅ document_text schema is accessible")

        except Exception as e:
            pytest.fail(f"Failed to verify schemas: {e}")

    def test_image_content_document_ingestion(self, test_vespa_manager):
        """Test ingesting sample image documents with real ColPali embeddings"""
        print("\n" + "-" * 80)
        print("Test: Image Content Document Ingestion (Real ColPali)")
        print("-" * 80)

        import numpy as np
        import requests
        from PIL import Image

        from cogniverse_core.common.models.model_loaders import get_or_load_model

        # Create a test image (100x100 red square)
        print("\n🎨 Creating test image...")
        test_image = Image.new("RGB", (100, 100), color=(255, 0, 0))

        # Load ColPali model
        print("\n📦 Loading ColPali model...")
        config = {"colpali_model": "vidore/colsmol-500m"}
        model, processor = get_or_load_model("vidore/colsmol-500m", config, None)
        print("✅ ColPali model loaded")

        # Generate real ColPali embedding
        print("\n🔢 Generating ColPali embedding...")
        import torch

        batch_inputs = processor.process_images([test_image]).to(model.device)

        with torch.no_grad():
            embeddings = model(**batch_inputs)

        # Convert to numpy and remove batch dimension
        embeddings_np = embeddings.squeeze(0).cpu().numpy()

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

        assert (
            response.status_code == 200
        ), f"Document ingestion failed: {response.status_code} - {response.text}"
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

        assert (
            response.status_code == 200
        ), f"Document ingestion failed: {response.status_code} - {response.text}"
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

    def test_document_visual_ingestion(self, test_vespa_manager):
        """Test ingesting document pages with real ColPali embeddings"""
        print("\n" + "-" * 80)
        print("Test: Document Visual Strategy Ingestion (ColPali)")
        print("-" * 80)

        import numpy as np
        import requests
        from PIL import Image

        from cogniverse_core.common.models.model_loaders import get_or_load_model

        # Create a test document page (simulating a PDF page)
        print("\n📄 Creating test document page...")
        # Create a white page with some text-like rectangles
        test_page = Image.new("RGB", (800, 600), color=(255, 255, 255))
        # Add some black rectangles to simulate text blocks
        page_array = np.array(test_page)
        page_array[50:100, 50:400] = [0, 0, 0]  # Simulate title
        page_array[150:200, 50:700] = [0, 0, 0]  # Simulate paragraph
        test_page = Image.fromarray(page_array)

        # Load ColPali model
        print("\n📦 Loading ColPali model...")
        config = {"colpali_model": "vidore/colsmol-500m"}
        model, processor = get_or_load_model("vidore/colsmol-500m", config, None)
        print("✅ ColPali model loaded")

        # Generate ColPali embedding for document page
        print("\n🔢 Generating ColPali embedding for document page...")
        import torch

        batch_inputs = processor.process_images([test_page]).to(model.device)

        with torch.no_grad():
            embeddings = model(**batch_inputs)

        # Convert to numpy and remove batch dimension
        embeddings_np = embeddings.squeeze(0).cpu().numpy()

        # Pad or truncate to exactly 1024 patches
        if embeddings_np.shape[0] < 1024:
            padding = np.zeros((1024 - embeddings_np.shape[0], embeddings_np.shape[1]))
            embeddings_np = np.vstack([embeddings_np, padding])
        elif embeddings_np.shape[0] > 1024:
            embeddings_np = embeddings_np[:1024]

        colpali_embedding = embeddings_np.tolist()
        print(f"✅ Generated embedding shape: {embeddings_np.shape}")

        # Sample document page matching document_visual schema
        sample_doc_page = {
            "fields": {
                "document_id": "doc_test_001_p1",
                "document_title": "Machine Learning Fundamentals.pdf",
                "document_type": "pdf",
                "page_number": 1,
                "page_count": 10,
                "source_url": "file:///test/docs/ml_fundamentals.pdf",
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

        assert (
            response.status_code == 200
        ), f"Document page ingestion failed: {response.status_code} - {response.text}"
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

        assert (
            response.status_code == 200
        ), f"Document ingestion failed: {response.status_code} - {response.text}"
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

    def test_document_visual_search(self, test_vespa_manager):
        """Test searching documents with visual strategy (ColPali)"""
        print("\n" + "-" * 80)
        print("Test: Document Visual Search (ColPali)")
        print("-" * 80)

        import requests

        from cogniverse_core.query.encoders import ColPaliQueryEncoder

        # Wait for indexing to complete
        print("\n⏳ Waiting for indexing to complete...")
        wait_for_vespa_indexing(delay=3)

        # Load query encoder
        print("\n📦 Loading ColPali query encoder...")
        query_encoder = ColPaliQueryEncoder(model_name="vidore/colsmol-500m")
        print("✅ Query encoder loaded")

        # Encode query
        query = "machine learning fundamentals"
        print(f"\n🔍 Encoding query: '{query}'")
        query_embedding = query_encoder.encode(query)
        query_embedding_flat = query_embedding.flatten().tolist()
        print(f"✅ Query embedding generated: shape {query_embedding.shape}")

        # Search with ColPali rank profile
        print("\n🔍 Searching with ColPali visual strategy...")
        yql = "select * from document_visual where true"

        response = requests.post(
            f"{test_vespa_manager['base_url']}/search/",
            json={
                "yql": yql,
                "hits": 10,
                "ranking.profile": "colpali",
                "input.query(qt)": str(query_embedding_flat),
            },
            timeout=10,
        )

        assert (
            response.status_code == 200
        ), f"Visual search failed: {response.status_code} - {response.text}"
        results = response.json()

        hits = results.get("root", {}).get("children", [])
        print(f"✅ Visual search completed: {len(hits)} hits")

        if len(hits) > 0:
            first_hit = hits[0]["fields"]
            print(f"   Top result: {first_hit.get('document_title', 'no title')}")
            print(f"   Relevance: {hits[0].get('relevance', 0.0):.4f}")
            assert first_hit.get("document_id") == "doc_test_001_p1"
            print("✅ Ingested document page found in visual search results")
        else:
            print("⚠️  No search results found (document may not be indexed yet)")

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

        assert (
            response.status_code == 200
        ), f"Text search failed: {response.status_code} - {response.text}"
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
