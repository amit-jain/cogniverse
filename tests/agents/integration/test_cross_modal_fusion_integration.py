"""
Integration tests for Cross-Modal Fusion

Tests multi-agent orchestration with cross-modal fusion capabilities.
Spins up test Vespa instance, ingests content across modalities,
executes multi-agent workflows, and validates fusion quality.
"""

import subprocess
import time
from datetime import datetime

import numpy as np
import pytest
import requests
from PIL import Image

from cogniverse_agents.multi_agent_orchestrator import (
    MultiAgentOrchestrator,
)
from cogniverse_agents.workflow_types import (
    TaskStatus,
    WorkflowPlan,
    WorkflowStatus,
    WorkflowTask,
)
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
from tests.utils.async_polling import wait_for_vespa_indexing


@pytest.fixture(scope="module")
def test_vespa_fusion():
    """
    Setup test Vespa Docker instance for cross-modal fusion tests

    Uses port 8083 to avoid conflicts with other test instances.
    """
    print("\n" + "=" * 80)
    print("Setting up Cross-Modal Fusion Test Vespa Instance")
    print("=" * 80)

    # Configuration
    test_port = 8083
    config_port = 19074
    container_name = f"vespa-fusion-test-{test_port}"

    # Step 1: Cleanup existing container
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

    yield test_vespa

    # Teardown: Stop and remove test Vespa
    print("\n" + "=" * 80)
    print("Tearing Down Cross-Modal Fusion Test Vespa Instance")
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


class TestCrossModalFusionIntegration:
    """Integration tests for cross-modal fusion with real Vespa instance"""

    def test_setup_multi_modal_schemas(self, test_vespa_fusion):
        """Test uploading all content type schemas for cross-modal fusion"""
        print("\n" + "-" * 80)
        print("Test: Multi-Modal Schema Setup")
        print("-" * 80)

        schema_manager = VespaSchemaManager(
            backend_endpoint=test_vespa_fusion["config_url"],
            backend_port=test_vespa_fusion["config_port"],
        )

        # Upload all content type schemas
        print("\n📤 Uploading all content type schemas...")
        try:
            schema_manager.upload_content_type_schemas(
                app_name="fusiontest",
                schemas=[
                    "image_content",
                    "audio_content",
                    "document_visual",
                    "document_text",
                ],
            )
            print("✅ All schemas uploaded successfully")
        except Exception as e:
            pytest.fail(f"Failed to upload schemas: {e}")

        # Wait for application to be ready
        print("\n⏳ Waiting for application to be ready...")
        for i in range(60):
            try:
                response = requests.get(
                    f"{test_vespa_fusion['base_url']}/ApplicationStatus", timeout=5
                )
                if response.status_code == 200:
                    print(f"✅ Application ready (took {i*2}s)")
                    break
            except Exception:
                pass
            wait_for_vespa_indexing(delay=2)
        else:
            pytest.fail("Application not ready after 120 seconds")

    def test_ingest_multi_modal_content(self, test_vespa_fusion):
        """Test ingesting content across all modalities"""
        print("\n" + "-" * 80)
        print("Test: Multi-Modal Content Ingestion")
        print("-" * 80)

        import torch
        from sentence_transformers import SentenceTransformer

        from cogniverse_core.common.models.model_loaders import get_or_load_model

        # Load models
        print("\n📦 Loading models...")
        colpali_config = {"colpali_model": "vidore/colsmol-500m"}
        colpali_model, colpali_processor = get_or_load_model(
            "vidore/colsmol-500m", colpali_config, None
        )
        text_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        print("✅ Models loaded")

        # 1. Ingest image content
        print("\n🎨 Ingesting image content...")
        test_image = Image.new(
            "RGB", (100, 100), color=(0, 128, 255)
        )  # Blue robotics image
        batch_inputs = colpali_processor.process_images([test_image]).to(
            colpali_model.device
        )
        with torch.no_grad():
            embeddings = colpali_model(**batch_inputs)
        embeddings_np = embeddings.squeeze(0).cpu().numpy()
        if embeddings_np.shape[0] < 1024:
            padding = np.zeros((1024 - embeddings_np.shape[0], embeddings_np.shape[1]))
            embeddings_np = np.vstack([embeddings_np, padding])
        elif embeddings_np.shape[0] > 1024:
            embeddings_np = embeddings_np[:1024]

        image_doc = {
            "fields": {
                "image_id": "img_robot_001",
                "image_title": "Robot Assembly Diagram",
                "source_url": "http://example.com/images/robot.jpg",
                "creation_timestamp": int(time.time()),
                "image_description": "A blue robot assembly diagram showing mechanical components",
                "detected_objects": ["robot", "machinery"],
                "detected_scenes": ["workshop"],
                "colpali_embedding": embeddings_np.tolist(),
            }
        }

        response = requests.post(
            f"{test_vespa_fusion['base_url']}/document/v1/fusiontest/image_content/docid/img_robot_001",
            json=image_doc,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        assert response.status_code == 200, f"Image ingestion failed: {response.text}"
        print("✅ Image content ingested")

        # 2. Ingest document content (visual)
        print("\n📄 Ingesting document visual content...")
        test_doc_page = Image.new("RGB", (800, 600), color=(255, 255, 255))
        page_array = np.array(test_doc_page)
        page_array[50:100, 50:400] = [0, 0, 0]
        test_doc_page = Image.fromarray(page_array)

        batch_inputs = colpali_processor.process_images([test_doc_page]).to(
            colpali_model.device
        )
        with torch.no_grad():
            embeddings = colpali_model(**batch_inputs)
        embeddings_np = embeddings.squeeze(0).cpu().numpy()
        if embeddings_np.shape[0] < 1024:
            padding = np.zeros((1024 - embeddings_np.shape[0], embeddings_np.shape[1]))
            embeddings_np = np.vstack([embeddings_np, padding])
        elif embeddings_np.shape[0] > 1024:
            embeddings_np = embeddings_np[:1024]

        doc_visual = {
            "fields": {
                "document_id": "doc_robot_001_p1",
                "document_title": "Robotics Engineering Guide.pdf",
                "document_type": "pdf",
                "page_number": 1,
                "page_count": 50,
                "source_url": "file:///docs/robotics.pdf",
                "creation_timestamp": int(time.time()),
                "colpali_embedding": embeddings_np.tolist(),
            }
        }

        response = requests.post(
            f"{test_vespa_fusion['base_url']}/document/v1/fusiontest/document_visual/docid/doc_robot_001_p1",
            json=doc_visual,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        assert (
            response.status_code == 200
        ), f"Document visual ingestion failed: {response.text}"
        print("✅ Document visual content ingested")

        # 3. Ingest document content (text)
        print("\n📝 Ingesting document text content...")
        doc_text = """
        Robotics Engineering Guide

        Robotics combines mechanical engineering, electrical engineering, and computer science
        to design and build robots. Modern robotics involves advanced control systems,
        machine learning algorithms, and sensor fusion techniques. Key topics include
        kinematics, dynamics, motion planning, and autonomous navigation.
        """

        doc_embedding = text_model.encode(
            doc_text, convert_to_numpy=True, normalize_embeddings=True
        )

        doc_text_doc = {
            "fields": {
                "document_id": "doc_robot_001",
                "document_title": "Robotics Engineering Guide.pdf",
                "document_type": "pdf",
                "page_count": 50,
                "source_url": "file:///docs/robotics.pdf",
                "creation_timestamp": int(time.time()),
                "full_text": doc_text.strip(),
                "section_headings": [
                    "Introduction",
                    "Control Systems",
                    "Machine Learning",
                ],
                "key_entities": ["robotics", "engineering", "machine learning"],
                "document_embedding": doc_embedding.tolist(),
            }
        }

        response = requests.post(
            f"{test_vespa_fusion['base_url']}/document/v1/fusiontest/document_text/docid/doc_robot_001",
            json=doc_text_doc,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        assert (
            response.status_code == 200
        ), f"Document text ingestion failed: {response.text}"
        print("✅ Document text content ingested")

        # Wait for indexing
        print("\n⏳ Waiting for indexing to complete...")
        wait_for_vespa_indexing(delay=5)
        print("✅ Indexing complete")

    def test_cross_modal_fusion_workflow(
        self, test_vespa_fusion, telemetry_manager_without_phoenix
    ):
        """Test multi-agent workflow with cross-modal fusion"""
        print("\n" + "-" * 80)
        print("Test: Cross-Modal Fusion Workflow")
        print("-" * 80)

        # Create orchestrator (will use mocked agents)
        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            telemetry_config=telemetry_manager_without_phoenix.config,
            enable_workflow_intelligence=False,
        )

        # Create mock workflow with completed tasks from different modalities
        print("\n🎭 Creating mock multi-modal workflow...")
        tasks = []

        # Video search task
        video_task = WorkflowTask(
            task_id="task_video",
            agent_name="video_search_agent",
            query="Find videos about robotics",
            dependencies=set(),
        )
        video_task.status = TaskStatus.COMPLETED
        video_task.start_time = datetime.now()
        video_task.end_time = datetime.now()
        video_task.result = {
            "videos": [{"title": "Robot Assembly Tutorial", "score": 0.92}],
            "count": 1,
            "confidence": 0.92,
        }
        tasks.append(video_task)

        # Image search task
        image_task = WorkflowTask(
            task_id="task_image",
            agent_name="image_search_agent",
            query="Find images about robotics",
            dependencies=set(),
        )
        image_task.status = TaskStatus.COMPLETED
        image_task.start_time = datetime.now()
        image_task.end_time = datetime.now()
        image_task.result = {
            "images": [{"title": "Robot Assembly Diagram", "score": 0.88}],
            "count": 1,
            "confidence": 0.88,
        }
        tasks.append(image_task)

        # Document search task
        doc_task = WorkflowTask(
            task_id="task_document",
            agent_name="document_agent",
            query="Find documents about robotics",
            dependencies=set(),
        )
        doc_task.status = TaskStatus.COMPLETED
        doc_task.start_time = datetime.now()
        doc_task.end_time = datetime.now()
        doc_task.result = {
            "documents": [{"title": "Robotics Engineering Guide", "score": 0.85}],
            "count": 1,
            "confidence": 0.85,
        }
        tasks.append(doc_task)

        # Create workflow plan
        workflow_plan = WorkflowPlan(
            workflow_id="fusion_test_workflow",
            original_query="Find comprehensive information about robotics",
            status=WorkflowStatus.COMPLETED,
            tasks=tasks,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        # Test cross-modal fusion aggregation
        print("\n🔀 Testing cross-modal fusion...")
        import asyncio

        result = asyncio.run(orchestrator._aggregate_results(workflow_plan))

        # Validate fusion results
        print("\n✅ Validating fusion results...")
        assert "aggregated_content" in result
        assert "confidence" in result
        assert "fusion_strategy" in result
        assert "fusion_quality" in result
        assert "cross_modal_consistency" in result
        assert "modality_coverage" in result

        # Check modality coverage
        assert "video" in result["modality_coverage"]
        assert "image" in result["modality_coverage"]
        assert "document" in result["modality_coverage"]
        print(f"   Modality coverage: {result['modality_coverage']}")

        # Check fusion strategy was selected
        fusion_strategy = result["fusion_strategy"]
        print(f"   Fusion strategy: {fusion_strategy}")
        assert fusion_strategy in [
            "score",
            "temporal",
            "semantic",
            "hierarchical",
            "simple",
        ]

        # Check fusion quality metrics
        quality = result["fusion_quality"]
        print(f"   Fusion quality: {quality['overall_quality']:.2f}")
        assert "coverage" in quality
        assert "consistency" in quality
        assert "coherence" in quality
        assert "redundancy" in quality
        assert "complementarity" in quality
        assert 0.0 <= quality["overall_quality"] <= 1.0

        # Check cross-modal consistency
        consistency = result["cross_modal_consistency"]
        print(f"   Consistency score: {consistency['consistency_score']:.2f}")
        assert "consistency_score" in consistency
        assert "modality_count" in consistency
        assert consistency["modality_count"] == 3

        print("✅ Cross-modal fusion validated successfully")

    def test_fusion_strategies(
        self, test_vespa_fusion, telemetry_manager_without_phoenix
    ):
        """Test different fusion strategies"""
        print("\n" + "-" * 80)
        print("Test: Fusion Strategies")
        print("-" * 80)

        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            telemetry_config=telemetry_manager_without_phoenix.config,
            enable_workflow_intelligence=False,
        )

        # Create sample task results
        task_results = {
            "task_1": {
                "agent": "video_search_agent",
                "modality": "video",
                "query": "test",
                "result": {"content": "Video results"},
                "execution_time": 1.5,
                "confidence": 0.9,
            },
            "task_2": {
                "agent": "image_search_agent",
                "modality": "image",
                "query": "test",
                "result": {"content": "Image results"},
                "execution_time": 1.2,
                "confidence": 0.85,
            },
        }

        # Test score-based fusion
        print("\n📊 Testing score-based fusion...")
        result = orchestrator._fuse_by_score(task_results)
        assert result["confidence"] > 0
        assert "VIDEO" in result["content"]
        print("✅ Score-based fusion works")

        # Test temporal fusion
        print("\n⏰ Testing temporal fusion...")
        result = orchestrator._fuse_by_temporal_alignment(task_results)
        assert result["confidence"] > 0
        # IMAGE should come before VIDEO (shorter execution time)
        assert result["content"].find("IMAGE") < result["content"].find("VIDEO")
        print("✅ Temporal fusion works")

        # Test semantic fusion
        print("\n🔤 Testing semantic fusion...")
        import asyncio

        result = asyncio.run(
            orchestrator._fuse_by_semantic_similarity(task_results, "test query")
        )
        assert result["confidence"] > 0
        print("✅ Semantic fusion works")

        # Test hierarchical fusion
        print("\n🏗️ Testing hierarchical fusion...")
        agent_modalities = {"task_1": "video", "task_2": "image"}
        result = orchestrator._fuse_hierarchically(task_results, agent_modalities)
        assert result["confidence"] > 0
        assert "## VIDEO RESULTS" in result["content"]
        assert "## IMAGE RESULTS" in result["content"]
        print("✅ Hierarchical fusion works")

        print("\n✅ All fusion strategies validated")

    def test_fusion_quality_metrics(
        self, test_vespa_fusion, telemetry_manager_without_phoenix
    ):
        """Test fusion quality metrics calculation"""
        print("\n" + "-" * 80)
        print("Test: Fusion Quality Metrics")
        print("-" * 80)

        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            telemetry_config=telemetry_manager_without_phoenix.config,
            enable_workflow_intelligence=False,
        )

        # Create diverse task results
        task_results = {
            "task_video": {
                "modality": "video",
                "result": "robotics assembly video tutorial machine learning",
                "confidence": 0.9,
            },
            "task_image": {
                "modality": "image",
                "result": "robotics assembly diagram machine components",
                "confidence": 0.85,
            },
            "task_document": {
                "modality": "document",
                "result": "robotics engineering guide machine learning algorithms",
                "confidence": 0.8,
            },
        }

        fused_result = {"content": "combined results", "confidence": 0.85}
        consistency_metrics = {"consistency_score": 0.7}

        # Calculate fusion quality
        print("\n📏 Calculating fusion quality metrics...")
        quality = orchestrator._calculate_fusion_quality(
            task_results, fused_result, consistency_metrics
        )

        # Validate metrics
        print("\n✅ Validating quality metrics...")
        assert "overall_quality" in quality
        assert "coverage" in quality
        assert "consistency" in quality
        assert "coherence" in quality
        assert "redundancy" in quality
        assert "complementarity" in quality

        # Check values are in valid ranges
        assert 0.0 <= quality["overall_quality"] <= 1.0
        assert 0.0 <= quality["coverage"] <= 1.0
        assert 0.0 <= quality["consistency"] <= 1.0
        assert 0.0 <= quality["coherence"] <= 1.0
        assert 0.0 <= quality["redundancy"] <= 1.0
        assert 0.0 <= quality["complementarity"] <= 1.0

        # Check modality coverage
        assert quality["modality_count"] == 3
        assert set(quality["modalities"]) == {"video", "image", "document"}

        print(f"   Overall quality: {quality['overall_quality']:.3f}")
        print(f"   Coverage: {quality['coverage']:.3f}")
        print(f"   Consistency: {quality['consistency']:.3f}")
        print(f"   Coherence: {quality['coherence']:.3f}")
        print(f"   Redundancy: {quality['redundancy']:.3f}")
        print(f"   Complementarity: {quality['complementarity']:.3f}")

        print("✅ Fusion quality metrics validated")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
