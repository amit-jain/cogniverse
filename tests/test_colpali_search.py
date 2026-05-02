#!/usr/bin/env python3
"""
Test ColPali text-to-video search (semantic search from text queries)
This tests the core functionality used by the multi-agent system
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

from cogniverse_agents.search.service import SearchService
from cogniverse_core.common.models.model_loaders import RemoteColPaliLoader
from cogniverse_foundation.config.utils import create_default_config_manager, get_config
from tests.test_utils import TestResultsFormatter

COLPALI_MODEL = "vidore/colpali-v1.3-hf"
_logger = logging.getLogger(__name__)


def _spawn_colpali_client(vllm_sidecar):
    sidecar_url = vllm_sidecar.spawn(
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
        config={"remote_inference_url": sidecar_url},
        logger=_logger,
    )
    client, _ = loader.load_model()
    return client


def _encode_query(client, query_text):
    result = client.process_queries([query_text], model_name=COLPALI_MODEL)
    embeddings_np = np.asarray(result["embeddings"]).astype(np.float32)
    if embeddings_np.ndim == 3:
        embeddings_np = embeddings_np.squeeze(0)
    return embeddings_np


def load_test_queries(num_queries=5, seed=42):
    """Load random test queries from our evaluation set"""
    # Try to load from our retrieval test queries
    query_file = (
        Path(__file__).parent.parent / "retrieval_test_queries_with_temporal.json"
    )

    if query_file.exists():
        with open(query_file, "r") as f:
            data = json.load(f)
            all_queries = data.get("queries", [])

            # Filter for queries suitable for ColPali (visual-focused)
            colpali_suitable = []
            for q in all_queries:
                # Prefer visual and object queries for ColPali
                if q["category"] in [
                    "action_retrieval",
                    "object_retrieval",
                    "scene_understanding",
                    "visual_attribute_retrieval",
                    "colpali_optimized",
                ]:
                    colpali_suitable.append(q)

            # If we have suitable queries, use them; otherwise use all
            queries_to_sample = colpali_suitable if colpali_suitable else all_queries

            # Sample random queries
            random.seed(seed)
            sampled = random.sample(
                queries_to_sample, min(num_queries, len(queries_to_sample))
            )

            return [(q["query"], q.get("expected_videos", [])) for q in sampled]

    # Fallback to default queries if file not found
    return [
        ("doctor explaining medical procedures", []),
        ("people playing sports outdoors", []),
        ("cooking in a kitchen", []),
    ]


def test_colpali_search(output_format="table", save_results=False, num_queries=5):
    """Test ColPali search using the new SearchService"""

    # Initialize results formatter
    formatter = TestResultsFormatter("colpali_search_service")

    # Get config and create search service
    # Create temporary ConfigManager for test
    from pathlib import Path

    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

    config_manager = create_default_config_manager()
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
    config = get_config(tenant_id="test_tenant", config_manager=config_manager)
    profile = "frame_based_colpali"

    print(f"Creating SearchService for profile: {profile}")
    try:
        search_service = SearchService(
            config, profile, config_manager=config_manager, schema_loader=schema_loader
        )
        print("✅ SearchService created")
    except Exception as e:
        print(f"❌ Failed to create SearchService: {e}")
        return

    # Load random test queries
    test_queries = load_test_queries(num_queries)
    print(f"\nLoaded {len(test_queries)} test queries")

    # Test all queries
    all_results = []

    for test_query, expected_videos in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Query: '{test_query}'")
        if expected_videos:
            print(f"Expected videos: {expected_videos[:3]}...")  # Show first 3

        print("\n🔍 Performing search...")

        try:
            # Use SearchService to perform search
            search_results = search_service.search(test_query, top_k=20)

            print(f"✅ Search completed! Found {len(search_results)} results")

            # Convert SearchResults to dicts and extract video IDs
            result_dicts = [r.to_dict() for r in search_results]
            retrieved_videos = [
                r.get("source_id", r["document_id"].split("_")[0]) for r in result_dicts
            ]

            if expected_videos:
                # Calculate recall@k
                recall_at_5 = len(
                    set(retrieved_videos[:5]) & set(expected_videos)
                ) / len(expected_videos)
                recall_at_10 = len(
                    set(retrieved_videos[:10]) & set(expected_videos)
                ) / len(expected_videos)

                # Calculate MRR
                mrr = 0
                for i, vid in enumerate(retrieved_videos):
                    if vid in expected_videos:
                        mrr = 1.0 / (i + 1)
                        break

                print("\n📊 Evaluation Metrics:")
                print(f"  Recall@5: {recall_at_5:.3f}")
                print(f"  Recall@10: {recall_at_10:.3f}")
                print(f"  MRR: {mrr:.3f}")

            # Collect results for formatting
            results = []
            for i, result_dict in enumerate(result_dicts[:5]):
                video_id = result_dict.get(
                    "source_id", result_dict["document_id"].split("_")[0]
                )
                is_relevant = (
                    "✓" if expected_videos and video_id in expected_videos else ""
                )
                result = {
                    "Rank": i + 1,
                    "Video ID": video_id,
                    "Frame ID": result_dict.get("metadata", {}).get("frame_id", "N/A"),
                    "Score": f"{result_dict['score']:.4f}",
                    "Relevant": is_relevant,
                    "Description": result_dict.get("metadata", {}).get(
                        "frame_description", ""
                    )[:50]
                    + "...",
                }
                results.append(result)

            # Display results based on format
            if output_format == "table":
                print("\n" + formatter.format_table(results))
            else:
                for i, result_dict in enumerate(result_dicts[:5]):
                    video_id = result_dict.get(
                        "source_id", result_dict["document_id"].split("_")[0]
                    )
                    print(f"\nResult {i + 1}:")
                    print(f"  - Video ID: {video_id}")
                    print(
                        f"  - Frame ID: {result_dict.get('metadata', {}).get('frame_id', 'N/A')}"
                    )
                    print(f"  - Score: {result_dict['score']:.4f}")
                    if "temporal_info" in result_dict and result_dict["temporal_info"]:
                        print(
                            f"  - Time: {result_dict['temporal_info']['start_time']:.1f}s"
                        )
                    desc = result_dict.get("metadata", {}).get("frame_description", "")
                    if desc:
                        print(f"  - Description: {desc[:100]}...")

            all_results.append(
                {
                    "query": test_query,
                    "expected_videos": expected_videos,
                    "results": results,
                    "metrics": {
                        "recall_at_5": recall_at_5 if expected_videos else None,
                        "recall_at_10": recall_at_10 if expected_videos else None,
                        "mrr": mrr if expected_videos else None,
                    },
                }
            )

        except Exception as e:
            print(f"❌ Search failed: {e}")
            all_results.append({"query": test_query, "error": str(e)})

    # Save all results if requested
    if save_results:
        import json

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = f"outputs/test_results/colpali_search_{timestamp}.json"
        Path("outputs/test_results").mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n📊 All results saved to: {json_path}")

    return all_results


def test_float_float_search(
    vllm_sidecar, output_format="table", save_results=False, monkeypatch=None
):
    """Test pure float visual search"""

    if monkeypatch:
        monkeypatch.setenv("VESPA_SCHEMA", "video_colpali_smol500_mv_frame")
    else:
        os.environ["VESPA_SCHEMA"] = "video_colpali_smol500_mv_frame"

    formatter = TestResultsFormatter("colpali_float_float")

    config_manager = create_default_config_manager()
    config = get_config(tenant_id="test_tenant", config_manager=config_manager)
    vespa_url = config.get("backend_url", "http://localhost")
    vespa_port = config.get("backend_port", 8080)

    print("\n\n=== Testing Float-Float Visual Search ===")
    colpali_client = _spawn_colpali_client(vllm_sidecar)
    print("✅ ColPali sidecar ready")

    test_query = "doctor explaining medical procedures"
    print(f"\nTest query: '{test_query}'")

    embeddings_np = _encode_query(colpali_client, test_query)
    print(f"Query embedding shape: {embeddings_np.shape}")
    print(f"Embeddings dtype: {embeddings_np.dtype}")

    # Use the search client directly
    from cogniverse_vespa.vespa_search_client import VespaVideoSearchClient

    # Initialize search client
    search_client = VespaVideoSearchClient(
        backend_url=vespa_url,
        backend_port=vespa_port,
        tenant_id="test_tenant",
        config_manager=config_manager,
    )

    # Prepare search params - NO text query for pure visual search
    search_params = {
        "query": "",  # Empty query for pure visual search
        "ranking": "float_float",
        "top_k": 10,
    }

    print("\n🔍 Performing Float-Float visual search...")
    print(f"   Vespa: {vespa_url}:{vespa_port}")
    print("   Ranking: float_float")

    try:
        # Use the search client with proper embedding format
        results = search_client.search(search_params, embeddings_np)

        print(f"✅ Search completed! Found {len(results)} results")

        # Format results for display
        formatted_results = []
        for i, result in enumerate(results[:5]):
            formatted_result = {
                "Rank": i + 1,
                "Video ID": result.get("video_id"),
                "Frame ID": result.get("frame_id"),
                "Score": f"{result.get('relevance'):.4f}",
                "Description": result.get("frame_description", "")[:60] + "...",
            }
            formatted_results.append(formatted_result)

        # Display results based on format
        if output_format == "table":
            print("\n" + formatter.format_table(formatted_results))
        else:
            for i, result in enumerate(results[:5]):
                print(f"\nResult {i + 1}:")
                print(f"  - Video ID: {result.get('video_id')}")
                print(f"  - Frame ID: {result.get('frame_id')}")
                print(f"  - Relevance score: {result.get('relevance')}")
                print(
                    f"  - Description: {result.get('frame_description', '')[:100]}..."
                )

        # Save results if requested
        if save_results:
            csv_path = formatter.save_csv(formatted_results)
            print(f"\n📊 Results saved to: {csv_path}")

        return formatted_results

    except Exception as e:
        print(f"Search failed: {e}")
        import traceback

        traceback.print_exc()


def test_hybrid_float_bm25(
    vllm_sidecar, output_format="table", save_results=False, monkeypatch=None
):
    """Test hybrid search with float embeddings - exactly like video agent"""

    if monkeypatch:
        monkeypatch.setenv("VESPA_SCHEMA", "video_colpali_smol500_mv_frame")
    else:
        os.environ["VESPA_SCHEMA"] = "video_colpali_smol500_mv_frame"

    formatter = TestResultsFormatter("colpali_hybrid_float_bm25")

    config_manager = create_default_config_manager()
    config = get_config(tenant_id="test_tenant", config_manager=config_manager)
    vespa_url = config.get("backend_url", "http://localhost")
    vespa_port = config.get("backend_port", 8080)

    print("\n\n=== Testing Hybrid Float BM25 (like video agent) ===")
    colpali_client = _spawn_colpali_client(vllm_sidecar)
    print("✅ ColPali sidecar ready")

    test_query = "doctor explaining medical procedures"
    print(f"\nTest query: '{test_query}'")

    embeddings_np = _encode_query(colpali_client, test_query)
    print(f"Query embedding shape: {embeddings_np.shape}")
    print(f"Embeddings dtype: {embeddings_np.dtype}")

    # Use the search client directly to ensure proper formatting
    from cogniverse_vespa.vespa_search_client import VespaVideoSearchClient

    # Initialize search client
    search_client = VespaVideoSearchClient(
        backend_url=vespa_url,
        backend_port=vespa_port,
        tenant_id="test_tenant",
        config_manager=config_manager,
    )

    # Prepare search params
    search_params = {"query": test_query, "ranking": "hybrid_float_bm25", "top_k": 10}

    print("\n🔍 Performing Hybrid Float+BM25 search...")
    print(f"   Vespa: {vespa_url}:{vespa_port}")
    print("   Ranking: hybrid_float_bm25")

    try:
        # Use the search client with proper embedding format
        results = search_client.search(search_params, embeddings_np)

        print(f"✅ Search completed! Found {len(results)} results")

        # Format results for display
        formatted_results = []
        for i, result in enumerate(results[:5]):
            formatted_result = {
                "Rank": i + 1,
                "Video ID": result.get("video_id"),
                "Frame ID": result.get("frame_id"),
                "Score": f"{result.get('relevance'):.4f}",
                "Description": result.get("frame_description", "")[:40] + "...",
                "Transcript": result.get("audio_transcript", "")[:40] + "...",
            }
            formatted_results.append(formatted_result)

        # Display results based on format
        if output_format == "table":
            print("\n" + formatter.format_table(formatted_results))
        else:
            for i, result in enumerate(results[:5]):
                print(f"\nResult {i + 1}:")
                print(f"  - Video ID: {result.get('video_id')}")
                print(f"  - Frame ID: {result.get('frame_id')}")
                print(f"  - Relevance score: {result.get('relevance')}")
                print(
                    f"  - Description: {result.get('frame_description', '')[:100]}..."
                )
                print(
                    f"  - Transcript snippet: {result.get('audio_transcript', '')[:100]}..."
                )

        # Save results if requested
        if save_results:
            csv_path = formatter.save_csv(formatted_results)
            print(f"\n📊 Results saved to: {csv_path}")

        return formatted_results

    except Exception as e:
        print(f"Search failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ColPali search functionality")
    parser.add_argument(
        "--format",
        choices=["table", "text"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument("--save", action="store_true", help="Save results to CSV file")
    parser.add_argument(
        "--test",
        choices=["binary", "float", "hybrid", "all"],
        default="all",
        help="Which test to run (default: all)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=5,
        help="Number of random queries to test (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for query selection (default: 42)",
    )

    args = parser.parse_args()

    config_manager = create_default_config_manager()
    config = get_config(tenant_id="test_tenant", config_manager=config_manager)
    output_format = config.get("test_output_format", args.format)

    print("Running ColPali search tests...")
    print("=" * 60)

    from tests.utils.vllm_sidecar import VllmSidecarFactory

    factory = VllmSidecarFactory()
    try:
        if args.test in ["binary", "all"]:
            print("\n🔍 Binary Search Test:")
            test_colpali_search(
                output_format=output_format,
                save_results=args.save,
                num_queries=args.num_queries,
            )

        if args.test in ["float", "all"]:
            print("\n🔍 Float-Float Search Test:")
            test_float_float_search(
                factory, output_format=output_format, save_results=args.save
            )

        if args.test in ["hybrid", "all"]:
            print("\n🔍 Hybrid Float+BM25 Test:")
            test_hybrid_float_bm25(
                factory, output_format=output_format, save_results=args.save
            )
    finally:
        factory.teardown()
