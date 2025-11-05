"""
Inspect AI tasks for video retrieval evaluation
"""

import logging
from dataclasses import dataclass
from typing import Any

from cogniverse_core.config.utils import get_config
from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample

from .scorers import VideoRetrievalScorer
from .solvers import (
    CogniverseRetrievalSolver,
    RelevanceJudgmentCollector,
    ResultRankingAnalyzer,
)

logger = logging.getLogger(__name__)


@dataclass
class VideoQuery:
    """Video query with expected results"""

    query: str
    category: str
    expected_videos: list[str]
    metadata: dict[str, Any] = None


@task
def video_retrieval_accuracy(
    profiles: list[str] = None, strategies: list[str] = None, dataset_path: str = None
) -> Task:
    """
    Evaluate video retrieval accuracy across different query types

    Args:
        profiles: List of video processing profiles to test
        strategies: List of ranking strategies to test
        dataset_path: Path to evaluation dataset

    Returns:
        Inspect AI Task for video retrieval evaluation
    """
    # Load configuration
    from cogniverse_core.config.manager import ConfigManager
    config_manager = ConfigManager()
    _ = get_config(tenant_id="default", config_manager=config_manager)  # noqa: F841

    # Default profiles and strategies if not specified
    if profiles is None:
        profiles = [
            "video_colpali_smol500_mv_frame",
            "video_videoprism_lvt_base_sv_global",
        ]

    if strategies is None:
        strategies = ["float_float", "binary_binary", "hybrid_binary_bm25"]

    # Load dataset
    dataset = load_video_retrieval_dataset(dataset_path)

    # Create solvers
    solvers = [
        CogniverseRetrievalSolver(profiles=profiles, strategies=strategies),
        ResultRankingAnalyzer(),
        RelevanceJudgmentCollector(),
    ]

    # Create scorer
    scorer = VideoRetrievalScorer(metrics=["mrr", "ndcg", "precision", "recall"])

    return Task(
        dataset=dataset,
        solver=solvers,
        scorer=scorer,
        metadata={
            "task_type": "video_retrieval",
            "profiles": profiles,
            "strategies": strategies,
        },
    )


@task
def temporal_understanding(
    profiles: list[str] = None, dataset_path: str = None
) -> Task:
    """
    Evaluate temporal query understanding

    Args:
        profiles: List of video processing profiles to test
        dataset_path: Path to temporal queries dataset

    Returns:
        Inspect AI Task for temporal understanding evaluation
    """
    from .scorers import TemporalAccuracyScorer
    from .solvers import TemporalQueryProcessor, TimeRangeExtractor

    # Default profiles if not specified
    if profiles is None:
        profiles = [
            "video_videoprism_lvt_base_sv_global",
            "video_colpali_smol500_mv_frame",
        ]

    # Load dataset
    dataset = load_temporal_dataset(dataset_path)

    # Create solvers
    solvers = [
        TemporalQueryProcessor(),
        TimeRangeExtractor(),
        CogniverseRetrievalSolver(profiles=profiles),
    ]

    # Create scorer
    scorer = TemporalAccuracyScorer()

    return Task(
        dataset=dataset,
        solver=solvers,
        scorer=scorer,
        metadata={"task_type": "temporal_understanding", "profiles": profiles},
    )


@task
def multimodal_alignment(profiles: list[str] = None, dataset_path: str = None) -> Task:
    """
    Evaluate cross-modal understanding and alignment

    Args:
        profiles: List of video processing profiles to test
        dataset_path: Path to multimodal dataset

    Returns:
        Inspect AI Task for multimodal alignment evaluation
    """
    from .scorers import AlignmentScorer
    from .solvers import (
        CrossModalAlignmentChecker,
        TextQueryEncoder,
        VisualQueryEncoder,
    )

    # Default profiles if not specified
    if profiles is None:
        profiles = [
            "video_videoprism_base_sv_global",
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_sv_chunk",
        ]

    # Load dataset
    dataset = load_multimodal_dataset(dataset_path)

    # Create solvers
    solvers = [VisualQueryEncoder(), TextQueryEncoder(), CrossModalAlignmentChecker()]

    # Create scorer
    scorer = AlignmentScorer()

    return Task(
        dataset=dataset,
        solver=solvers,
        scorer=scorer,
        metadata={"task_type": "multimodal_alignment", "profiles": profiles},
    )


@task
def failure_analysis(
    profiles: list[str] = None, strategies: list[str] = None, dataset_path: str = None
) -> Task:
    """
    Analyze failure cases and patterns

    Args:
        profiles: List of video processing profiles to test
        strategies: List of ranking strategies to test
        dataset_path: Path to evaluation dataset

    Returns:
        Inspect AI Task for failure analysis
    """
    from .scorers import FailureAnalysisScorer
    from .solvers import ErrorPatternDetector, FailureAnalyzer

    # Default profiles and strategies if not specified
    if profiles is None:
        profiles = [
            "video_colpali_smol500_mv_frame",
            "video_videoprism_lvt_base_sv_global",
        ]

    if strategies is None:
        strategies = ["float_float", "binary_binary", "hybrid_binary_bm25"]

    # Load dataset
    dataset = load_video_retrieval_dataset(dataset_path)

    # Create solvers
    solvers = [
        CogniverseRetrievalSolver(profiles=profiles, strategies=strategies),
        FailureAnalyzer(),
        ErrorPatternDetector(),
    ]

    # Create scorer
    scorer = FailureAnalysisScorer()

    return Task(
        dataset=dataset,
        solver=solvers,
        scorer=scorer,
        metadata={
            "task_type": "failure_analysis",
            "profiles": profiles,
            "strategies": strategies,
        },
    )


def load_video_retrieval_dataset(dataset_path: str | None = None) -> Dataset:
    """Load video retrieval evaluation dataset"""
    if dataset_path is None:
        # Use default test queries
        from tests.comprehensive_video_query_test_v2 import VISUAL_TEST_QUERIES

        samples = []
        for query_data in VISUAL_TEST_QUERIES:
            sample = Sample(
                input=query_data["query"],
                target=str(query_data["expected_videos"]),
                metadata={
                    "category": query_data["category"],
                    "expected_videos": query_data["expected_videos"],
                },
            )
            samples.append(sample)

        return Dataset(name="video_retrieval", samples=samples)
    else:
        # Load from file
        import json
        from pathlib import Path

        path = Path(dataset_path)
        with open(path) as f:
            data = json.load(f)

        samples = []
        for item in data:
            sample = Sample(
                input=item["query"],
                target=str(item["expected_videos"]),
                metadata={
                    "category": item.get("category", "general"),
                    "expected_videos": item["expected_videos"],
                },
            )
            samples.append(sample)

        return Dataset(name="video_retrieval", samples=samples)


def load_temporal_dataset(dataset_path: str | None = None) -> Dataset:
    """Load temporal understanding evaluation dataset"""
    if dataset_path is None:
        # Create default temporal queries
        temporal_queries = [
            {
                "query": "show me the first 30 seconds of the video",
                "expected_time_range": [0, 30],
                "category": "absolute_time",
            },
            {
                "query": "what happens at the end of the video",
                "expected_time_range": [-30, -1],  # Last 30 seconds
                "category": "relative_time",
            },
            {
                "query": "find the middle part of the video",
                "expected_time_range": [0.4, 0.6],  # 40-60% of video
                "category": "proportional_time",
            },
        ]

        samples = []
        for query_data in temporal_queries:
            sample = Sample(
                input=query_data["query"],
                target=str(query_data["expected_time_range"]),
                metadata={
                    "category": query_data["category"],
                    "expected_time_range": query_data["expected_time_range"],
                },
            )
            samples.append(sample)

        return Dataset(name="video_retrieval", samples=samples)
    else:
        # Load from file
        import json
        from pathlib import Path

        path = Path(dataset_path)
        with open(path) as f:
            data = json.load(f)

        samples = []
        for item in data:
            sample = Sample(
                input=item["query"],
                target=str(item["expected_time_range"]),
                metadata={
                    "category": item.get("category", "general"),
                    "expected_time_range": item["expected_time_range"],
                },
            )
            samples.append(sample)

        return Dataset(name="video_retrieval", samples=samples)


def load_multimodal_dataset(dataset_path: str | None = None) -> Dataset:
    """Load multimodal alignment evaluation dataset"""
    if dataset_path is None:
        # Create default multimodal queries
        multimodal_queries = [
            {
                "text_query": "person wearing winter clothes",
                "visual_description": "snowy outdoor scene with person in heavy jacket",
                "expected_alignment": True,
                "category": "clothing",
            },
            {
                "text_query": "industrial machinery",
                "visual_description": "dark factory setting with metal equipment",
                "expected_alignment": True,
                "category": "scene",
            },
            {
                "text_query": "animated cartoon",
                "visual_description": "3D rendered character with bright colors",
                "expected_alignment": True,
                "category": "style",
            },
        ]

        samples = []
        for query_data in multimodal_queries:
            sample = Sample(
                input=f"{query_data['text_query']}|{query_data['visual_description']}",
                target=str(query_data["expected_alignment"]),
                metadata={
                    "category": query_data["category"],
                    "text_query": query_data["text_query"],
                    "visual_description": query_data["visual_description"],
                    "expected_alignment": query_data["expected_alignment"],
                },
            )
            samples.append(sample)

        return Dataset(name="video_retrieval", samples=samples)
    else:
        # Load from file
        import json
        from pathlib import Path

        path = Path(dataset_path)
        with open(path) as f:
            data = json.load(f)

        samples = []
        for item in data:
            sample = Sample(
                input=f"{item['text_query']}|{item['visual_description']}",
                target=str(item["expected_alignment"]),
                metadata={
                    "category": item.get("category", "general"),
                    "text_query": item["text_query"],
                    "visual_description": item["visual_description"],
                    "expected_alignment": item["expected_alignment"],
                },
            )
            samples.append(sample)

        return Dataset(name="video_retrieval", samples=samples)
