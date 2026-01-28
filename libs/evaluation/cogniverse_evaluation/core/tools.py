"""
Tools for Inspect AI solvers to interact with external services.
"""

import logging
from typing import Any

from inspect_ai.tool import tool

logger = logging.getLogger(__name__)


@tool
def video_search_tool():
    """
    Tool that wraps our search service for use in Inspect AI.
    """

    async def run(
        query: str, profile: str, strategy: str, top_k: int = 10
    ) -> list[dict[str, Any]]:
        """
        Execute a search using the Cogniverse search service.

        Args:
            query: Search query
            profile: Video processing profile
            strategy: Ranking strategy
            top_k: Number of results to return

        Returns:
            List of search results
        """
        try:
            # Import here to avoid circular dependencies
            from cogniverse_foundation.config.utils import (
                create_default_config_manager,
                get_config,
            )
            from cogniverse_runtime.search.service import SearchService

            # Initialize ConfigManager for dependency injection
            config_manager = create_default_config_manager()
            config = get_config(tenant_id="default", config_manager=config_manager)

            # Create search service with specified profile
            search_service = SearchService(config, profile)

            logger.info(
                f"Executing search: query='{query[:50]}...', profile={profile}, strategy={strategy}"
            )

            # Run the search
            search_results_raw = search_service.search(
                query=query, top_k=top_k, ranking_strategy=strategy
            )

            # Convert results to standardized format
            search_results = []
            for i, result in enumerate(search_results_raw):
                result_dict = result.to_dict()

                # Extract video_id
                video_id = result_dict.get("source_id", "")
                if not video_id and "document_id" in result_dict:
                    # Extract from document_id (e.g., "video_frame_0" -> "video")
                    doc_id = result_dict["document_id"]
                    if "_frame_" in doc_id:
                        video_id = doc_id.split("_frame_")[0]
                    else:
                        video_id = doc_id

                # Get score (ensure it's not 0)
                score = result_dict.get("score", 0.0)
                if score == 0.0:
                    # Use rank-based score if no score available
                    score = 1.0 / (i + 1)

                search_results.append(
                    {
                        "video_id": video_id,
                        "score": float(score),
                        "rank": i + 1,
                        "document_id": result_dict.get("document_id", ""),
                        "content": result_dict.get("content", ""),
                        "temporal_info": result_dict.get("temporal_info", {}),
                        "metadata": result_dict.get("metadata", {}),
                    }
                )

            logger.info(f"Search returned {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            # Raise the error to properly signal failure
            raise RuntimeError(f"Search tool failed: {e}") from e

    return run


@tool
def phoenix_query_tool():
    """
    Tool for querying telemetry provider for traces and datasets.

    Note: Name kept as 'phoenix_query_tool' for backward compatibility.
    """

    async def run(query_type: str, **kwargs) -> Any:
        """
        Query telemetry provider for various data types.

        Args:
            query_type: Type of query ("traces", "datasets", "experiments")
            **kwargs: Additional query parameters

        Returns:
            Query results
        """
        from cogniverse_evaluation.providers import get_evaluation_provider

        provider = get_evaluation_provider(
            tenant_id=kwargs.get("tenant_id", "default"),
            project_name=kwargs.get("project_name", "cogniverse-default"),
        )

        try:
            if query_type == "traces":
                # Query for traces
                df = await provider.telemetry.traces.get_spans(
                    project=kwargs.get("project_name", "cogniverse-default"),
                    start_time=kwargs.get("start_time"),
                    end_time=kwargs.get("end_time"),
                    limit=kwargs.get("limit", 100),
                )
                return df.to_dict(orient="records")

            elif query_type == "datasets":
                # Get dataset information
                dataset_name = kwargs.get("name")
                if dataset_name:
                    dataset = await provider.telemetry.datasets.get_dataset(
                        dataset_name
                    )
                    return {
                        "name": dataset.get("name"),
                        "num_examples": dataset.get("num_examples", 0),
                        "examples": dataset.get("examples", [])[:10],
                    }
                else:
                    raise ValueError("Dataset name is required for dataset queries")

            elif query_type == "experiments":
                # Query experiments through spans with experiment metadata
                experiment_name = kwargs.get("name")
                if not experiment_name:
                    # List all experiments by finding unique experiment names in spans
                    df = await provider.telemetry.traces.get_spans(
                        project=kwargs.get("project_name", "cogniverse-default"),
                        limit=1000,
                    )
                    if (
                        not df.empty
                        and "attributes.metadata.experiment_name" in df.columns
                    ):
                        experiments = (
                            df["attributes.metadata.experiment_name"]
                            .dropna()
                            .unique()
                            .tolist()
                        )
                        return {"experiments": experiments, "count": len(experiments)}
                    else:
                        return {"experiments": [], "count": 0}
                else:
                    # Get specific experiment data
                    df = await provider.telemetry.traces.get_spans(
                        project=kwargs.get("project_name", "cogniverse-default"),
                        limit=kwargs.get("limit", 1000),
                    )
                    # Filter by experiment name
                    if (
                        not df.empty
                        and "attributes.metadata.experiment_name" in df.columns
                    ):
                        df = df[
                            df["attributes.metadata.experiment_name"] == experiment_name
                        ]
                    if not df.empty:
                        return {
                            "experiment_name": experiment_name,
                            "traces": df.to_dict(orient="records"),
                            "count": len(df),
                        }
                    else:
                        return {
                            "experiment_name": experiment_name,
                            "traces": [],
                            "count": 0,
                        }

            else:
                raise ValueError(f"Unknown query type: {query_type}")

        except Exception as e:
            logger.error(f"Phoenix query failed: {e}")
            # Re-raise with context
            raise RuntimeError(f"Phoenix query tool failed: {e}") from e

    return run
