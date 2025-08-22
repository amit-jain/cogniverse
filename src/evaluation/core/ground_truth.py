"""
Generic ground truth extraction using schema-driven analysis.

This module provides ground truth extraction that adapts to any schema
without hardcoded assumptions about the domain.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
import asyncio

from src.evaluation.core.schema_analyzer import get_schema_analyzer

logger = logging.getLogger(__name__)


class GroundTruthError(Exception):
    """Base exception for ground truth extraction errors."""

    pass


class SchemaDiscoveryError(GroundTruthError):
    """Error discovering schema information."""

    pass


class BackendError(GroundTruthError):
    """Error interacting with backend."""

    pass


def get_ground_truth_strategy(config: Dict[str, Any]) -> "GroundTruthStrategy":
    """Get appropriate ground truth strategy based on config.

    Args:
        config: Configuration dictionary

    Returns:
        Ground truth strategy instance
    """
    strategy_type = config.get("ground_truth_strategy", "schema_aware")

    if strategy_type == "schema_aware":
        return SchemaAwareGroundTruthStrategy()
    elif strategy_type == "dataset":
        return DatasetGroundTruthStrategy()
    elif strategy_type == "backend":
        return BackendGroundTruthStrategy()
    elif strategy_type == "hybrid":
        return HybridGroundTruthStrategy()
    else:
        # Default to schema-aware
        return SchemaAwareGroundTruthStrategy()


class GroundTruthStrategy(ABC):
    """Abstract base class for ground truth extraction."""

    @abstractmethod
    async def extract_ground_truth(
        self, trace_data: Dict[str, Any], backend: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Extract ground truth for a trace.

        Args:
            trace_data: Trace data containing query, results, metadata
            backend: Optional backend instance for querying

        Returns:
            Dictionary containing:
                - expected_items: List of ground truth item IDs
                - confidence: float between 0-1 indicating confidence
                - source: str indicating where ground truth came from
                - metadata: additional metadata about extraction
        """
        pass


class SchemaAwareGroundTruthStrategy(GroundTruthStrategy):
    """Ground truth extraction that adapts to schema automatically."""

    def __init__(self):
        """Initialize strategy."""
        self.schema_cache = {}

    async def extract_ground_truth(
        self, trace_data: Dict[str, Any], backend: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Extract ground truth using schema-aware analysis."""
        if not backend:
            return {
                "expected_items": [],
                "confidence": 0.0,
                "source": "no_backend",
                "metadata": {
                    "error": "No backend provided for ground truth extraction"
                },
            }

        query = trace_data.get("query", "")
        if not query:
            return {
                "expected_items": [],
                "confidence": 0.0,
                "source": "no_query",
                "metadata": {"error": "No query provided in trace data"},
            }

        try:
            # Get schema information with proper error handling
            schema_info = await self._get_schema_info(trace_data, backend)

            schema_name = schema_info["name"]
            schema_fields = schema_info["fields"]

            # Get appropriate analyzer for this schema
            analyzer = get_schema_analyzer(schema_name, schema_fields)

            # Analyze query using schema-specific analyzer
            query_constraints = analyzer.analyze_query(query, schema_fields)

            # Search based on query type and constraints
            results = await self._search_with_constraints(
                backend, query, query_constraints, schema_fields
            )

            # Extract item IDs using analyzer
            expected_items = []
            extraction_errors = []

            for result in results:
                try:
                    item_id = analyzer.extract_item_id(result)
                    if item_id and item_id not in expected_items:
                        expected_items.append(item_id)
                except Exception as e:
                    extraction_errors.append(str(e))

            # Calculate confidence based on actual data quality
            confidence = self._calculate_confidence(
                query_constraints,
                len(expected_items),
                len(extraction_errors),
                len(results),
            )

            # Get the appropriate field name from analyzer
            expected_field_name = analyzer.get_expected_field_name()

            return {
                expected_field_name: expected_items,  # Dynamic field name
                "expected_items": expected_items,  # Also include generic name
                "confidence": confidence,
                "source": "schema_aware_backend",
                "metadata": {
                    "schema": schema_name,
                    "query_type": query_constraints.get("query_type"),
                    "total_results": len(results),
                    "extraction_errors": len(extraction_errors),
                    "constraints_used": list(query_constraints.keys()),
                },
            }

        except SchemaDiscoveryError as e:
            logger.error(f"Failed to discover schema: {e}")
            return {
                "expected_items": [],
                "confidence": 0.0,
                "source": "schema_discovery_error",
                "metadata": {"error": str(e)},
            }
        except BackendError as e:
            logger.error(f"Backend error during ground truth extraction: {e}")
            return {
                "expected_items": [],
                "confidence": 0.0,
                "source": "backend_error",
                "metadata": {"error": str(e)},
            }
        except Exception as e:
            logger.error(f"Unexpected error during ground truth extraction: {e}")
            return {
                "expected_items": [],
                "confidence": 0.0,
                "source": "extraction_error",
                "metadata": {"error": str(e)},
            }

    async def _get_schema_info(
        self, trace_data: Dict[str, Any], backend: Any
    ) -> Dict[str, Any]:
        """Get schema information from trace or backend."""
        # Check if schema info is in trace metadata
        metadata = trace_data.get("metadata", {})
        if "schema" in metadata and "fields" in metadata:
            return {"name": metadata["schema"], "fields": metadata["fields"]}

        # Try to get from backend
        schema_name = None

        # Try different methods to get schema name
        if hasattr(backend, "schema_name"):
            schema_name = backend.schema_name
        elif hasattr(backend, "get_schema_name"):
            try:
                schema_name = await self._safe_async_call(backend.get_schema_name())
            except Exception as e:
                logger.warning(f"Failed to get schema name: {e}")

        if not schema_name:
            schema_name = metadata.get("schema", "unknown")

        # Discover fields with multiple methods
        fields = await self._discover_schema_fields(schema_name, backend)

        if not fields or not any(fields.values()):
            raise SchemaDiscoveryError(
                f"Failed to discover schema fields for '{schema_name}'. "
                "Backend may not be properly configured or accessible."
            )

        return {"name": schema_name, "fields": fields}

    async def _discover_schema_fields(
        self, schema_name: str, backend: Any
    ) -> Dict[str, List[str]]:
        """Discover available fields in schema using multiple methods."""
        fields = {
            "id_fields": [],
            "content_fields": [],
            "metadata_fields": [],
            "temporal_fields": [],
            "numeric_fields": [],
            "text_fields": [],
        }

        # Method 1: Try to get schema info directly from backend
        if hasattr(backend, "get_schema_info"):
            try:
                schema_info = await self._safe_async_call(
                    backend.get_schema_info(schema_name)
                )
                if schema_info:
                    return self._parse_schema_info(schema_info)
            except Exception as e:
                logger.warning(f"Failed to get schema info from backend: {e}")

        # Method 2: Try to get field mappings
        if hasattr(backend, "get_field_mappings"):
            try:
                mappings = await self._safe_async_call(backend.get_field_mappings())
                if mappings:
                    return self._parse_field_mappings(mappings)
            except Exception as e:
                logger.warning(f"Failed to get field mappings: {e}")

        # Method 3: Introspect from a sample query
        if hasattr(backend, "search"):
            try:
                # Do a minimal search to see what fields come back
                sample_results = await self._safe_async_call(
                    backend.search(
                        query_text="*", query_embeddings=None, top_k=1  # Wildcard query
                    )
                )

                if sample_results:
                    discovered = self._infer_fields_from_results(sample_results[0])
                    if any(discovered.values()):
                        logger.info(
                            f"Discovered fields from sample query: {discovered}"
                        )
                        return discovered
            except Exception as e:
                logger.warning(f"Failed to introspect schema from sample query: {e}")

        # Method 4: Try to list schema fields
        if hasattr(backend, "list_fields"):
            try:
                field_list = await self._safe_async_call(backend.list_fields())
                if field_list:
                    return self._categorize_fields(field_list)
            except Exception as e:
                logger.warning(f"Failed to list fields: {e}")

        # If all methods failed, raise an error instead of returning defaults
        raise SchemaDiscoveryError(
            f"Could not discover schema fields for '{schema_name}' using any available method"
        )

    def _parse_schema_info(self, schema_info: Dict[str, Any]) -> Dict[str, List[str]]:
        """Parse schema info from backend."""
        fields = {
            "id_fields": [],
            "content_fields": [],
            "metadata_fields": [],
            "temporal_fields": [],
            "numeric_fields": [],
            "text_fields": [],
        }

        # Parse different schema formats
        if "fields" in schema_info:
            for field_name, field_info in schema_info["fields"].items():
                field_type = field_info.get("type", "").lower()

                # Categorize based on type
                if "id" in field_name.lower() or field_info.get("is_id"):
                    fields["id_fields"].append(field_name)
                elif field_type in ["text", "string"]:
                    if (
                        "content" in field_name.lower()
                        or "description" in field_name.lower()
                    ):
                        fields["content_fields"].append(field_name)
                    else:
                        fields["text_fields"].append(field_name)
                elif field_type in ["timestamp", "datetime", "date"]:
                    fields["temporal_fields"].append(field_name)
                elif field_type in ["float", "int", "number"]:
                    fields["numeric_fields"].append(field_name)
                else:
                    fields["metadata_fields"].append(field_name)

        return fields

    def _parse_field_mappings(self, mappings: Dict[str, Any]) -> Dict[str, List[str]]:
        """Parse field mappings into categorized fields."""
        # If mappings already in our format, return as is
        if all(k in mappings for k in ["id_fields", "content_fields"]):
            return mappings

        # Otherwise, categorize the fields
        return self._categorize_fields(list(mappings.keys()))

    def _categorize_fields(self, field_list: List[str]) -> Dict[str, List[str]]:
        """Categorize a flat list of fields by their likely purpose."""
        fields = {
            "id_fields": [],
            "content_fields": [],
            "metadata_fields": [],
            "temporal_fields": [],
            "numeric_fields": [],
            "text_fields": [],
        }

        for field in field_list:
            field_lower = field.lower()

            # ID fields
            if any(id_term in field_lower for id_term in ["_id", "id_", "identifier"]):
                fields["id_fields"].append(field)
            # Temporal fields
            elif any(
                time_term in field_lower
                for time_term in ["time", "date", "timestamp", "created", "updated"]
            ):
                fields["temporal_fields"].append(field)
            # Content fields
            elif any(
                content_term in field_lower
                for content_term in ["content", "body", "text", "description"]
            ):
                fields["content_fields"].append(field)
            # Numeric fields
            elif any(
                num_term in field_lower
                for num_term in ["score", "rank", "count", "size", "length"]
            ):
                fields["numeric_fields"].append(field)
            # Text fields
            elif any(
                text_term in field_lower
                for text_term in ["title", "name", "label", "summary"]
            ):
                fields["text_fields"].append(field)
            # Everything else is metadata
            else:
                fields["metadata_fields"].append(field)

        return fields

    def _infer_fields_from_results(self, result: Any) -> Dict[str, List[str]]:
        """Infer field categories from a sample result."""
        fields = {
            "id_fields": [],
            "content_fields": [],
            "metadata_fields": [],
            "temporal_fields": [],
            "numeric_fields": [],
            "text_fields": [],
        }

        # Get all fields from result
        if hasattr(result, "__dict__"):
            all_fields = result.__dict__.keys()
        elif isinstance(result, dict):
            all_fields = result.keys()
        else:
            return fields

        # Categorize each field based on name and value type
        for field_name in all_fields:
            value = (
                getattr(result, field_name, None)
                if hasattr(result, field_name)
                else result.get(field_name)
            )

            # Skip None values
            if value is None:
                continue

            field_lower = field_name.lower()
            value_type = type(value).__name__

            # Categorize based on name and type
            if "id" in field_lower or field_name == "_id":
                fields["id_fields"].append(field_name)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                fields["numeric_fields"].append(field_name)
            elif "time" in field_lower or "date" in field_lower:
                fields["temporal_fields"].append(field_name)
            elif isinstance(value, str):
                if len(value) > 100:  # Long text is likely content
                    fields["content_fields"].append(field_name)
                else:
                    fields["text_fields"].append(field_name)
            elif isinstance(value, dict):
                fields["metadata_fields"].append(field_name)

        return fields

    async def _search_with_constraints(
        self,
        backend: Any,
        query: str,
        constraints: Dict[str, Any],
        schema_fields: Dict[str, List[str]],
    ) -> List[Any]:
        """Search backend using constraints."""
        if not hasattr(backend, "search"):
            raise BackendError("Backend does not support search operation")

        try:
            # Build search parameters based on constraints
            search_params = {
                "query_text": query,
                "top_k": constraints.get("max_results", 50),
            }

            # Add field-specific constraints if backend supports them
            if constraints.get("field_constraints"):
                search_params["field_constraints"] = constraints["field_constraints"]

            # Add temporal constraints if available
            if constraints.get("temporal_constraints"):
                search_params["temporal_constraints"] = constraints[
                    "temporal_constraints"
                ]

            # Execute search
            results = await self._safe_async_call(backend.search(**search_params))

            if not results:
                logger.warning(f"No results found for query: {query}")

            return results or []

        except Exception as e:
            raise BackendError(f"Search failed: {e}")

    def _calculate_confidence(
        self,
        constraints: Dict[str, Any],
        num_items: int,
        num_errors: int,
        total_results: int,
    ) -> float:
        """Calculate confidence based on actual extraction quality."""
        if total_results == 0:
            return 0.0

        # Start with extraction success rate
        extraction_rate = (
            (total_results - num_errors) / total_results if total_results > 0 else 0.0
        )

        # Adjust based on query specificity
        query_type = constraints.get("query_type", "generic")
        specificity_scores = {
            "generic": 0.3,
            "field_specific": 0.5,
            "temporal": 0.7,
            "structured": 0.8,
            "exact": 0.9,
        }

        # Determine query specificity
        if (
            constraints.get("field_constraints")
            and len(constraints["field_constraints"]) > 2
        ):
            query_specificity = specificity_scores["structured"]
        elif "temporal" in query_type:
            query_specificity = specificity_scores["temporal"]
        elif constraints.get("field_constraints"):
            query_specificity = specificity_scores["field_specific"]
        else:
            query_specificity = specificity_scores.get(
                query_type, specificity_scores["generic"]
            )

        # Adjust based on result count (too few or too many reduces confidence)
        if num_items == 0:
            count_factor = 0.0
        elif num_items < 3:
            count_factor = 0.7  # Few results might be incomplete
        elif num_items > 50:
            count_factor = 0.6  # Too many results might be too broad
        elif 3 <= num_items <= 20:
            count_factor = 1.0  # Ideal range
        else:
            count_factor = 0.8

        # Combine factors
        confidence = extraction_rate * query_specificity * count_factor

        # Add small penalty for any extraction errors
        if num_errors > 0:
            error_penalty = min(0.2, num_errors * 0.05)
            confidence = max(0.1, confidence - error_penalty)

        return min(1.0, max(0.0, confidence))

    async def _safe_async_call(self, coro_or_result):
        """Safely call async or sync methods."""
        if asyncio.iscoroutine(coro_or_result):
            return await coro_or_result
        return coro_or_result


class DatasetGroundTruthStrategy(GroundTruthStrategy):
    """Extract ground truth from pre-defined datasets."""

    async def extract_ground_truth(
        self, trace_data: Dict[str, Any], backend: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Extract ground truth from dataset annotations."""
        # This would connect to a dataset with labeled ground truth
        query = trace_data.get("query", "")
        dataset_name = trace_data.get("metadata", {}).get("dataset")

        if not dataset_name:
            return {
                "expected_items": [],
                "confidence": 0.0,
                "source": "no_dataset",
                "metadata": {"error": "No dataset specified"},
            }

        try:
            # Connect to Phoenix or other dataset store
            import phoenix as px

            try:
                client = px.Client()
                dataset = client.get_dataset(dataset_name)

                # Find matching query in dataset
                for example in dataset.examples:
                    if example.input.get("query") == query:
                        # Found matching query
                        expected_items = example.output.get("expected_items", [])

                        # Try multiple field names for compatibility
                        if not expected_items:
                            expected_items = example.output.get("expected_videos", [])
                        if not expected_items:
                            expected_items = example.output.get("ground_truth", [])

                        return {
                            "expected_items": expected_items,
                            "confidence": 0.95,  # High confidence for labeled data
                            "source": "dataset",
                            "metadata": {
                                "dataset": dataset_name,
                                "matched_query": query,
                            },
                        }

                # Query not found in dataset
                return {
                    "expected_items": [],
                    "confidence": 0.0,
                    "source": "dataset_no_match",
                    "metadata": {
                        "dataset": dataset_name,
                        "query": query,
                        "error": "Query not found in dataset",
                    },
                }

            except Exception as e:
                logger.warning(f"Failed to load from Phoenix dataset: {e}")

                # Try loading from local file as fallback
                import json
                import os

                dataset_path = f"data/datasets/{dataset_name}.json"
                if os.path.exists(dataset_path):
                    with open(dataset_path, "r") as f:
                        dataset_data = json.load(f)

                    # Search for query in dataset
                    for item in dataset_data.get("queries", []):
                        if item.get("query") == query:
                            return {
                                "expected_items": item.get("expected_items", []),
                                "confidence": 0.9,  # Slightly lower confidence for file-based
                                "source": "dataset_file",
                                "metadata": {
                                    "dataset": dataset_name,
                                    "file": dataset_path,
                                },
                            }

                # Dataset exists but query not found
                return {
                    "expected_items": [],
                    "confidence": 0.0,
                    "source": "dataset_no_match",
                    "metadata": {
                        "dataset": dataset_name,
                        "query": query,
                        "error": "Query not found in dataset",
                    },
                }

        except Exception as e:
            return {
                "expected_items": [],
                "confidence": 0.0,
                "source": "dataset_error",
                "metadata": {"error": str(e)},
            }


class BackendGroundTruthStrategy(GroundTruthStrategy):
    """Extract ground truth by re-querying backend with high precision."""

    async def extract_ground_truth(
        self, trace_data: Dict[str, Any], backend: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Extract ground truth using high-precision backend query."""
        if not backend:
            return {
                "expected_items": [],
                "confidence": 0.0,
                "source": "no_backend",
                "metadata": {"error": "Backend required for this strategy"},
            }

        # Delegate to schema-aware strategy but with high-precision settings
        strategy = SchemaAwareGroundTruthStrategy()

        # Modify trace data to request high precision
        modified_trace = trace_data.copy()
        modified_trace["metadata"] = modified_trace.get("metadata", {})
        modified_trace["metadata"]["high_precision"] = True

        return await strategy.extract_ground_truth(modified_trace, backend)


class HybridGroundTruthStrategy(GroundTruthStrategy):
    """Combine multiple strategies for best ground truth extraction."""

    async def extract_ground_truth(
        self, trace_data: Dict[str, Any], backend: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Extract ground truth using multiple strategies and combine results."""
        strategies = [SchemaAwareGroundTruthStrategy(), BackendGroundTruthStrategy()]

        results = []
        for strategy in strategies:
            try:
                result = await strategy.extract_ground_truth(trace_data, backend)
                if result["confidence"] > 0:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Strategy {strategy.__class__.__name__} failed: {e}")

        if not results:
            return {
                "expected_items": [],
                "confidence": 0.0,
                "source": "all_strategies_failed",
                "metadata": {"error": "All ground truth strategies failed"},
            }

        # Use the result with highest confidence
        best_result = max(results, key=lambda r: r["confidence"])
        best_result["source"] = f"hybrid_{best_result['source']}"
        best_result["metadata"]["strategies_tried"] = len(strategies)

        return best_result
