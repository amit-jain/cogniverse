"""
Structured output format for passing data through Inspect AI's string interface.

This module defines how we serialize our rich evaluation data (Phoenix traces,
search results, metadata) through Inspect AI's solver->scorer pipeline.
"""

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvaluationOutput:
    """
    Structured output from solver that can be serialized/deserialized.

    This contains all the data our scorers need to evaluate results.
    """

    query: str
    search_configs: dict[str, dict[str, Any]]  # config_key -> results
    phoenix_trace_id: str | None = None
    metadata: dict[str, Any] = None

    def to_json(self) -> str:
        """Serialize to JSON string for Inspect AI."""
        try:
            return json.dumps(asdict(self))
        except TypeError as e:
            # Debug: show what we're trying to serialize
            data = asdict(self)
            logger.error(f"Failed to serialize: {e}")
            logger.error(f"Data keys: {data.keys()}")
            for key, value in data.items():
                logger.error(f"  {key}: type={type(value)}, value={str(value)[:100]}")
            raise

    @classmethod
    def from_json(cls, json_str: str) -> "EvaluationOutput":
        """Deserialize a complete, successful solver result or raise."""
        try:
            data = json.loads(json_str)
            output = cls(**data)
            if not isinstance(output.query, str) or not output.query.strip():
                raise ValueError("query must be a nonempty string")
            if not isinstance(output.search_configs, dict) or not output.search_configs:
                raise ValueError("search_configs must contain a scored retrieval")
            for key, config in output.search_configs.items():
                if not isinstance(config, dict) or config.get("success") is not True:
                    raise ValueError(f"retrieval {key!r} failed: {config}")
                results = config.get("results")
                if not isinstance(results, list) or any(
                    not isinstance(result, dict) for result in results
                ):
                    raise ValueError(
                        f"retrieval {key!r} results must be a list of objects"
                    )
            return output
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid evaluation output: {e}") from e


def pack_solver_output(
    query: str,
    search_results: dict[str, Any],
    phoenix_trace_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Pack solver results into a JSON string for Inspect AI.

    Args:
        query: The search query
        search_results: Dict of config_key -> search results
        phoenix_trace_id: Optional Phoenix trace ID for this evaluation
        metadata: Additional metadata to pass to scorers

    Returns:
        JSON string containing all evaluation data
    """
    output = EvaluationOutput(
        query=query,
        search_configs=search_results,
        phoenix_trace_id=phoenix_trace_id,
        metadata=metadata or {},
    )
    result = output.to_json()
    if not result:
        logger.error(
            f"pack_solver_output returned empty string! Query: {query}, results: {search_results}"
        )
    return result


def unpack_solver_output(output_str: str) -> EvaluationOutput:
    """
    Unpack solver output from JSON string.

    Args:
        output_str: JSON string from solver

    Returns:
        EvaluationOutput object with all data
    """
    return EvaluationOutput.from_json(output_str)
