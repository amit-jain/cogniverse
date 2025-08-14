"""
Structured output format for passing data through Inspect AI's string interface.

This module defines how we serialize our rich evaluation data (Phoenix traces,
search results, metadata) through Inspect AI's solver->scorer pipeline.
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationOutput:
    """
    Structured output from solver that can be serialized/deserialized.
    
    This contains all the data our scorers need to evaluate results.
    """
    query: str
    search_configs: Dict[str, Dict[str, Any]]  # config_key -> results
    phoenix_trace_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
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
    def from_json(cls, json_str: str) -> 'EvaluationOutput':
        """Deserialize from JSON string."""
        try:
            data = json.loads(json_str)
            return cls(**data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse evaluation output: {e}")
            # Return empty output on error
            return cls(query="", search_configs={})


def pack_solver_output(
    query: str,
    search_results: Dict[str, Any],
    phoenix_trace_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
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
        metadata=metadata or {}
    )
    result = output.to_json()
    if not result:
        logger.error(f"pack_solver_output returned empty string! Query: {query}, results: {search_results}")
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