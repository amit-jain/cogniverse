"""Inference modules for Cogniverse agents.

This package provides various inference strategies:
- RLMInference: Recursive Language Model inference for large context processing
"""

from cogniverse_agents.inference.rlm_inference import RLMInference, RLMResult

__all__ = [
    "RLMInference",
    "RLMResult",
]
