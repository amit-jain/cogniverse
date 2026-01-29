"""Inference modules for Cogniverse agents.

This package provides various inference strategies:
- RLMInference: Recursive Language Model inference for large context processing
- InstrumentedRLM: RLM with EventQueue integration for real-time progress tracking
"""

from cogniverse_agents.inference.instrumented_rlm import (
    InstrumentedRLM,
    RLMCancelledError,
)
from cogniverse_agents.inference.rlm_inference import (
    RLMInference,
    RLMResult,
    RLMTimeoutError,
)

__all__ = [
    "RLMInference",
    "RLMResult",
    "RLMTimeoutError",
    "InstrumentedRLM",
    "RLMCancelledError",
]
