"""Inference modules for Cogniverse agents.

This package provides various inference strategies:
- RLMInference: Recursive Language Model inference for large context processing
- InstrumentedRLM: RLM with EventQueue integration for real-time progress tracking
- deno_check: Fast-fail probe for the Deno runtime that DSPy RLM requires
"""

from cogniverse_agents.inference.deno_check import (
    DenoNotInstalledError,
    assert_deno_available,
    is_deno_available,
)
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
    "DenoNotInstalledError",
    "assert_deno_available",
    "is_deno_available",
]
