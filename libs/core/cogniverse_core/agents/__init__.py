"""
Type-safe Agent Base Classes

This module provides the foundation for all agents in the Cogniverse framework:

- AgentBase: Generic base with type-safe input/output/dependencies
- A2AAgent: A2A protocol + DSPy integration on top of AgentBase
- AgentInput/AgentOutput/AgentDeps: Pydantic base models for type safety
"""

from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import (
    AgentBase,
    AgentDeps,
    AgentInput,
    AgentOutput,
    AgentValidationError,
    DepsT,
    InputT,
    OutputT,
)

__all__ = [
    # Base classes
    "AgentBase",
    "AgentInput",
    "AgentOutput",
    "AgentDeps",
    "AgentValidationError",
    # Type variables
    "InputT",
    "OutputT",
    "DepsT",
    # A2A Agent
    "A2AAgent",
    "A2AAgentConfig",
]
