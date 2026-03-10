"""
A2A Protocol Agent with Type Safety

Extends AgentBase with A2A agent configuration and DSPy integration.
Type safety is inherited from AgentBase.

In the unified runtime, HTTP concerns are handled by the runtime's
A2AStarletteApplication (a2a-sdk). This class only holds configuration
metadata (name, capabilities, version) and the optional DSPy module.

Usage:
    class RoutingInput(AgentInput):
        query: str
        context: Optional[str] = None

    class RoutingOutput(AgentOutput):
        recommended_agent: str
        confidence: float

    class RoutingDeps(AgentDeps):
        model_name: str = "smollm3:3b"

    class RoutingAgent(A2AAgent[RoutingInput, RoutingOutput, RoutingDeps]):
        async def _process_impl(self, input: RoutingInput) -> RoutingOutput:
            return RoutingOutput(recommended_agent="search", confidence=0.9)
"""

import logging
from typing import Any, Dict, Generic, List, Optional

import dspy
from pydantic import BaseModel

from cogniverse_core.agents.base import (
    AgentBase,
    DepsT,
    InputT,
    OutputT,
)

logger = logging.getLogger(__name__)


class A2AAgentConfig(BaseModel):
    """Configuration for A2A agent."""

    agent_name: str
    agent_description: str
    capabilities: List[str]
    port: int = 8000
    version: str = "1.0.0"


class A2AAgent(AgentBase[InputT, OutputT, DepsT], Generic[InputT, OutputT, DepsT]):
    """
    Type-safe agent with A2A protocol metadata and DSPy integration.

    Combines:
    - AgentBase: Type-safe input/output with Pydantic validation
    - A2A Metadata: Agent name, capabilities, version (used by runtime)
    - DSPy Integration: AI module support with optimization

    HTTP server concerns are handled by the runtime's A2A server.

    Type Parameters:
        InputT: Agent input type (extends AgentInput)
        OutputT: Agent output type (extends AgentOutput)
        DepsT: Agent dependencies type (extends AgentDeps)
    """

    def __init__(
        self,
        deps: DepsT,
        config: A2AAgentConfig,
        dspy_module: Optional[dspy.Module] = None,
    ) -> None:
        """
        Initialize A2A agent.

        Args:
            deps: Typed agent dependencies
            config: A2A agent configuration (name, capabilities, etc.)
            dspy_module: Optional DSPy module for AI processing
        """
        super().__init__(deps=deps)

        self.config = config
        self.agent_name = config.agent_name
        self.agent_description = config.agent_description
        self.capabilities = config.capabilities
        self.port = config.port
        self.version = config.version

        self.dspy_module = dspy_module

        logger.info(
            f"Initialized {self.agent_name} "
            f"with types: Input={self._input_type.__name__}, "
            f"Output={self._output_type.__name__}"
        )

    def _get_skills(self) -> List[Dict[str, Any]]:
        """Generate A2A skills from typed schemas."""
        return [
            {
                "name": "process",
                "description": f"Process {self._input_type.__name__} and return {self._output_type.__name__}",
                "input_schema": self.get_input_schema(),
                "output_schema": self.get_output_schema(),
            }
        ]
