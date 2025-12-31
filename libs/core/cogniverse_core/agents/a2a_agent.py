"""
A2A Protocol Agent with Type Safety

Extends AgentBase with Google A2A protocol support, DSPy integration,
and FastAPI endpoints. Type safety is inherited from AgentBase.

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
        async def process(self, input: RoutingInput) -> RoutingOutput:
            # Full type safety, A2A endpoints auto-generated
            return RoutingOutput(recommended_agent="search", confidence=0.9)
"""

import logging
import time
import uuid
from typing import Any, Dict, Generic, List, Optional

import dspy
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from cogniverse_core.agents.base import (
    AgentBase,
    AgentValidationError,
    DepsT,
    InputT,
    OutputT,
)
from cogniverse_core.common.a2a_utils import A2AClient, AgentCard

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
    Type-safe agent with A2A protocol support.

    Combines:
    - AgentBase: Type-safe input/output with Pydantic
    - A2A Protocol: Standard endpoints, agent card, inter-agent communication
    - DSPy Integration: AI module support with optimization
    - FastAPI: HTTP server for A2A endpoints

    Type Parameters:
        InputT: Agent input type (extends AgentInput)
        OutputT: Agent output type (extends AgentOutput)
        DepsT: Agent dependencies type (extends AgentDeps)

    Example:
        class MyAgent(A2AAgent[MyInput, MyOutput, MyDeps]):
            async def process(self, input: MyInput) -> MyOutput:
                return MyOutput(result=input.query.upper())

        # Usage
        deps = MyDeps(tenant_id="tenant-1")
        agent = MyAgent(
            deps=deps,
            config=A2AAgentConfig(
                agent_name="my_agent",
                agent_description="My agent",
                capabilities=["text_processing"],
            ),
        )
        agent.run()  # Starts FastAPI server with A2A endpoints
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
            config: A2A agent configuration
            dspy_module: Optional DSPy module for AI processing
        """
        # Initialize base class (validates deps)
        super().__init__(deps=deps)

        # A2A configuration
        self.config = config
        self.agent_name = config.agent_name
        self.agent_description = config.agent_description
        self.capabilities = config.capabilities
        self.port = config.port
        self.version = config.version

        # DSPy module (optional)
        self.dspy_module = dspy_module

        # A2A client for inter-agent communication
        self.a2a_client = A2AClient()

        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0

        # FastAPI app
        self.app = FastAPI(
            title=f"{self.agent_name} A2A Agent",
            description=self.agent_description,
            version=self.version,
        )

        # Setup A2A endpoints
        self._setup_a2a_endpoints()

        logger.info(
            f"Initialized {self.agent_name} for tenant {self.tenant_id} "
            f"with types: Input={self._input_type.__name__}, "
            f"Output={self._output_type.__name__}"
        )

    def _setup_a2a_endpoints(self) -> None:
        """Setup standard A2A protocol endpoints."""

        @self.app.get("/.well-known/agent-card.json")
        async def get_agent_card() -> Dict[str, Any]:
            """Standard A2A agent card endpoint (Google spec)."""
            return AgentCard(
                name=self.agent_name,
                description=self.agent_description,
                url=f"http://localhost:{self.port}",
                version=self.version,
                protocol="a2a",
                protocol_version="1.0",
                capabilities=self.capabilities,
                skills=self._get_skills(),
            ).model_dump()

        @self.app.get("/agent.json")
        async def get_agent_card_legacy() -> Dict[str, Any]:
            """Legacy agent card endpoint."""
            return await get_agent_card()

        @self.app.post("/tasks/send")
        async def handle_task(task: Dict[str, Any]) -> Dict[str, Any]:
            """
            Standard A2A task endpoint.

            Flow: A2A Task → Typed Input → process() → Typed Output → A2A Response
            """
            start_time = time.time()
            self.request_count += 1

            try:
                # Extract input from A2A task
                raw_input = self._extract_input_from_task(task)
                logger.debug(f"Extracted input: {list(raw_input.keys())}")

                # Validate and convert to typed input
                typed_input = self.validate_input(raw_input)

                # Process with typed method
                typed_output = await self.process(typed_input)

                # Convert to A2A response
                a2a_response = self._create_a2a_response(typed_output)

                # Track performance
                processing_time = time.time() - start_time
                self.total_processing_time += processing_time

                logger.info(
                    f"Task processed in {processing_time:.3f}s "
                    f"(avg: {self.total_processing_time/self.request_count:.3f}s)"
                )

                return a2a_response

            except AgentValidationError as e:
                self.error_count += 1
                processing_time = time.time() - start_time
                logger.error(f"Validation failed after {processing_time:.3f}s: {e}")

                return {
                    "status": "error",
                    "error": str(e),
                    "error_type": "ValidationError",
                    "processing_time": processing_time,
                    "agent": self.agent_name,
                }

            except Exception as e:
                self.error_count += 1
                processing_time = time.time() - start_time
                logger.error(f"Task failed after {processing_time:.3f}s: {e}")

                return {
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "processing_time": processing_time,
                    "agent": self.agent_name,
                }

        @self.app.get("/health")
        async def health_check() -> Dict[str, Any]:
            """Health check with performance metrics."""
            return {
                "status": "healthy",
                "agent": self.agent_name,
                "tenant_id": self.tenant_id,
                "version": self.version,
                "metrics": {
                    "requests_processed": self.request_count,
                    "error_count": self.error_count,
                    "success_rate": (
                        (self.request_count - self.error_count) / self.request_count
                        if self.request_count > 0
                        else 1.0
                    ),
                    "avg_processing_time": (
                        self.total_processing_time / self.request_count
                        if self.request_count > 0
                        else 0.0
                    ),
                },
                "types": {
                    "input": self._input_type.__name__,
                    "output": self._output_type.__name__,
                    "deps": self._deps_type.__name__,
                },
            }

        @self.app.get("/metrics")
        async def get_metrics() -> Dict[str, Any]:
            """Detailed metrics endpoint."""
            return {
                "agent_name": self.agent_name,
                "tenant_id": self.tenant_id,
                "performance": {
                    "total_requests": self.request_count,
                    "successful_requests": self.request_count - self.error_count,
                    "error_count": self.error_count,
                    "success_rate": (
                        (self.request_count - self.error_count) / self.request_count
                        if self.request_count > 0
                        else 1.0
                    ),
                    "total_processing_time": self.total_processing_time,
                    "avg_processing_time": (
                        self.total_processing_time / self.request_count
                        if self.request_count > 0
                        else 0.0
                    ),
                },
                "types": {
                    "input_schema": self.get_input_schema(),
                    "output_schema": self.get_output_schema(),
                },
                "dspy_module": {
                    "type": (
                        type(self.dspy_module).__name__
                        if self.dspy_module
                        else None
                    ),
                    "available": self.dspy_module is not None,
                },
            }

        @self.app.get("/schema")
        async def get_schemas() -> Dict[str, Any]:
            """Get input/output schemas for this agent."""
            return {
                "input": self.get_input_schema(),
                "output": self.get_output_schema(),
                "deps": self.get_deps_schema(),
            }

    def _extract_input_from_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract typed input from A2A task format.

        A2A Task Structure:
        {
            "id": "task_id",
            "messages": [
                {
                    "role": "user",
                    "parts": [
                        {"type": "text", "text": "query"},
                        {"type": "data", "data": {...}},
                    ]
                }
            ]
        }
        """
        extracted = {
            "task_id": task.get("id", str(uuid.uuid4())),
        }

        # Extract from A2A message parts
        for message in task.get("messages", []):
            for part in message.get("parts", []):
                part_type = part.get("type")

                if part_type == "text":
                    extracted["query"] = part.get("text", "")

                elif part_type == "data":
                    # Merge data part into extracted input
                    data = part.get("data", {})
                    extracted.update(data)

                elif part_type == "video":
                    extracted["video_data"] = part.get("video_data")
                    extracted["video_filename"] = part.get("filename")

                elif part_type == "image":
                    extracted["image_data"] = part.get("image_data")
                    extracted["image_filename"] = part.get("filename")

                elif part_type == "file":
                    extracted["file_uri"] = part.get("file_uri")
                    extracted["mime_type"] = part.get("mime_type")

        return extracted

    def _create_a2a_response(self, output: OutputT) -> Dict[str, Any]:
        """Convert typed output to A2A response format."""
        return {
            "status": "success",
            "agent": self.agent_name,
            "tenant_id": self.tenant_id,
            **output.model_dump(),
        }

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

    async def call_agent(
        self,
        agent_url: str,
        query: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Call another A2A agent.

        Args:
            agent_url: URL of target A2A agent
            query: Query to send
            **kwargs: Additional parameters

        Returns:
            Response from target agent
        """
        try:
            return await self.a2a_client.send_task(agent_url, query, **kwargs)
        except Exception as e:
            logger.error(f"Failed to call agent {agent_url}: {e}")
            return {
                "error": f"Agent communication failed: {e}",
                "status": "failed",
                "target_agent": agent_url,
            }

    def start(self, host: str = "0.0.0.0") -> None:
        """Start the A2A agent HTTP server."""
        logger.info(f"Starting {self.agent_name} on {host}:{self.port}")
        uvicorn.run(self.app, host=host, port=self.port)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "agent": self.agent_name,
            "tenant_id": self.tenant_id,
            "requests_processed": self.request_count,
            "error_count": self.error_count,
            "success_rate": (
                (self.request_count - self.error_count) / self.request_count
                if self.request_count > 0
                else 1.0
            ),
            "avg_processing_time": (
                self.total_processing_time / self.request_count
                if self.request_count > 0
                else 0.0
            ),
        }
