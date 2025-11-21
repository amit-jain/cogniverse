"""
DSPy 3.0 + A2A Protocol Integration Base Class

This module provides the foundation for integrating DSPy 3.0 modules with the A2A
(Agent-to-Agent) protocol, enabling DSPy-powered agents to communicate seamlessly
with the existing agent ecosystem while leveraging advanced DSPy capabilities.
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import dspy
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from cogniverse_core.common.a2a_utils import A2AClient, AgentCard

logger = logging.getLogger(__name__)


class DSPyA2AConversionError(Exception):
    """Exception raised when conversion between A2A and DSPy formats fails"""

    pass


class AgentCapability(BaseModel):
    """Definition of an agent capability for A2A protocol"""

    name: str
    description: str
    input_schema: Dict[str, str]
    output_schema: Dict[str, str]
    examples: Optional[List[Dict[str, Any]]] = None


class DSPyA2AAgentBase(ABC):
    """
    Base class that bridges DSPy 3.0 modules with A2A protocol.

    Architecture:
    - A2A Protocol Layer: Handles external communication via standard endpoints
    - DSPy 3.0 Core: Provides advanced AI capabilities and optimization
    - Conversion Layer: Seamlessly converts between A2A and DSPy data formats

    Features:
    - Standard A2A endpoints (/agent.json, /tasks/send, /health)
    - Automatic A2A ↔ DSPy data conversion
    - Error handling and fallback mechanisms
    - Performance tracking and observability
    - Support for multi-modal inputs (text, images, video, audio)
    """

    def __init__(
        self,
        agent_name: str,
        agent_description: str,
        dspy_module: dspy.Module,
        capabilities: List[str],
        port: int = 8000,
        version: str = "1.0.0",
    ):
        """
        Initialize DSPy-A2A agent.

        Args:
            agent_name: Name of the agent for A2A protocol
            agent_description: Description for agent card
            dspy_module: Core DSPy 3.0 module for AI processing
            capabilities: List of agent capabilities
            port: Port for A2A HTTP server
            version: Agent version
        """
        # A2A Protocol Configuration
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.capabilities = capabilities
        self.port = port
        self.version = version

        # DSPy 3.0 Core
        self.dspy_module = dspy_module

        # FastAPI app for A2A endpoints
        self.app = FastAPI(
            title=f"{agent_name} A2A Agent",
            description=agent_description,
            version=version,
        )

        # A2A Client for inter-agent communication
        self.a2a_client = A2AClient()

        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0

        # Setup A2A protocol endpoints
        self._setup_a2a_endpoints()

        logger.info(f"Initialized {agent_name} with DSPy 3.0 + A2A integration")

    def _setup_a2a_endpoints(self):
        """Setup standard A2A protocol endpoints"""

        @self.app.get("/agent.json")
        async def get_agent_card():
            """Standard A2A agent card endpoint"""
            return AgentCard(
                name=self.agent_name,
                description=self.agent_description,
                url=f"http://localhost:{self.port}",
                version=self.version,
                protocol="a2a",
                protocol_version="1.0",
                capabilities=self.capabilities,
                skills=self._get_agent_skills(),
            ).dict()

        @self.app.post("/tasks/send")
        async def handle_task(task: Dict[str, Any]):
            """
            Standard A2A task endpoint.

            Flow: A2A Task → DSPy Input → DSPy Processing → A2A Response
            """
            start_time = time.time()
            self.request_count += 1

            try:
                # Step 1: Convert A2A task to DSPy input format
                dspy_input = self._a2a_to_dspy_input(task)
                logger.debug(
                    f"Converted A2A task to DSPy input: {list(dspy_input.keys())}"
                )

                # Step 2: Process with agent's core logic
                dspy_output = await self._process(dspy_input)
                logger.debug("Agent processing completed")

                # Step 3: Convert DSPy output back to A2A format
                a2a_response = self._dspy_to_a2a_output(dspy_output)

                # Record performance metrics
                processing_time = time.time() - start_time
                self.total_processing_time += processing_time

                logger.info(
                    f"Task processed successfully in {processing_time:.3f}s "
                    f"(avg: {self.total_processing_time/self.request_count:.3f}s)"
                )

                return a2a_response

            except Exception as e:
                self.error_count += 1
                processing_time = time.time() - start_time

                logger.error(
                    f"Task processing failed after {processing_time:.3f}s: {e}"
                )

                return {
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "processing_time": processing_time,
                    "agent": self.agent_name,
                }

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint with performance metrics"""
            return {
                "status": "healthy",
                "agent": self.agent_name,
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
            }

        @self.app.get("/metrics")
        async def get_metrics():
            """Detailed metrics endpoint for monitoring"""
            return {
                "agent_name": self.agent_name,
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
                    "average_processing_time": (
                        self.total_processing_time / self.request_count
                        if self.request_count > 0
                        else 0.0
                    ),
                },
                "dspy_module": {
                    "type": type(self.dspy_module).__name__,
                    "has_history": hasattr(self.dspy_module, "history"),
                    "has_tools": hasattr(self.dspy_module, "tools"),
                },
            }

    def _a2a_to_dspy_input(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert A2A Task format to DSPy module input format.

        A2A Task Structure:
        {
            "id": "task_id",
            "messages": [
                {
                    "role": "user",
                    "parts": [
                        {"type": "text", "text": "query"},
                        {"type": "data", "data": {...}},
                        {"type": "video", "video_data": bytes},
                        {"type": "image", "image_data": bytes}
                    ]
                }
            ]
        }
        """
        try:
            dspy_input = {
                "task_id": task.get("id", str(uuid.uuid4())),
                "messages": task.get("messages", []),
            }

            # Extract information from A2A message parts
            for message in task.get("messages", []):
                for part in message.get("parts", []):
                    part_type = part.get("type")

                    if part_type == "text":
                        dspy_input["query"] = part.get("text", "")

                    elif part_type == "data":
                        # Merge data part into DSPy input
                        data = part.get("data", {})
                        dspy_input.update(data)

                    elif part_type == "video":
                        dspy_input["video_data"] = part.get("video_data")
                        dspy_input["video_filename"] = part.get("filename")

                    elif part_type == "image":
                        dspy_input["image_data"] = part.get("image_data")
                        dspy_input["image_filename"] = part.get("filename")

                    else:
                        logger.warning(f"Unknown A2A part type: {part_type}")

            # Ensure query is present
            if "query" not in dspy_input:
                dspy_input["query"] = ""

            return dspy_input

        except Exception as e:
            raise DSPyA2AConversionError(
                f"Failed to convert A2A task to DSPy input: {e}"
            )

    @abstractmethod
    async def _process(self, dspy_input: Dict[str, Any]) -> Any:
        """
        Process input with agent's core logic - must be implemented by subclasses.

        Args:
            dspy_input: Converted input from A2A task format

        Returns:
            Agent-specific output (will be converted to A2A format)
        """
        pass

    @abstractmethod
    def _dspy_to_a2a_output(self, dspy_output: Any) -> Dict[str, Any]:
        """
        Convert DSPy output to A2A response format - must be implemented by subclasses.

        Args:
            dspy_output: Output from DSPy module

        Returns:
            A2A-compatible response dictionary
        """
        pass

    @abstractmethod
    def _get_agent_skills(self) -> List[Dict[str, Any]]:
        """
        Define agent skills for A2A protocol - must be implemented by subclasses.

        Returns:
            List of skill definitions for A2A agent card
        """
        pass

    async def call_other_agent(
        self, agent_url: str, query: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Call other A2A agents while maintaining protocol compliance.

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

    def run(self, host: str = "0.0.0.0"):
        """Run the A2A agent server"""
        logger.info(f"Starting {self.agent_name} A2A server on {host}:{self.port}")
        uvicorn.run(self.app, host=host, port=self.port)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring"""
        return {
            "agent": self.agent_name,
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


class SimpleDSPyA2AAgent(DSPyA2AAgentBase):
    """
    Simple example implementation of DSPy-A2A agent for testing and demonstration.
    """

    def __init__(self, port: int = 8000):
        # Create a simple DSPy module for testing
        class SimpleSignature(dspy.Signature):
            query: str = dspy.InputField(desc="Input query")
            response: str = dspy.OutputField(desc="Agent response")

        class SimpleModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.generator = dspy.ChainOfThought(SimpleSignature)

            def forward(self, query: str) -> dspy.Prediction:
                return self.generator(query=query)

        super().__init__(
            agent_name="SimpleDSPyAgent",
            agent_description="Simple DSPy 3.0 agent for testing A2A integration",
            dspy_module=SimpleModule(),
            capabilities=["text_processing", "simple_reasoning"],
            port=port,
        )

    async def _process(self, dspy_input: Dict[str, Any]) -> Any:
        """Process with simple DSPy module"""
        query = dspy_input.get("query", "")

        # Use DSPy 3.0 async capabilities if available
        try:
            with dspy.context(async_mode=True):
                result = await self.dspy_module.forward(query=query)
        except Exception:
            # Fallback to sync processing
            result = self.dspy_module.forward(query=query)

        return result

    def _dspy_to_a2a_output(self, dspy_output: Any) -> Dict[str, Any]:
        """Convert simple DSPy output to A2A format"""
        return {
            "status": "success",
            "response": str(
                dspy_output.response
                if hasattr(dspy_output, "response")
                else str(dspy_output)
            ),
            "agent": self.agent_name,
        }

    def _get_agent_skills(self) -> List[Dict[str, Any]]:
        """Define skills for simple agent"""
        return [
            {
                "name": "process_query",
                "description": "Process text queries with DSPy reasoning",
                "input_schema": {"query": "string"},
                "output_schema": {"response": "string", "status": "string"},
            }
        ]


# Example usage and testing
if __name__ == "__main__":
    # Test DSPy 3.0 integration
    import logging

    logging.basicConfig(level=logging.INFO)

    agent = SimpleDSPyA2AAgent(port=8001)
    print("🚀 Starting Simple DSPy-A2A Agent for testing...")
    print("Endpoints:")
    print("  - Agent Card: http://localhost:8001/agent.json")
    print("  - Health: http://localhost:8001/health")
    print("  - Metrics: http://localhost:8001/metrics")
    print("  - Tasks: POST http://localhost:8001/tasks/send")

    # Run the agent
    agent.run()
