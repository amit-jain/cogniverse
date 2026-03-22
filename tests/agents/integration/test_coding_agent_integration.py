"""
Integration test for CodingAgent.

Tests the full coding agent pipeline: DSPy planning + code generation +
local execution + output evaluation. Uses real Ollama LLM when available.

Requires: Ollama with qwen3:4b model running at localhost:11434.
"""

import logging

import pytest

from tests.agents.integration.conftest import skip_if_no_ollama

logger = logging.getLogger(__name__)


class TestCodingAgentUnit:
    """Unit-level tests that verify CodingAgent wiring without infrastructure."""

    def test_coding_agent_instantiation(self):
        """CodingAgent can be instantiated with minimal deps."""
        from cogniverse_agents.coding_agent import (
            CodingAgent,
            CodingDeps,
        )

        deps = CodingDeps(tenant_id="test")
        agent = CodingAgent(deps=deps)

        assert agent.agent_name == "coding_agent"
        assert "coding" in agent.capabilities

    def test_coding_input_validation(self):
        """CodingInput validates required fields."""
        from cogniverse_agents.coding_agent import CodingInput

        inp = CodingInput(task="Write a hello world function")
        assert inp.task == "Write a hello world function"
        assert inp.language == "python"
        assert inp.max_iterations == 5

    def test_coding_output_schema(self):
        """CodingOutput has expected fields."""
        from cogniverse_agents.coding_agent import CodingOutput

        out = CodingOutput(
            plan="Step 1: write function",
            code_changes=[{"file_path": "test.py", "content": "print('hi')"}],
            execution_results=[{"exit_code": 0, "stdout": "hi"}],
            summary="Done",
            iterations_used=1,
            files_modified=["test.py"],
        )
        dumped = out.model_dump()
        assert dumped["plan"] == "Step 1: write function"
        assert len(dumped["code_changes"]) == 1
        assert dumped["iterations_used"] == 1

    def test_coding_agent_schema_export(self):
        """CodingAgent exports valid JSON schemas."""
        from cogniverse_agents.coding_agent import CodingAgent

        input_schema = CodingAgent.get_input_schema()
        output_schema = CodingAgent.get_output_schema()

        assert "task" in input_schema["properties"]
        assert "plan" in output_schema["properties"]
        assert "code_changes" in output_schema["properties"]

    def test_coding_deps_accepts_sandbox_manager(self):
        """CodingDeps can hold a sandbox_manager reference."""
        from cogniverse_agents.coding_agent import CodingDeps

        deps = CodingDeps(tenant_id="test", sandbox_manager="mock_manager")
        assert deps.sandbox_manager == "mock_manager"

    def test_dspy_signatures_defined(self):
        """DSPy signatures are importable and have expected fields."""
        from cogniverse_agents.coding_agent import (
            CodeGenerationSignature,
            OutputEvaluationSignature,
            TaskPlanningSignature,
        )

        # DSPy Signatures expose fields via model_fields (Pydantic)
        planning_fields = set(TaskPlanningSignature.model_fields.keys())
        assert "task" in planning_fields
        assert "plan" in planning_fields

        gen_fields = set(CodeGenerationSignature.model_fields.keys())
        assert "code" in gen_fields
        assert "test_command" in gen_fields

        eval_fields = set(OutputEvaluationSignature.model_fields.keys())
        assert "is_successful" in eval_fields
        assert "feedback" in eval_fields


class TestCodingAgentDispatchWiring:
    """Test that the coding agent is properly wired in the dispatcher."""

    def test_config_loader_has_coding_agent(self):
        """ConfigLoader knows about coding_agent."""
        from cogniverse_runtime.config_loader import AGENT_CAPABILITIES, ConfigLoader

        assert "coding_agent" in AGENT_CAPABILITIES
        assert "coding" in AGENT_CAPABILITIES["coding_agent"]
        assert "coding_agent" in ConfigLoader.AGENT_CLASSES

    def test_config_json_has_coding_agent(self):
        """config.json has coding_agent entry."""
        import json
        from pathlib import Path

        config_path = Path(__file__).resolve().parents[3] / "configs" / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        assert "coding_agent" in config["agents"]
        assert config["agents"]["coding_agent"]["enabled"] is True

    def test_config_json_has_code_profile(self):
        """config.json has code_lateon_mv profile."""
        import json
        from pathlib import Path

        config_path = Path(__file__).resolve().parents[3] / "configs" / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        profiles = config["backend"]["profiles"]
        assert "code_lateon_mv" in profiles

        profile = profiles["code_lateon_mv"]
        assert profile["type"] == "code"
        assert profile["embedding_model"] == "lightonai/LateOn-Code-edge"
        assert profile["model_loader"] == "colbert"
        assert profile["strategies"]["segmentation"]["class"] == "CodeSegmentationStrategy"
        assert profile["strategies"]["embedding"]["class"] == "CodeTextEmbeddingStrategy"
        assert profile["schema_config"]["embedding_dim"] == 48
        assert profile["schema_config"]["binary_dim"] == 6
        assert profile["schema_config"]["num_patches"] == 2048


@skip_if_no_ollama
class TestCodingAgentWithOllama:
    """Integration tests requiring a real Ollama LLM instance."""

    @pytest.fixture
    def dspy_configured(self, dspy_lm):
        """Ensure DSPy is configured with a real LLM."""
        return dspy_lm

    @pytest.mark.asyncio
    async def test_coding_agent_generates_code(self, dspy_configured):
        """Full coding agent: plan → generate → execute → evaluate.

        Verifies the agent produces actual Python code that gets written
        to /tmp/coding_workspace/ and executed, with real stdout/stderr
        captured in execution_results.
        """
        from cogniverse_agents.coding_agent import (
            CodingAgent,
            CodingDeps,
            CodingInput,
        )

        deps = CodingDeps(tenant_id="test")
        agent = CodingAgent(deps=deps)

        input_data = CodingInput(
            task="Write a Python function called 'add' that takes two numbers and returns their sum. Print add(2, 3).",
            language="python",
            max_iterations=2,
        )

        result = await agent.process(input_data)

        # Plan must be non-empty — the LLM actually planned something
        assert result.plan, "Plan should be non-empty"
        assert result.iterations_used >= 1

        # Code changes must contain actual generated code
        assert len(result.code_changes) > 0, "No code was generated"
        code_change = result.code_changes[0]
        assert code_change["file_path"].endswith(".py"), (
            f"Expected .py file, got: {code_change['file_path']}"
        )
        generated_code = code_change["content"]
        assert len(generated_code) > 10, (
            f"Generated code is suspiciously short: {generated_code!r}"
        )
        assert "def " in generated_code or "add" in generated_code, (
            f"Generated code doesn't contain a function definition: {generated_code[:200]}"
        )

        # Execution results must exist — code was actually run
        assert len(result.execution_results) > 0, "No execution results"
        last_exec = result.execution_results[-1]
        assert "exit_code" in last_exec, "Execution result missing exit_code"
        assert "stdout" in last_exec, "Execution result missing stdout"
        assert "stderr" in last_exec, "Execution result missing stderr"

        # Files modified should reference the workspace path
        assert len(result.files_modified) > 0
        assert "/tmp/coding_workspace/" in result.files_modified[0]

        logger.info(f"Iterations: {result.iterations_used}")
        logger.info(f"Generated code ({len(generated_code)} chars):\n{generated_code[:300]}")
        logger.info(f"Exit code: {last_exec['exit_code']}")
        logger.info(f"stdout: {last_exec['stdout'][:200]}")
        logger.info(f"stderr: {last_exec['stderr'][:200]}")

    @pytest.mark.asyncio
    async def test_coding_agent_with_search_fn(self, dspy_configured):
        """Coding agent with mock search context: generates code, writes file, executes."""
        from cogniverse_agents.coding_agent import (
            CodingAgent,
            CodingDeps,
            CodingInput,
        )

        search_called = False

        async def mock_search_fn(query: str, tenant_id: str):
            nonlocal search_called
            search_called = True
            return [
                {
                    "document_id": "example_utils",
                    "metadata": {
                        "file": "utils.py",
                        "chunk_name": "validate_email",
                        "extracted_text": "def validate_email(email: str) -> bool:\n    return '@' in email",
                    },
                }
            ]

        deps = CodingDeps(tenant_id="test")
        agent = CodingAgent(deps=deps, search_fn=mock_search_fn)

        input_data = CodingInput(
            task="Write a function that validates email addresses using regex. Print whether 'test@example.com' is valid.",
            language="python",
            max_iterations=2,
        )

        result = await agent.process(input_data)

        # search_fn was actually called
        assert search_called, "search_fn was never called by the agent"

        # Code was generated and executed
        assert result.plan
        assert result.iterations_used >= 1
        assert len(result.code_changes) > 0, "No code generated"
        assert len(result.execution_results) > 0, "No execution results"

        generated_code = result.code_changes[0]["content"]
        assert len(generated_code) > 10
        logger.info(f"Search called: {search_called}")
        logger.info(f"Generated code:\n{generated_code[:300]}")
        logger.info(f"Execution: exit={result.execution_results[-1].get('exit_code')}")
