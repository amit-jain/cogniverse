"""
Integration test for CodingAgent.

Tests the full coding agent pipeline: DSPy planning + code generation +
sandboxed execution + output evaluation. Uses real Ollama LLM and real
OpenShell sandbox (started/destroyed per test module).

Requires: Ollama at localhost:11434, openshell CLI, Docker.
"""

import logging

import pytest

from cogniverse_runtime.sandbox_manager import SandboxManager
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


def _openshell_cli_available() -> bool:
    import subprocess as _sp

    try:
        return _sp.run(
            ["openshell", "--version"], capture_output=True, timeout=5
        ).returncode == 0
    except (FileNotFoundError, _sp.TimeoutExpired):
        return False


def _docker_available() -> bool:
    import subprocess as _sp

    try:
        return _sp.run(
            ["docker", "info"], capture_output=True, timeout=10
        ).returncode == 0
    except (FileNotFoundError, _sp.TimeoutExpired):
        return False


CODING_GW_NAME = "cogniverse-coding-test-gw"
CODING_GW_PORT = 19091


@pytest.fixture(scope="module")
def coding_sandbox():
    """Start an OpenShell gateway, yield a SandboxManager, destroy on teardown."""
    import subprocess

    if not _openshell_cli_available():
        pytest.skip("openshell CLI not installed")
    if not _docker_available():
        pytest.skip("Docker not running")

    subprocess.run(
        ["openshell", "gateway", "destroy", "--name", CODING_GW_NAME],
        capture_output=True, timeout=30, check=False,
    )

    result = subprocess.run(
        ["openshell", "gateway", "start", "--name", CODING_GW_NAME,
         "--port", str(CODING_GW_PORT)],
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        pytest.skip(f"Failed to start OpenShell gateway: {result.stderr}")

    subprocess.run(
        ["openshell", "gateway", "select", CODING_GW_NAME],
        capture_output=True, timeout=10, check=False,
    )

    manager = SandboxManager(policy_dir="configs/openshell", enabled=True)
    if not manager.available:
        pytest.skip("SandboxManager could not connect to gateway")

    yield manager

    manager.close()
    subprocess.run(
        ["openshell", "gateway", "destroy", "--name", CODING_GW_NAME],
        capture_output=True, timeout=60, check=False,
    )


@skip_if_no_ollama
class TestCodingAgentWithOllama:
    """Integration tests: real Ollama LLM + real OpenShell sandbox.

    All LLM-generated code executes inside an OpenShell sandbox with the
    coding_agent policy (Ollama-only network, /tmp/coding_workspace/ write).
    """

    @pytest.fixture
    def dspy_configured(self, dspy_lm):
        """Ensure DSPy is configured with a real LLM."""
        return dspy_lm

    @pytest.mark.asyncio
    async def test_coding_agent_generates_working_code(self, dspy_configured, coding_sandbox):
        """LLM generates add(2,3) → executes in OpenShell sandbox → stdout contains '5'."""
        from cogniverse_agents.coding_agent import (
            CodingAgent,
            CodingDeps,
            CodingInput,
        )

        deps = CodingDeps(tenant_id="test", sandbox_manager=coding_sandbox)
        agent = CodingAgent(deps=deps, sandbox_manager=coding_sandbox)

        input_data = CodingInput(
            task="Write a Python function called 'add' that takes two numbers and returns their sum. Print add(2, 3).",
            language="python",
            max_iterations=3,
        )

        result = await agent.process(input_data)

        assert result.plan, "Plan should be non-empty"
        assert result.iterations_used >= 1
        assert len(result.code_changes) > 0, "No code was generated"

        generated_code = result.code_changes[0]["content"]
        assert len(generated_code) > 10, f"Code too short: {generated_code!r}"

        # The agent already ran this in the sandbox — check its results
        assert len(result.execution_results) > 0, "No execution results"
        last_exec = result.execution_results[-1]

        logger.info(f"Sandbox execution: exit={last_exec.get('exit_code')}")
        logger.info(f"stdout: {last_exec.get('stdout', '')[:300]!r}")
        logger.info(f"stderr: {last_exec.get('stderr', '')[:300]!r}")
        logger.info(f"Generated code:\n{generated_code}")

        # Also verify independently via sandbox
        verify = coding_sandbox.exec_in_sandbox(
            agent_type="coding_agent",
            command=["python3", "-c", generated_code],
            timeout_seconds=30,
        )
        assert verify is not None, "Sandbox exec returned None"
        stdout = verify["stdout"].strip()

        logger.info(f"Independent sandbox run: exit={verify['exit_code']}, stdout={stdout!r}")

        assert verify["exit_code"] == 0, (
            f"Generated code failed in sandbox with exit {verify['exit_code']}.\n"
            f"stderr: {verify['stderr']}\ncode:\n{generated_code}"
        )
        assert "5" in stdout, (
            f"Expected '5' in stdout from add(2,3), got: {stdout!r}\n"
            f"code:\n{generated_code}"
        )

    @pytest.mark.asyncio
    async def test_coding_agent_with_search_fn(self, dspy_configured, coding_sandbox):
        """LLM generates email validation with search context → OpenShell sandbox verifies output."""
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

        deps = CodingDeps(tenant_id="test", sandbox_manager=coding_sandbox)
        agent = CodingAgent(deps=deps, search_fn=mock_search_fn, sandbox_manager=coding_sandbox)

        input_data = CodingInput(
            task="Write a function that validates email addresses using regex. Print whether 'test@example.com' is valid.",
            language="python",
            max_iterations=3,
        )

        result = await agent.process(input_data)

        assert search_called, "search_fn was never called by the agent"
        assert result.plan
        assert result.iterations_used >= 1
        assert len(result.code_changes) > 0, "No code generated"

        generated_code = result.code_changes[0]["content"]
        assert len(generated_code) > 10

        # Run generated code + our test harness in sandbox to verify correctness.
        # The LLM may or may not include a print call, so we append our own
        # assertion that exercises the function it defined.
        test_harness = (
            generated_code + "\n\n"
            "# Test harness: verify the function works\n"
            "result = validate_email('test@example.com')\n"
            "assert result, f'Expected True for test@example.com, got {result}'\n"
            "print('PASS:', result)\n"
        )
        verify = coding_sandbox.exec_in_sandbox(
            agent_type="coding_agent",
            command=["python3", "-c", test_harness],
            timeout_seconds=30,
        )
        assert verify is not None, "Sandbox exec returned None"
        stdout = verify["stdout"].strip().lower()

        logger.info(f"Sandbox run: exit={verify['exit_code']}, stdout={stdout!r}")
        logger.info(f"Generated code:\n{generated_code}")
        logger.info(f"Test harness:\n{test_harness}")

        assert verify["exit_code"] == 0, (
            f"Generated code + test harness failed in sandbox with exit {verify['exit_code']}.\n"
            f"stderr: {verify['stderr']}\nharness:\n{test_harness}"
        )
        assert "pass" in stdout, (
            f"Expected 'PASS' in output, got: {stdout!r}\n"
            f"harness:\n{test_harness}"
        )
