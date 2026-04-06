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
        from cogniverse_runtime.config_loader import ConfigLoader

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


CODING_GW_NAME = "cogniverse-coding-test-gw"
CODING_GW_PORT = 19091


@pytest.fixture(scope="module")
def code_search_infra(vespa_with_schema):
    """Deploy code_lateon_mv schema into the existing test Vespa, ingest real code,
    start OpenShell gateway.

    Uses vespa_with_schema for the Vespa container, deploys the native 48-dim
    code schema alongside the existing video schema, feeds real code segments
    with LateOn-Code-edge embeddings.
    """
    import subprocess
    import time
    from pathlib import Path

    import numpy as np
    import requests

    if not _openshell_cli_available():
        pytest.skip("openshell CLI not installed")

    base_url = vespa_with_schema["base_url"]
    manager = vespa_with_schema["manager"]

    # --- 1. Deploy code_lateon_mv schema via SchemaRegistry (tenant-scoped) ---
    from cogniverse_foundation.config.utils import create_default_config_manager

    config_manager = create_default_config_manager()
    backend = manager.get_backend_via_registry(
        tenant_id="test_tenant",
        config_manager=config_manager,
        backend_type="ingestion",
    )
    tenant_schema = backend.schema_registry.deploy_schema(
        tenant_id="test_tenant",
        base_schema_name="code_lateon_mv",
        force=True,
    )
    logger.info(f"Deployed tenant schema: {tenant_schema}")

    # Wait for schema to be active
    for i in range(30):
        try:
            resp = requests.get(
                f"{base_url}/search/",
                params={"query": "test", "restrict": tenant_schema},
                timeout=5,
            )
            if resp.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(2)

    # --- 2. Ingest real code with LateOn-Code-edge ---
    from pylate import models as pylate_models

    from cogniverse_runtime.ingestion.strategies import CodeSegmentationStrategy

    colbert_model = pylate_models.ColBERT("lightonai/LateOn-Code-edge", device="cpu")
    strategy = CodeSegmentationStrategy(languages=["python"])

    repo_root = Path(__file__).resolve().parents[3]
    source_files = [
        repo_root / "libs" / "agents" / "cogniverse_agents" / "deep_research_agent.py",
        repo_root / "libs" / "agents" / "cogniverse_agents" / "coding_agent.py",
        repo_root / "libs" / "agents" / "cogniverse_agents" / "search_agent.py",
    ]

    all_segments = []
    for f in source_files:
        if f.exists():
            all_segments.extend(strategy.parse_file(f))

    segments_to_ingest = all_segments[:15]
    texts = [seg["content"][:8192] for seg in segments_to_ingest]
    doc_embeddings = colbert_model.encode(texts, is_query=False)

    schema_name = tenant_schema
    for idx, (seg, emb) in enumerate(zip(segments_to_ingest, doc_embeddings)):
        emb_np = np.array(emb, dtype=np.float32)
        if emb_np.shape[0] > 2048:
            emb_np = emb_np[:2048]

        float_dict = {
            str(i): emb_np[i].tolist() for i in range(emb_np.shape[0])
        }
        binary = np.packbits(
            np.where(emb_np > 0, 1, 0).astype(np.uint8), axis=1
        ).astype(np.int8)
        binary_dict = {
            str(i): binary[i].tolist() for i in range(binary.shape[0])
        }

        meta = seg["metadata"]
        doc = {
            "fields": {
                "code_id": f"seg_{idx}",
                "file_path": meta.get("file", ""),
                "chunk_name": meta.get("name", ""),
                "chunk_type": meta.get("type", ""),
                "language": meta.get("language", "python"),
                "signature": meta.get("signature", ""),
                "line_start": meta.get("line_start", 0),
                "line_end": meta.get("line_end", 0),
                "source_code": seg["content"][:4096],
                "embedding": float_dict,
                "embedding_binary": binary_dict,
            }
        }
        resp = requests.post(
            f"{base_url}/document/v1/{schema_name}/{schema_name}/docid/seg_{idx}",
            json=doc, timeout=10,
        )
        assert resp.status_code == 200, (
            f"Feed failed for seg_{idx}: {resp.status_code} - {resp.text}"
        )

    logger.info(f"Ingested {len(segments_to_ingest)} code segments into Vespa")
    time.sleep(3)

    # --- 3. Start OpenShell gateway ---
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

    sandbox = SandboxManager(policy_dir="configs/openshell", enabled=True)
    if not sandbox.available:
        pytest.skip("SandboxManager could not connect to gateway")

    yield {
        "sandbox": sandbox,
        "vespa_url": base_url,
        "schema_name": schema_name,
        "colbert_model": colbert_model,
    }

    sandbox.close()
    subprocess.run(
        ["openshell", "gateway", "destroy", "--name", CODING_GW_NAME],
        capture_output=True, timeout=60, check=False,
    )


def _build_vespa_search_fn(vespa_url, schema_name, colbert_model):
    """Build a real search function that queries Vespa with LateOn-Code-edge."""

    async def search_fn(query: str, tenant_id: str):
        import numpy as np
        import requests as _requests

        query_emb = colbert_model.encode([query], is_query=True)[0]
        query_np = np.array(query_emb, dtype=np.float32)

        qt_cells = []
        for tok_idx in range(query_np.shape[0]):
            for v_idx in range(query_np.shape[1]):
                qt_cells.append({
                    "address": {"querytoken": str(tok_idx), "v": str(v_idx)},
                    "value": float(query_np[tok_idx, v_idx]),
                })

        resp = _requests.post(
            f"{vespa_url}/search/",
            json={
                "yql": f"select * from {schema_name} where true",
                "hits": 5,
                "ranking.profile": "float_float",
                "input.query(qt)": {"cells": qt_cells},
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return []

        hits = resp.json().get("root", {}).get("children", [])
        return [
            {
                "document_id": h["fields"].get("code_id", ""),
                "score": h.get("relevance", 0),
                "metadata": {
                    "file": h["fields"].get("file_path", ""),
                    "chunk_name": h["fields"].get("chunk_name", ""),
                    "extracted_text": h["fields"].get("source_code", ""),
                },
            }
            for h in hits
        ]

    return search_fn


@skip_if_no_ollama
class TestCodingAgentWithOllama:
    """Full integration: real Ollama + real Vespa code search + real OpenShell sandbox.

    code_search_infra deploys code_lateon_mv into the test Vespa, ingests real
    source with LateOn-Code-edge, starts OpenShell. The agent's search_fn hits
    real Vespa with real embeddings. Generated code runs in the sandbox.
    """

    @pytest.fixture
    def dspy_configured(self):
        """Configure DSPy with the coding_agent's resolved LLM config."""
        import dspy

        from cogniverse_foundation.config.llm_factory import create_dspy_lm
        from cogniverse_foundation.config.utils import (
            create_default_config_manager,
            get_config,
        )

        cm = create_default_config_manager()
        config = get_config(tenant_id="default", config_manager=cm)
        endpoint = config.get_llm_config().resolve("coding_agent")
        lm = create_dspy_lm(endpoint)
        dspy.configure(lm=lm)
        return lm

    @pytest.mark.asyncio
    async def test_coding_agent_generates_working_code(self, dspy_configured, code_search_infra):
        """LLM generates add(2,3) with real code search context,
        executes in OpenShell sandbox, stdout contains '5'."""
        from cogniverse_agents.coding_agent import (
            CodingAgent,
            CodingDeps,
            CodingInput,
        )

        infra = code_search_infra
        sandbox = infra["sandbox"]
        search_fn = _build_vespa_search_fn(
            infra["vespa_url"], infra["schema_name"], infra["colbert_model"]
        )

        deps = CodingDeps(tenant_id="test", sandbox_manager=sandbox)
        agent = CodingAgent(deps=deps, search_fn=search_fn, sandbox_manager=sandbox)

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

        verify = sandbox.exec_in_sandbox(
            agent_type="coding_agent",
            command=["python3", "-c", generated_code],
            timeout_seconds=30,
        )
        assert verify is not None, "Sandbox exec returned None"
        stdout = verify["stdout"].strip()

        logger.info(f"Sandbox run: exit={verify['exit_code']}, stdout={stdout!r}")
        logger.info(f"Generated code:\n{generated_code}")

        assert verify["exit_code"] == 0, (
            f"Generated code failed in sandbox with exit {verify['exit_code']}.\n"
            f"stderr: {verify['stderr']}\ncode:\n{generated_code}"
        )
        assert "5" in stdout, (
            f"Expected '5' in stdout from add(2,3), got: {stdout!r}\n"
            f"code:\n{generated_code}"
        )

    @pytest.mark.asyncio
    async def test_coding_agent_with_real_code_search(self, dspy_configured, code_search_infra):
        """Agent searches real Vespa code index with LateOn-Code-edge,
        generates email validation, executes in OpenShell sandbox."""
        from cogniverse_agents.coding_agent import (
            CodingAgent,
            CodingDeps,
            CodingInput,
        )

        infra = code_search_infra
        sandbox = infra["sandbox"]

        search_called = False
        raw_search_fn = _build_vespa_search_fn(
            infra["vespa_url"], infra["schema_name"], infra["colbert_model"]
        )

        async def tracked_search_fn(query: str, tenant_id: str):
            nonlocal search_called
            search_called = True
            return await raw_search_fn(query, tenant_id)

        deps = CodingDeps(tenant_id="test", sandbox_manager=sandbox)
        agent = CodingAgent(
            deps=deps, search_fn=tracked_search_fn, sandbox_manager=sandbox
        )

        input_data = CodingInput(
            task="Write a function that validates email addresses using regex. Print whether 'test@example.com' is valid.",
            language="python",
            max_iterations=3,
        )

        result = await agent.process(input_data)

        assert search_called, "search_fn was never called"
        assert result.plan
        assert result.iterations_used >= 1
        assert len(result.code_changes) > 0, "No code generated"

        generated_code = result.code_changes[0]["content"]
        assert len(generated_code) > 10

        # Find the function name the LLM chose (it may not be "validate_email")
        import re as _re

        fn_match = _re.search(r"def\s+(\w+)\s*\(", generated_code)
        fn_name = fn_match.group(1) if fn_match else "validate_email"

        test_harness = (
            generated_code + "\n\n"
            f"result = {fn_name}('test@example.com')\n"
            "assert result, f'Expected True for test@example.com, got {result}'\n"
            "print('PASS:', result)\n"
        )
        verify = sandbox.exec_in_sandbox(
            agent_type="coding_agent",
            command=["python3", "-c", test_harness],
            timeout_seconds=30,
        )
        assert verify is not None, "Sandbox exec returned None"
        stdout = verify["stdout"].strip().lower()

        logger.info(f"Sandbox run: exit={verify['exit_code']}, stdout={stdout!r}")
        logger.info(f"Generated code:\n{generated_code}")

        assert verify["exit_code"] == 0, (
            f"Code + harness failed in sandbox: exit {verify['exit_code']}\n"
            f"stderr: {verify['stderr']}\nharness:\n{test_harness}"
        )
        assert "pass" in stdout, (
            f"Expected 'PASS' in output, got: {stdout!r}\nharness:\n{test_harness}"
        )
