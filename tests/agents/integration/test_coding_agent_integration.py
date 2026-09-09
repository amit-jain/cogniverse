"""
Integration test for CodingAgent.

Tests the full coding agent pipeline: DSPy planning + code generation +
sandboxed execution + output evaluation. Uses the configured LM and real
OpenShell sandbox (started/destroyed per test module).

Requires: the configured LM endpoint, openshell CLI, Docker.
"""

import json
import logging
from pathlib import Path

import pytest

from cogniverse_runtime.sandbox_manager import SandboxManager, SandboxPolicy
from tests.agents.integration.conftest import skip_if_no_lm


def _memory_config_manager():
    """The injected ConfigManager every agent constructor requires."""
    from cogniverse_foundation.config.manager import ConfigManager
    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    store.initialize()
    return ConfigManager(store=store)


pytestmark = pytest.mark.integration

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
        agent = CodingAgent(deps=deps, config_manager=_memory_config_manager())

        assert agent.agent_name == "coding_agent"
        assert "coding" in agent.capabilities

    def test_coding_input_validation(self):
        """CodingInput validates required fields."""
        from cogniverse_agents.coding_agent import CodingInput

        inp = CodingInput(task="Write a hello world function", tenant_id="test:unit")
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
        assert (
            profile["strategies"]["segmentation"]["class"] == "CodeSegmentationStrategy"
        )
        assert (
            profile["strategies"]["embedding"]["class"] == "CodeTextEmbeddingStrategy"
        )
        assert profile["schema_config"]["embedding_dim"] == 48
        assert profile["schema_config"]["binary_dim"] == 6
        assert profile["schema_config"]["num_patches"] == 2048


CODING_GW_NAME = "cogniverse-coding-test-gw"
CODING_GW_PORT = 19091
INGESTED_SEGMENTS = 15
SEARCH_HITS = 5


@pytest.fixture(scope="module")
def coding_test_gateway(tmp_path_factory):
    """Start this module's OpenShell gateway in a private config root.

    ``XDG_CONFIG_HOME`` is pointed at that root for the module so the SDK
    resolves ``CODING_GW_NAME`` there; the host's own registrations and
    active-gateway pointer stay byte-identical across the module.
    """
    from tests.agents.integration.conftest import OpenShellTestGateway

    host_active = Path.home() / ".config" / "openshell" / "active_gateway"
    host_active_before = host_active.read_bytes() if host_active.exists() else None
    config_home = tmp_path_factory.mktemp("openshell-config")
    gateway = OpenShellTestGateway(CODING_GW_NAME, CODING_GW_PORT, config_home)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("XDG_CONFIG_HOME", str(config_home))
        mp.delenv("OPENSHELL_GATEWAY", raising=False)
        mp.delenv("OPENSHELL_GATEWAY_ENDPOINT", raising=False)
        gateway.start()
        try:
            assert json.loads(gateway.metadata_path.read_text()) == {
                "name": CODING_GW_NAME,
                "gateway_endpoint": f"https://127.0.0.1:{CODING_GW_PORT}",
                "is_remote": False,
                "gateway_port": CODING_GW_PORT,
            }
            host_active_after = (
                host_active.read_bytes() if host_active.exists() else None
            )
            assert host_active_after == host_active_before
            yield gateway
        finally:
            gateway.destroy()
    host_active_final = host_active.read_bytes() if host_active.exists() else None
    assert host_active_final == host_active_before


@pytest.fixture(scope="module")
def code_search_infra(served_code_colbert, coding_test_gateway, vespa_with_schema):
    """Deploy code_lateon_mv into the test Vespa, ingest real code, connect the
    sandbox manager to this module's gateway.

    Uses vespa_with_schema for the Vespa container, deploys the native 48-dim
    code schema alongside the existing video schema, feeds real code segments
    with LateOn-Code-edge embeddings.
    """
    import time

    import numpy as np
    import requests

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
    from cogniverse_runtime.ingestion.strategies import CodeSegmentationStrategy

    colbert_model = served_code_colbert
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

    segments_to_ingest = all_segments[:INGESTED_SEGMENTS]
    assert len(segments_to_ingest) == INGESTED_SEGMENTS
    texts = [seg["content"][:8192] for seg in segments_to_ingest]
    doc_embeddings = colbert_model.encode(texts, is_query=False)

    schema_name = tenant_schema
    for idx, (seg, emb) in enumerate(zip(segments_to_ingest, doc_embeddings)):
        emb_np = np.array(emb, dtype=np.float32)
        if emb_np.shape[0] > 2048:
            emb_np = emb_np[:2048]

        float_dict = {str(i): emb_np[i].tolist() for i in range(emb_np.shape[0])}
        binary = np.packbits(
            np.where(emb_np > 0, 1, 0).astype(np.uint8), axis=1
        ).astype(np.int8)
        binary_dict = {str(i): binary[i].tolist() for i in range(binary.shape[0])}

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
            json=doc,
            timeout=10,
        )
        assert resp.status_code == 200, (
            f"Feed failed for seg_{idx}: {resp.status_code} - {resp.text}"
        )

    logger.info(f"Ingested {len(segments_to_ingest)} code segments into Vespa")
    time.sleep(3)

    sandbox = SandboxManager(
        policy_dir="configs/agent_policies",
        cluster=CODING_GW_NAME,
        policy=SandboxPolicy.REQUIRED,
    )
    canary = sandbox.exec_in_sandbox(
        agent_type="coding_agent",
        command=["echo", "gateway-ready"],
        timeout_seconds=60,
    )
    assert canary["exit_code"] == 0, f"sandbox canary failed: {canary}"
    assert canary["stderr"] == ""
    assert canary["stdout"].strip() == "gateway-ready"

    yield {
        "sandbox": sandbox,
        "vespa_url": base_url,
        "schema_name": schema_name,
        "colbert_model": colbert_model,
        "ingested_ids": [f"seg_{idx}" for idx in range(len(segments_to_ingest))],
    }

    sandbox.close()


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
                qt_cells.append(
                    {
                        "address": {"querytoken": str(tok_idx), "v": str(v_idx)},
                        "value": float(query_np[tok_idx, v_idx]),
                    }
                )

        resp = _requests.post(
            f"{vespa_url}/search/",
            json={
                "yql": f"select * from {schema_name} where true",
                "hits": SEARCH_HITS,
                "ranking.profile": "float_float",
                "model.restrict": schema_name,
                "input.query(qt)": {"cells": qt_cells},
            },
            timeout=10,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Vespa code search failed: {resp.status_code} - {resp.text}"
            )

        hits = resp.json().get("root", {}).get("children", [])
        return [
            {
                "document_id": h["fields"]["code_id"],
                "score": h["relevance"],
                "metadata": {
                    "file": h["fields"]["file_path"],
                    "chunk_name": h["fields"]["chunk_name"],
                    "extracted_text": h["fields"]["source_code"],
                },
            }
            for h in hits
        ]

    return search_fn


def _assert_single_solution(result, max_iterations: int) -> dict:
    """Pin the shape CodingAgent persists for a task that ends in a passing run."""
    assert result.iterations_used in range(1, max_iterations + 1)
    assert len(result.execution_results) == result.iterations_used
    final = result.execution_results[-1]
    assert set(final) == {"stdout", "stderr", "exit_code", "command", "success"}
    assert final["exit_code"] == 0, f"final run failed: {final}"
    assert final["success"] is True
    assert result.summary == (
        f"Completed coding task in {result.iterations_used} iteration(s). "
        "Generated 1 file(s). Final execution: exit_code=0"
    )
    (change,) = result.code_changes
    assert set(change) == {"file_path", "content", "change_type"}
    assert change["change_type"] == "create"
    assert change["file_path"].endswith("/solution.py")
    assert final["command"] == f"python {change['file_path']}"
    assert result.files_modified == [change["file_path"]]
    return change


@skip_if_no_lm
class TestCodingAgentWithRealLM:
    """Full integration: configured LM + real Vespa code search + real OpenShell sandbox.

    code_search_infra deploys code_lateon_mv into the test Vespa, ingests real
    source with LateOn-Code-edge, connects to this module's OpenShell gateway.
    The agent's search_fn hits real Vespa with real embeddings. Generated code
    runs in the sandbox.
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
        config = get_config(tenant_id="test:unit", config_manager=cm)
        endpoint = config.get_llm_config().resolve("coding_agent")
        lm = create_dspy_lm(endpoint)
        dspy.configure(lm=lm)
        return lm

    @pytest.mark.asyncio
    async def test_coding_agent_generates_working_code(
        self, dspy_configured, code_search_infra
    ):
        """LLM generates add(2,3) with real code search context, the agent's
        sandbox run and an independent re-run both print exactly 5."""
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
        agent = CodingAgent(
            deps=deps,
            search_fn=search_fn,
            sandbox_manager=sandbox,
            config_manager=_memory_config_manager(),
        )

        input_data = CodingInput(
            task=(
                "Write a Python function called 'add' that takes two numbers "
                "and returns their sum. Print only the result of add(2, 3), "
                "nothing else."
            ),
            language="python",
            max_iterations=3,
            tenant_id="test:unit",
        )

        result = await agent.process(input_data)

        change = _assert_single_solution(result, input_data.max_iterations)
        assert result.execution_results[-1]["stdout"].strip() == "5"

        generated_code = change["content"]
        verify = sandbox.exec_in_sandbox(
            agent_type="coding_agent",
            command=["python3", "-c", generated_code],
            timeout_seconds=30,
        )
        logger.info(f"Sandbox run: {verify}")
        logger.info(f"Generated code:\n{generated_code}")

        assert verify["exit_code"] == 0, (
            f"Generated code failed in sandbox with exit {verify['exit_code']}.\n"
            f"stderr: {verify['stderr']}\ncode:\n{generated_code}"
        )
        assert verify["stderr"] == ""
        assert verify["stdout"].strip() == "5", (
            f"Expected stdout '5' from add(2,3), got: {verify['stdout']!r}\n"
            f"code:\n{generated_code}"
        )

    @pytest.mark.asyncio
    async def test_coding_agent_with_real_code_search(
        self, dspy_configured, code_search_infra
    ):
        """Agent searches the real Vespa code index exactly once with the task
        text, generates email validation, and the validator accepts
        test@example.com inside the OpenShell sandbox."""
        from cogniverse_agents.coding_agent import (
            CodingAgent,
            CodingDeps,
            CodingInput,
        )

        infra = code_search_infra
        sandbox = infra["sandbox"]

        search_calls: list[tuple[str, str]] = []
        search_results: list[list[dict]] = []
        raw_search_fn = _build_vespa_search_fn(
            infra["vespa_url"], infra["schema_name"], infra["colbert_model"]
        )

        async def tracked_search_fn(query: str, tenant_id: str):
            search_calls.append((query, tenant_id))
            hits = await raw_search_fn(query, tenant_id)
            search_results.append(hits)
            return hits

        deps = CodingDeps(tenant_id="test", sandbox_manager=sandbox)
        agent = CodingAgent(
            deps=deps,
            search_fn=tracked_search_fn,
            sandbox_manager=sandbox,
            config_manager=_memory_config_manager(),
        )

        input_data = CodingInput(
            task=(
                "Write a function that validates email addresses using regex "
                "and returns True or False. Print whether 'test@example.com' "
                "is valid."
            ),
            language="python",
            max_iterations=3,
            tenant_id="test:unit",
        )

        result = await agent.process(input_data)

        assert search_calls == [(input_data.task, "test:unit")]
        (hits,) = search_results
        assert len(hits) == SEARCH_HITS
        assert set(h["document_id"] for h in hits) <= set(infra["ingested_ids"])
        assert len({h["document_id"] for h in hits}) == SEARCH_HITS
        scores = [h["score"] for h in hits]
        assert scores == sorted(scores, reverse=True)

        change = _assert_single_solution(result, input_data.max_iterations)
        generated_code = change["content"]

        import re as _re

        fn_match = _re.search(r"def\s+(\w+)\s*\(", generated_code)
        assert fn_match, f"no function definition in generated code:\n{generated_code}"
        fn_name = fn_match.group(1)

        test_harness = (
            generated_code + "\n\n"
            f"result = {fn_name}('test@example.com')\n"
            "print('PASS:', bool(result))\n"
        )
        verify = sandbox.exec_in_sandbox(
            agent_type="coding_agent",
            command=["python3", "-c", test_harness],
            timeout_seconds=30,
        )
        logger.info(f"Sandbox run: {verify}")
        logger.info(f"Generated code:\n{generated_code}")

        assert verify["exit_code"] == 0, (
            f"Code + harness failed in sandbox: exit {verify['exit_code']}\n"
            f"stderr: {verify['stderr']}\nharness:\n{test_harness}"
        )
        assert verify["stderr"] == ""
        assert verify["stdout"].strip().splitlines()[-1] == "PASS: True", (
            f"Expected final line 'PASS: True', got: {verify['stdout']!r}\n"
            f"harness:\n{test_harness}"
        )
