"""
Coding Agent — iterative code generation with semantic code search and sandboxed execution.

Searches code semantically via the code_lateon_mv Vespa profile (LateOn-Code-edge
multi-vector embeddings with tree-sitter AST chunking), plans implementation via DSPy,
generates code, executes in an OpenShell sandbox, evaluates output, and iterates.
Advertised client tools select workspace turns that suspend for tool results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import uuid
from typing import Any, Dict, List, Optional

import dspy
from pydantic import Field

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.agents.rlm_options import RLMOptions

logger = logging.getLogger(__name__)

WORKSPACE_MAX_ROUNDS = 8
WORKSPACE_ACTION_MAX_ATTEMPTS = 3


class CodingInput(AgentInput):
    task: str = Field(..., description="Coding task description")
    codebase_path: str = Field("", description="Path to codebase for context search")
    tenant_id: str = Field(..., description="Tenant identifier (required)")
    max_iterations: int = Field(5, description="Maximum plan-code-execute iterations")
    language: str = Field("python", description="Primary programming language")
    rlm: Optional[RLMOptions] = Field(
        None,
        description="RLM configuration. None=disabled, set RLMOptions to enable RLM inference for large codebases",
    )

    external_tools: List[Dict[str, Any]] = Field(
        default_factory=list, description="Client tools selecting workspace execution"
    )
    tool_results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Results for the latest assistant tool calls"
    )
    assistant_tool_calls: List[Dict[str, Any]] = Field(
        default_factory=list, description="Latest tool calls being resumed"
    )
    continuation_state: Dict[str, Any] = Field(
        default_factory=dict, description="Workspace plan and consumed round count"
    )
    tool_exchange: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Tool calls and results in chronological rounds",
    )


class CodeChange(AgentOutput):
    file_path: str = Field(..., description="Path of created/modified file")
    content: str = Field(..., description="File content")
    change_type: str = Field("create", description="create, modify, or delete")


class ExecutionResult(AgentOutput):
    command: str = Field(..., description="Command that was executed")
    stdout: str = Field("", description="Standard output")
    stderr: str = Field("", description="Standard error")
    exit_code: int = Field(0, description="Process exit code")
    success: bool = Field(True, description="Whether execution succeeded")


class CodingOutput(AgentOutput):
    plan: str = Field("", description="Implementation plan")
    code_changes: List[Dict[str, str]] = Field(
        default_factory=list, description="List of file changes"
    )
    execution_results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Sandbox execution results"
    )
    summary: str = Field("", description="Summary of what was accomplished")
    iterations_used: int = Field(0, description="Number of iterations used")
    files_modified: List[str] = Field(
        default_factory=list, description="List of modified file paths"
    )
    rlm_synthesis: Optional[str] = Field(
        None,
        description="RLM-synthesized answer from codebase context (only when RLM enabled)",
    )
    rlm_telemetry: Optional[Dict[str, Any]] = Field(
        None, description="RLM telemetry metrics for A/B testing"
    )

    pending_tool_calls: List[Dict[str, Any]] = Field(
        default_factory=list, description="Client workspace calls required to resume"
    )
    continuation_state: Dict[str, Any] = Field(
        default_factory=dict, description="Workspace plan and consumed round count"
    )
    success: bool = Field(True, description="Whether the coding step succeeded")
    error: Optional[str] = Field(None, description="Failure that prevented completion")


class CodingDeps(AgentDeps):
    # tenant_id is enforced at CodingAgent.__init__ via require_tenant_id;
    # the None default is only here to satisfy pydantic's field ordering.
    tenant_id: Optional[str] = None
    sandbox_manager: Optional[Any] = None


class TaskPlanningSignature(dspy.Signature):
    """Plan an implementation given a coding task and relevant code context."""

    task: str = dspy.InputField(desc="Coding task description")
    code_context: str = dspy.InputField(
        desc="Relevant code snippets from semantic search"
    )
    language: str = dspy.InputField(desc="Primary programming language")
    plan: str = dspy.OutputField(
        desc="Step-by-step implementation plan with file paths and changes"
    )


class CodeGenerationSignature(dspy.Signature):
    """Generate code implementing the plan."""

    task: str = dspy.InputField(desc="Coding task description")
    plan: str = dspy.InputField(desc="Implementation plan")
    code_context: str = dspy.InputField(desc="Relevant existing code for reference")
    language: str = dspy.InputField(desc="Programming language")
    previous_error: str = dspy.InputField(
        desc="Error from previous attempt, empty if first attempt"
    )
    code: str = dspy.OutputField(desc="Complete implementation code as a single file")
    test_command: str = dspy.OutputField(desc="Command to test/run the generated code")


class WorkspaceStepSignature(dspy.Signature):
    """Choose a client workspace action, or finish when the task is complete."""

    task: str = dspy.InputField(desc="Coding task description")
    plan: str = dspy.InputField(desc="Implementation plan")
    observations: str = dspy.InputField(
        desc="Prior workspace actions and their results"
    )
    available_tools: str = dspy.InputField(
        desc="Available tool names and argument schemas"
    )
    remaining_steps: int = dspy.InputField(desc="Remaining workspace decision budget")
    tool_name: str = dspy.OutputField(desc="One advertised tool name, or finish")
    tool_args_json: str = dspy.OutputField(desc="JSON object of tool arguments")
    summary: str = dspy.OutputField(
        desc="Completed work when finishing; otherwise rationale"
    )


def parse_workspace_action(
    tool_name: str, tool_args_json: str, tools: Dict[str, Dict[str, Any]]
) -> Optional[tuple[str, Dict[str, Any]]]:
    """Return a validated action, None for finish, or raise for a failed decision."""
    name = tool_name.strip()
    if name != "finish" and name not in tools:
        raise ValueError(f"unknown tool {name!r}")
    try:
        arguments = json.loads(tool_args_json)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"arguments for {name!r} must be valid JSON") from exc
    if not isinstance(arguments, dict):
        raise ValueError(f"arguments for {name!r} must be a JSON object")
    if name == "finish":
        return None
    for required in tools[name].get("parameters", {}).get("required", []):
        if required not in arguments:
            raise ValueError(f"missing required argument {required!r} for {name!r}")
    return name, arguments


def _index_workspace_round(
    round_data: Dict[str, Any],
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Validate a complete round and index calls and results by call id."""
    calls = round_data.get("tool_calls", [])
    results = round_data.get("results", [])
    calls_by_id = {}
    for call in calls:
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("Tool call requires a non-empty id")
        if call_id in calls_by_id:
            raise ValueError(f"Duplicate tool call id {call_id!r}")
        calls_by_id[call_id] = call
    results_by_id = {}
    for result in results:
        call_id = result.get("tool_call_id")
        if call_id not in calls_by_id:
            raise ValueError(f"Unmatched tool result id {call_id!r}")
        if call_id in results_by_id:
            raise ValueError(f"Duplicate tool result id {call_id!r}")
        results_by_id[call_id] = result
    missing = calls_by_id.keys() - results_by_id.keys()
    if missing:
        raise ValueError(f"Missing tool results for: {', '.join(sorted(missing))}")
    return calls_by_id, results_by_id


def format_observations(tool_exchange: List[Dict[str, Any]]) -> str:
    """Render each result beside its matching tool call, in call order."""
    lines = []
    for index, round_data in enumerate(tool_exchange, start=1):
        calls_by_id, results_by_id = _index_workspace_round(round_data)
        for call_id, call in calls_by_id.items():
            function = call["function"]
            result = results_by_id[call_id]
            lines.append(
                f"step {index}: {function['name']}({function['arguments']}) -> "
                f"{result.get('content', '')}"
            )
    return "\n".join(lines) if lines else "no workspace actions taken yet"


class OutputEvaluationSignature(dspy.Signature):
    """Evaluate whether code execution output meets the task requirements."""

    task: str = dspy.InputField(desc="Original task description")
    code: str = dspy.InputField(desc="Generated code")
    stdout: str = dspy.InputField(desc="Execution stdout")
    stderr: str = dspy.InputField(desc="Execution stderr")
    exit_code: int = dspy.InputField(desc="Execution exit code")
    is_successful: bool = dspy.OutputField(
        desc="True if execution output satisfies the task"
    )
    feedback: str = dspy.OutputField(
        desc="Specific feedback for improving the code if not successful"
    )


class CodingAgent(
    MemoryAwareMixin,
    RLMAwareMixin,
    A2AAgent[CodingInput, CodingOutput, CodingDeps],
):
    """Plan coding work and execute it through client tools or a sandbox.

    Tenant instructions and memories enrich plans through async context injection.
    Workspace turns suspend for client results; sandbox turns generate and evaluate code.
    """

    def __init__(
        self,
        deps: CodingDeps,
        config: A2AAgentConfig | None = None,
        search_fn: Any = None,
        sandbox_manager: Any = None,
        config_manager=None,
    ):
        from cogniverse_core.common.tenant_utils import require_tenant_id

        # CodingDeps.tenant_id defaults to None so pydantic doesn't force
        # an order constraint; enforce the real contract here.
        require_tenant_id(deps.tenant_id, source="CodingAgent(deps.tenant_id)")

        if config is None:
            config = A2AAgentConfig(
                agent_name="coding_agent",
                agent_description="Iterative coding agent with code search and sandboxed execution",
                capabilities=["coding", "code_generation", "code_search"],
            )
        super().__init__(deps=deps, config=config)

        self._search_fn = search_fn
        self._sandbox_manager = sandbox_manager or deps.sandbox_manager
        # Enables the RLM path (RLMAwareMixin) to route its LM through the
        # gateway for this tenant.
        self._config_manager = config_manager
        self._planner = dspy.ChainOfThought(TaskPlanningSignature)
        self._generator = dspy.ChainOfThought(CodeGenerationSignature)
        self._evaluator = dspy.ChainOfThought(OutputEvaluationSignature)
        self._workspace_step = dspy.ChainOfThought(WorkspaceStepSignature)

    async def _process_impl(self, input: CodingInput) -> CodingOutput:
        if input.external_tools:
            return await self._process_workspace(input)

        # Set tenant for memory/instructions injection and enrich the task
        # with the full context stack (instructions + learned strategies +
        # tenant memories) before planning. No-ops when memory isn't
        # initialized.
        self.set_tenant_for_context(input.tenant_id)
        enriched_task = await self.inject_context_into_prompt_async(
            input.task, input.task
        )

        # 1. Search for relevant code context
        self.emit_progress("search", "Searching for relevant code context...")
        code_context = await self._search_code_context(input.task, input.tenant_id)

        # 2. Plan implementation
        self.emit_progress("plan", "Planning implementation...")
        plan = await self._plan(enriched_task, code_context, input.language)

        all_code_changes: List[Dict[str, str]] = []
        all_exec_results: List[Dict[str, Any]] = []
        files_modified: List[str] = []
        previous_error = ""
        iteration = 0
        workspace_dir = tempfile.mkdtemp(prefix=f"coding_{uuid.uuid4().hex[:8]}_")
        try:
            for iteration in range(1, input.max_iterations + 1):
                self.emit_progress(
                    "generate",
                    f"Iteration {iteration}: generating code...",
                )

                code, test_command = await self._generate_code(
                    input.task, plan, code_context, input.language, previous_error
                )

                file_path = f"{workspace_dir}/solution.{self._ext(input.language)}"
                code_changes = [
                    {"file_path": file_path, "content": code, "change_type": "create"}
                ]
                all_code_changes = code_changes
                files_modified = [file_path]

                self.emit_progress("execute", f"Iteration {iteration}: executing...")
                exec_result = await self._execute_in_sandbox(
                    file_path, code, test_command, input.language
                )
                all_exec_results.append(exec_result)

                self.emit_progress("evaluate", f"Iteration {iteration}: evaluating...")
                is_successful, feedback = await self._evaluate_output(
                    input.task,
                    code,
                    exec_result.get("stdout", ""),
                    exec_result.get("stderr", ""),
                    exec_result.get("exit_code", -1),
                )

                if is_successful:
                    self.emit_progress(
                        "done", f"Task completed in {iteration} iterations"
                    )
                    break

                previous_error = (
                    f"Exit code: {exec_result.get('exit_code')}\n"
                    f"stderr: {exec_result.get('stderr', '')}\n"
                    f"Feedback: {feedback}"
                )

        finally:
            shutil.rmtree(workspace_dir)

        # 4. Synthesize summary
        self.emit_progress("summarize", "Generating summary...")
        summary = (
            f"Completed coding task in {iteration} iteration(s). "
            f"Generated {len(files_modified)} file(s). "
            f"Final execution: exit_code={all_exec_results[-1].get('exit_code', -1)}"
            if all_exec_results
            else f"Planned but no code executed after {iteration} iterations."
        )

        rlm_synthesis = None
        rlm_telemetry = None

        if self.should_use_rlm_for_query(input.rlm, code_context):
            self.emit_progress(
                "rlm_synthesis", "Synthesizing codebase context with RLM..."
            )
            logger.info(f"RLM enabled for coding task: {input.task[:50]}...")
            try:
                rlm_result = self.process_with_rlm(
                    query=input.task,
                    context=code_context,
                    rlm_options=input.rlm,
                )
                rlm_synthesis = rlm_result.answer
                rlm_telemetry = self.get_rlm_telemetry(rlm_result, len(code_context))
                logger.info(
                    f"RLM synthesis complete: depth={rlm_result.depth_reached}, "
                    f"calls={rlm_result.total_calls}, latency={rlm_result.latency_ms:.0f}ms"
                )
            except Exception as e:
                logger.error(f"RLM processing failed: {e}")
                rlm_telemetry = {
                    "rlm_enabled": False,
                    "rlm_attempted": True,
                    "rlm_error": str(e),
                }

        return CodingOutput(
            plan=plan,
            code_changes=all_code_changes,
            execution_results=all_exec_results,
            summary=summary,
            iterations_used=iteration,
            files_modified=files_modified,
            rlm_synthesis=rlm_synthesis,
            rlm_telemetry=rlm_telemetry,
        )

    async def _process_workspace(self, input: CodingInput) -> CodingOutput:
        """Suspend for client tools, with bounded retries for invalid decisions."""
        self.set_tenant_for_context(input.tenant_id)
        enriched_task = await self.inject_context_into_prompt_async(
            input.task, input.task
        )
        exchanges = list(input.tool_exchange)
        if input.assistant_tool_calls or input.tool_results:
            latest = {
                "tool_calls": input.assistant_tool_calls,
                "results": input.tool_results,
            }
            latest_calls, latest_results = _index_workspace_round(latest)
            if exchanges:
                recorded_calls, recorded_results = _index_workspace_round(exchanges[-1])
                if latest_calls.keys() == recorded_calls.keys():
                    if (
                        latest_calls != recorded_calls
                        or latest_results != recorded_results
                    ):
                        raise ValueError(
                            "Latest tool replay conflicts with recorded round"
                        )
                else:
                    exchanges.append(latest)
            else:
                exchanges.append(latest)
        observations = format_observations(exchanges)
        state = input.continuation_state
        step = state.get("step", len(exchanges))
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("Workspace step must be a non-negative integer")
        step = max(step, len(exchanges))
        plan = str(state.get("plan", ""))
        budget = WORKSPACE_MAX_ROUNDS
        if "max_iterations" in input.model_fields_set:
            budget = min(input.max_iterations, WORKSPACE_MAX_ROUNDS)
        if step >= budget:
            return CodingOutput(
                plan=plan,
                iterations_used=step,
                success=False,
                error=f"Workspace round limit reached ({budget})",
            )
        if not plan:
            self.emit_progress("plan", "Planning workspace actions...")
            code_context = await self._search_code_context(input.task, input.tenant_id)
            plan = await self._plan(enriched_task, code_context, input.language)
        tools = {
            tool["function"]["name"]: tool["function"] for tool in input.external_tools
        }
        descriptions = json.dumps(list(tools.values()))
        attempts = min(WORKSPACE_ACTION_MAX_ATTEMPTS, budget - step)
        for attempt in range(1, attempts + 1):
            self.emit_progress(
                "step", f"Choosing workspace action {step + 1}/{budget}..."
            )
            decision = await self.call_dspy(
                self._workspace_step,
                output_field="tool_name",
                task=enriched_task,
                plan=plan,
                observations=observations,
                available_tools=f"{descriptions}; finish: task is complete",
                remaining_steps=budget - step,
            )
            step += 1
            try:
                action = parse_workspace_action(
                    str(getattr(decision, "tool_name", "")),
                    getattr(decision, "tool_args_json", ""),
                    tools,
                )
            except ValueError as exc:
                self.emit_progress("step_failed", str(exc), data={"step": step})
                if attempt == attempts:
                    return CodingOutput(
                        plan=plan,
                        iterations_used=step,
                        success=False,
                        error=f"Invalid workspace action after {attempt} attempt(s): {exc}",
                    )
                observations += f"\nstep {step} failed: {exc}"
                continue
            if action is None:
                summary = str(getattr(decision, "summary", "") or "").strip()
                return CodingOutput(plan=plan, summary=summary, iterations_used=step)
            name, arguments = action
            return CodingOutput(
                plan=plan,
                pending_tool_calls=[
                    {
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "name": name,
                        "arguments": arguments,
                    }
                ],
                continuation_state={"mode": "workspace", "plan": plan, "step": step},
                iterations_used=step,
            )
        raise ValueError("Workspace decision budget must be positive")

    async def _search_code_context(self, task: str, tenant_id: str) -> str:
        """Search for relevant code using the code_lateon_mv profile."""
        if not self._search_fn:
            logger.info("No search_fn provided, proceeding without code context")
            return ""

        results = await self._search_fn(query=task, tenant_id=tenant_id)
        if not results:
            return ""

        context_parts = []
        for r in results[:5]:
            if isinstance(r, dict):
                metadata = r.get("metadata", {})
                file_path = metadata.get("file", r.get("document_id", "unknown"))
                chunk_name = metadata.get("chunk_name", "")
                text = metadata.get("extracted_text", r.get("description", ""))
                header = f"# {file_path}"
                if chunk_name:
                    header += f" :: {chunk_name}"
                context_parts.append(f"{header}\n{text}")

        return "\n\n---\n\n".join(context_parts)

    async def _plan(self, task: str, code_context: str, language: str) -> str:
        """Create implementation plan via DSPy."""
        result = await self.call_dspy(
            self._planner,
            output_field="plan",
            task=task,
            code_context=code_context or "No existing code context available.",
            language=language,
        )
        return str(result.plan)

    async def _generate_code(
        self,
        task: str,
        plan: str,
        code_context: str,
        language: str,
        previous_error: str,
    ) -> tuple[str, str]:
        """Generate code via DSPy."""
        result = await self.call_dspy(
            self._generator,
            output_field="code",
            task=task,
            plan=plan,
            code_context=code_context or "No existing code context available.",
            language=language,
            previous_error=previous_error or "First attempt.",
        )
        code = self._strip_markdown_fences(str(result.code))
        test_command = str(getattr(result, "test_command", ""))
        return code, test_command

    async def _execute_in_sandbox(
        self,
        file_path: str,
        code: str,
        test_command: str,
        language: str,
    ) -> Dict[str, Any]:
        """Execute code in an OpenShell sandbox. Refuses to run without one.

        The connectivity probe and both sandbox execs are synchronous gRPC/socket
        calls (write exec 30s, run exec up to 300s); offload them so a coding
        task cannot freeze the shared API loop and trip k8s liveness mid-run.
        """
        sandbox_available = (
            self._sandbox_manager is not None
            and await asyncio.to_thread(lambda: self._sandbox_manager.available)
        )
        if not sandbox_available:
            raise RuntimeError(
                "CodingAgent requires a SandboxManager with an available OpenShell "
                "gateway. Executing LLM-generated code without sandbox isolation "
                "is not permitted. Provide a sandbox_manager to CodingAgent or "
                "CodingDeps, or start the OpenShell gateway."
            )

        # Always run the file we actually wrote (``file_path`` = solution.<ext>).
        # The LLM's ``test_command`` frequently names a file from its plan
        # (e.g. ``python hello_world.py``) that doesn't match the fixed
        # solution file we write, so trusting it executes a nonexistent path.
        run_cmd = self._default_run_command(file_path, language)

        # Write code inside the sandbox via exec, not on the host filesystem
        write_cmd = (
            f"mkdir -p $(dirname {file_path}) && "
            f"cat > {file_path} << 'SANDBOX_CODE_EOF'\n{code}\nSANDBOX_CODE_EOF"
        )
        await asyncio.to_thread(
            self._sandbox_manager.exec_in_sandbox,
            agent_type="coding_agent",
            command=["sh", "-c", write_cmd],
            timeout_seconds=30,
        )

        result = await asyncio.to_thread(
            self._sandbox_manager.exec_in_sandbox,
            agent_type="coding_agent",
            command=["sh", "-c", run_cmd],
            timeout_seconds=300,
        )
        if result is None:
            raise RuntimeError(
                "Sandbox exec returned None — sandbox session may have failed. "
                "Check OpenShell gateway logs."
            )
        result["command"] = run_cmd
        result["success"] = result.get("exit_code", -1) == 0
        return result

    async def _evaluate_output(
        self,
        task: str,
        code: str,
        stdout: str,
        stderr: str,
        exit_code: int,
    ) -> tuple[bool, str]:
        """Evaluate execution output via DSPy."""
        result = await self.call_dspy(
            self._evaluator,
            output_field="feedback",
            task=task,
            code=code,
            stdout=stdout[:2000],
            stderr=stderr[:2000],
            exit_code=exit_code,
        )
        is_successful = bool(result.is_successful)
        feedback = str(result.feedback)
        return is_successful, feedback

    @staticmethod
    def _strip_markdown_fences(code: str) -> str:
        """Strip markdown code fences (```python ... ```) from LLM output."""
        import re

        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", code.strip())
        stripped = re.sub(r"\n?```$", "", stripped.strip())
        return stripped.strip()

    @staticmethod
    def _ext(language: str) -> str:
        """Return file extension for language."""
        return {
            "python": "py",
            "javascript": "js",
            "typescript": "ts",
            "go": "go",
        }.get(language, "py")

    @staticmethod
    def _default_run_command(file_path: str, language: str) -> str:
        """Return default run command for language."""
        return {
            "python": f"python {file_path}",
            "javascript": f"node {file_path}",
            "typescript": f"npx ts-node {file_path}",
            "go": f"go run {file_path}",
        }.get(language, f"python {file_path}")
