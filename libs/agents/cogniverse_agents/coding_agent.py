"""
Coding Agent — iterative code generation with semantic code search and sandboxed execution.

Searches code semantically via the code_lateon_mv Vespa profile (LateOn-Code-edge
multi-vector embeddings with tree-sitter AST chunking), plans implementation via DSPy,
generates code, executes in an OpenShell sandbox, evaluates output, and iterates.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import dspy
from pydantic import Field

from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput

logger = logging.getLogger(__name__)


class CodingInput(AgentInput):
    task: str = Field(..., description="Coding task description")
    codebase_path: str = Field("", description="Path to codebase for context search")
    tenant_id: str = Field("default", description="Tenant identifier")
    max_iterations: int = Field(5, description="Maximum plan-code-execute iterations")
    language: str = Field("python", description="Primary programming language")


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


class CodingDeps(AgentDeps):
    tenant_id: str = "default"
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
    code_context: str = dspy.InputField(
        desc="Relevant existing code for reference"
    )
    language: str = dspy.InputField(desc="Programming language")
    previous_error: str = dspy.InputField(
        desc="Error from previous attempt, empty if first attempt"
    )
    code: str = dspy.OutputField(
        desc="Complete implementation code as a single file"
    )
    test_command: str = dspy.OutputField(
        desc="Command to test/run the generated code"
    )


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


class CodingAgent(A2AAgent[CodingInput, CodingOutput, CodingDeps]):
    """
    Iterative coding agent with semantic code search and sandboxed execution.

    Flow:
    1. Search code context via SearchService (code_lateon_mv profile)
    2. Plan implementation via DSPy
    3. Generate code via DSPy
    4. Write code to sandbox workspace
    5. Execute in sandbox (or locally if sandbox unavailable)
    6. Evaluate output via DSPy
    7. Iterate if evaluation fails
    """

    def __init__(
        self,
        deps: CodingDeps,
        config: A2AAgentConfig | None = None,
        search_fn: Any = None,
        sandbox_manager: Any = None,
    ):
        if config is None:
            config = A2AAgentConfig(
                agent_name="coding_agent",
                agent_description="Iterative coding agent with code search and sandboxed execution",
                capabilities=["coding", "code_generation", "code_search"],
            )
        super().__init__(deps=deps, config=config)

        self._search_fn = search_fn
        self._sandbox_manager = sandbox_manager or deps.sandbox_manager
        self._planner = dspy.ChainOfThought(TaskPlanningSignature)
        self._generator = dspy.ChainOfThought(CodeGenerationSignature)
        self._evaluator = dspy.ChainOfThought(OutputEvaluationSignature)

    async def _process_impl(self, input: CodingInput) -> CodingOutput:
        # 1. Search for relevant code context
        self.emit_progress("search", "Searching for relevant code context...")
        code_context = await self._search_code_context(input.task, input.tenant_id)

        # 2. Plan implementation
        self.emit_progress("plan", "Planning implementation...")
        plan = await self._plan(input.task, code_context, input.language)

        # 3. Iterative code-execute-evaluate loop
        all_code_changes: List[Dict[str, str]] = []
        all_exec_results: List[Dict[str, Any]] = []
        files_modified: List[str] = []
        previous_error = ""
        iteration = 0
        for iteration in range(1, input.max_iterations + 1):
            self.emit_progress(
                "generate",
                f"Iteration {iteration}: generating code...",
            )

            # Generate code
            code, test_command = await self._generate_code(
                input.task, plan, code_context, input.language, previous_error
            )

            # Write to workspace and execute
            file_path = f"/tmp/coding_workspace/solution.{self._ext(input.language)}"
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

            # Evaluate
            self.emit_progress("evaluate", f"Iteration {iteration}: evaluating...")
            is_successful, feedback = await self._evaluate_output(
                input.task,
                code,
                exec_result.get("stdout", ""),
                exec_result.get("stderr", ""),
                exec_result.get("exit_code", -1),
            )

            if is_successful:
                self.emit_progress("done", f"Task completed in {iteration} iterations")
                break

            previous_error = (
                f"Exit code: {exec_result.get('exit_code')}\n"
                f"stderr: {exec_result.get('stderr', '')}\n"
                f"Feedback: {feedback}"
            )

        # 4. Synthesize summary
        self.emit_progress("summarize", "Generating summary...")
        summary = (
            f"Completed coding task in {iteration} iteration(s). "
            f"Generated {len(files_modified)} file(s). "
            f"Final execution: exit_code={all_exec_results[-1].get('exit_code', -1)}"
            if all_exec_results
            else f"Planned but no code executed after {iteration} iterations."
        )

        return CodingOutput(
            plan=plan,
            code_changes=all_code_changes,
            execution_results=all_exec_results,
            summary=summary,
            iterations_used=iteration,
            files_modified=files_modified,
        )

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
        """Execute code in an OpenShell sandbox. Refuses to run without one."""
        if not self._sandbox_manager or not self._sandbox_manager.available:
            raise RuntimeError(
                "CodingAgent requires a SandboxManager with an available OpenShell "
                "gateway. Executing LLM-generated code without sandbox isolation "
                "is not permitted. Provide a sandbox_manager to CodingAgent or "
                "CodingDeps, or start the OpenShell gateway."
            )

        run_cmd = test_command.strip()
        if not run_cmd:
            run_cmd = self._default_run_command(file_path, language)

        # Write code inside the sandbox via exec, not on the host filesystem
        write_cmd = (
            f"mkdir -p $(dirname {file_path}) && "
            f"cat > {file_path} << 'SANDBOX_CODE_EOF'\n{code}\nSANDBOX_CODE_EOF"
        )
        self._sandbox_manager.exec_in_sandbox(
            agent_type="coding_agent",
            command=["sh", "-c", write_cmd],
            timeout_seconds=30,
        )

        result = self._sandbox_manager.exec_in_sandbox(
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
