"""Unit tests for the coding agent CLI — REPL commands, apply, streaming."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cogniverse_cli.code import CodingSession, _handle_slash_command
from cogniverse_cli.index import CODE_EXTENSIONS, collect_files
from cogniverse_cli.streaming import (
    CodingResult,
    _build_a2a_request,
    _parse_coding_result,
)


@pytest.mark.unit
@pytest.mark.ci_fast
class TestA2ARequestBuilder:
    def test_builds_valid_jsonrpc_request(self):
        req = _build_a2a_request("write a fibonacci function", tenant_id="acme")
        assert req["jsonrpc"] == "2.0"
        assert req["method"] == "tasks/sendSubscribe"
        assert req["params"]["metadata"]["agent_name"] == "coding_agent"
        assert req["params"]["metadata"]["query"] == "write a fibonacci function"
        assert req["params"]["metadata"]["tenant_id"] == "acme"
        assert req["params"]["metadata"]["stream"] is True

    def test_includes_conversation_history(self):
        history = [
            {"role": "user", "content": "write a retry decorator"},
            {"role": "assistant", "content": "Created retry.py"},
        ]
        req = _build_a2a_request("now add tests", conversation_history=history)
        assert len(req["params"]["history"]) == 2
        assert req["params"]["history"][0]["role"] == "user"
        assert req["params"]["history"][0]["parts"][0]["text"] == "write a retry decorator"

    def test_includes_context(self):
        req = _build_a2a_request(
            "add pagination",
            context={"language": "rust", "max_iterations": 3},
        )
        meta = req["params"]["metadata"]
        assert meta["language"] == "rust"
        assert meta["max_iterations"] == 3


@pytest.mark.unit
@pytest.mark.ci_fast
class TestCodingResultParser:
    def test_parses_full_coding_output(self):
        data = {
            "status": "success",
            "result": {
                "plan": "1. Create retry.py\n2. Add tests",
                "code_changes": [
                    {"file_path": "retry.py", "content": "def retry(): pass", "change_type": "new"},
                ],
                "execution_results": [{"exit_code": 0, "stdout": "OK"}],
                "summary": "Created retry decorator",
                "iterations_used": 1,
                "files_modified": ["retry.py"],
            },
        }
        result = _parse_coding_result(data)
        assert result.plan == "1. Create retry.py\n2. Add tests"
        assert len(result.code_changes) == 1
        assert result.code_changes[0]["file_path"] == "retry.py"
        assert result.summary == "Created retry decorator"
        assert result.iterations_used == 1
        assert result.files_modified == ["retry.py"]

    def test_handles_string_result(self):
        result = _parse_coding_result({"result": "plain text response"})
        assert result.summary == "plain text response"

    def test_handles_empty_result(self):
        result = _parse_coding_result({})
        assert result.plan == ""
        assert result.code_changes == []


@pytest.mark.unit
@pytest.mark.ci_fast
class TestCodingSession:
    def _make_session(self):
        return CodingSession(
            tenant_id="test",
            language="python",
            max_iterations=3,
            codebase_path="",
            runtime_url="http://localhost:28000",
        )

    def test_apply_writes_new_file(self):
        session = self._make_session()
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = str(Path(tmpdir) / "new_file.py")
            session.last_result = CodingResult(
                code_changes=[
                    {"file_path": file_path, "content": "print('hello')", "change_type": "new"},
                ],
            )
            count = session.apply()
            assert count == 1
            assert Path(file_path).read_text() == "print('hello')"

    def test_apply_deletes_file(self):
        session = self._make_session()
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "to_delete.py"
            file_path.write_text("old content")
            session.last_result = CodingResult(
                code_changes=[
                    {"file_path": str(file_path), "content": "", "change_type": "delete"},
                ],
            )
            count = session.apply()
            assert count == 1
            assert not file_path.exists()

    def test_apply_no_changes(self):
        session = self._make_session()
        session.last_result = None
        count = session.apply()
        assert count == 0

    def test_clear_resets_history(self):
        session = self._make_session()
        session.history = [{"role": "user", "content": "test"}]
        session.last_result = CodingResult(summary="test")
        session.clear()
        assert session.history == []
        assert session.last_result is None

    def test_send_appends_to_history(self):
        session = self._make_session()
        mock_result = CodingResult(summary="Done")
        with patch("cogniverse_cli.code.stream_coding_response", return_value=mock_result):
            session.send("write a function")
        assert len(session.history) == 2
        assert session.history[0] == {"role": "user", "content": "write a function"}
        assert session.history[1] == {"role": "assistant", "content": "Done"}


@pytest.mark.unit
@pytest.mark.ci_fast
class TestSlashCommands:
    def _make_session(self):
        return CodingSession(
            tenant_id="test",
            language="python",
            max_iterations=5,
            codebase_path="",
            runtime_url="http://localhost:28000",
        )

    def test_exit_returns_false(self):
        assert _handle_slash_command(self._make_session(), "/exit") is False

    def test_language_sets_language(self):
        session = self._make_session()
        _handle_slash_command(session, "/language rust")
        assert session.language == "rust"

    def test_iterations_sets_value(self):
        session = self._make_session()
        _handle_slash_command(session, "/iterations 3")
        assert session.max_iterations == 3

    def test_codebase_sets_path(self):
        session = self._make_session()
        _handle_slash_command(session, "/codebase ./src")
        assert session.codebase_path == "./src"

    def test_clear_resets(self):
        session = self._make_session()
        session.history = [{"role": "user", "content": "x"}]
        _handle_slash_command(session, "/clear")
        assert session.history == []

    def test_unknown_command_continues(self):
        assert _handle_slash_command(self._make_session(), "/bogus") is True


@pytest.mark.unit
@pytest.mark.ci_fast
class TestFileCollector:
    def test_collects_python_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "main.py").write_text("print('hi')")
            (root / "utils.py").write_text("x = 1")
            (root / "readme.md").write_text("docs")
            (root / "data.csv").write_text("a,b")

            files = collect_files(root, "code")
            names = [f.name for f in files]
            assert "main.py" in names
            assert "utils.py" in names
            assert "readme.md" not in names
            assert "data.csv" not in names

    def test_ignores_venv_and_pycache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "app.py").write_text("x = 1")
            venv = root / ".venv" / "lib"
            venv.mkdir(parents=True)
            (venv / "site.py").write_text("y = 2")
            cache = root / "__pycache__"
            cache.mkdir()
            (cache / "app.cpython-312.pyc").write_text("z")

            files = collect_files(root, "code")
            paths = [str(f) for f in files]
            assert any("app.py" in p for p in paths)
            assert not any(".venv" in p for p in paths)
            assert not any("__pycache__" in p for p in paths)

    def test_collects_docs_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "readme.md").write_text("docs")
            (root / "guide.txt").write_text("guide")
            (root / "main.py").write_text("code")

            files = collect_files(root, "docs")
            names = [f.name for f in files]
            assert "readme.md" in names
            assert "guide.txt" in names
            assert "main.py" not in names
