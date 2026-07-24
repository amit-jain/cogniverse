"""Unit tests for the coding agent CLI — REPL commands, apply, streaming."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from cogniverse_cli.code import CodingSession, _handle_slash_command
from cogniverse_cli.index import collect_files
from cogniverse_cli.streaming import (
    CodingResult,
    _build_a2a_request,
    _handle_event,
    _parse_coding_result,
)


@pytest.mark.unit
@pytest.mark.ci_fast
class TestA2ARequestBuilder:
    def test_builds_valid_jsonrpc_request(self):
        req = _build_a2a_request("write a fibonacci function", tenant_id="acme")
        assert req["jsonrpc"] == "2.0"
        assert req["method"] == "message/stream"
        assert req["params"]["metadata"]["agent_name"] == "coding_agent"
        assert req["params"]["metadata"]["query"] == "write a fibonacci function"
        assert req["params"]["metadata"]["tenant_id"] == "acme"
        assert req["params"]["metadata"]["stream"] is True
        msg = req["params"]["message"]
        assert msg["kind"] == "message"
        assert msg["parts"][0]["text"] == "write a fibonacci function"

    def test_includes_conversation_history(self):
        history = [
            {"role": "user", "content": "write a retry decorator"},
            {"role": "assistant", "content": "Created retry.py"},
        ]
        req = _build_a2a_request(
            "now add tests", tenant_id="acme", conversation_history=history
        )
        msg_meta = req["params"]["message"]["metadata"]
        assert msg_meta["conversation_history"] == history

    def test_includes_context(self):
        req = _build_a2a_request(
            "add pagination",
            tenant_id="acme",
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
                    {
                        "file_path": "retry.py",
                        "content": "def retry(): pass",
                        "change_type": "new",
                    },
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
                    {
                        "file_path": file_path,
                        "content": "print('hello')",
                        "change_type": "new",
                    },
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
                    {
                        "file_path": str(file_path),
                        "content": "",
                        "change_type": "delete",
                    },
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
        with patch(
            "cogniverse_cli.code.stream_coding_response", return_value=mock_result
        ):
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


class TestApplyFailureIsolation:
    def _session_with_changes(self, changes):
        s = CodingSession(
            tenant_id="acme:acme",
            language="python",
            max_iterations=3,
            codebase_path="",
            runtime_url="http://runtime.test",
        )
        s.last_result = CodingResult(code_changes=changes, summary="s")
        return s

    def test_apply_reports_failed_file_and_continues(self, tmp_path: Path):
        """One unwritable target degrades to a per-file failure report — it
        previously raised out of /apply, killing the REPL with part of the
        change set already on disk and no summary of what landed."""
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file, not a directory")

        ok1 = tmp_path / "a.py"
        bad = blocker / "nested" / "b.py"  # parent mkdir fails: blocker is a file
        ok2 = tmp_path / "c.py"

        s = self._session_with_changes(
            [
                {"file_path": str(ok1), "content": "A", "change_type": "new"},
                {"file_path": str(bad), "content": "B", "change_type": "new"},
                {"file_path": str(ok2), "content": "C", "change_type": "new"},
            ]
        )
        applied = s.apply()

        assert applied == 2
        assert ok1.read_text() == "A"
        assert ok2.read_text() == "C"
        assert not bad.exists()

    def test_show_diff_survives_unreadable_file(self, tmp_path: Path):
        target = tmp_path / "x.py"
        target.write_text("old")
        s = self._session_with_changes(
            [{"file_path": str(target), "content": "new", "change_type": "modify"}]
        )
        with patch.object(Path, "read_text", side_effect=OSError("EACCES")):
            s.show_diff()  # must not raise


class TestRunReplStartupContract:
    def test_dead_runtime_exits_2(self):
        """An unreachable runtime exits 2 (script-detectable) — previously
        printed a message and exited 0. ConnectTimeout (SYN blackhole) is
        covered too, not just ConnectError."""
        import httpx
        from cogniverse_cli.code import run_repl

        for exc in (
            httpx.ConnectError("refused"),
            httpx.ConnectTimeout("blackhole"),
        ):
            with patch("httpx.get", side_effect=exc):
                with pytest.raises(SystemExit) as se:
                    run_repl(tenant_id="acme:acme", runtime_url="http://runtime.test")
                assert se.value.code == 2

    def test_unhealthy_runtime_exits_2(self):
        import httpx
        from cogniverse_cli.code import run_repl

        resp = httpx.Response(503, request=httpx.Request("GET", "http://x/health"))
        with patch("httpx.get", return_value=resp):
            with pytest.raises(SystemExit) as se:
                run_repl(tenant_id="acme:acme", runtime_url="http://runtime.test")
            assert se.value.code == 2


class TestHandleEventShapeGuards:
    """Malformed SSE events degrade to "keep current phase" — each of these
    shapes previously raised AttributeError/TypeError and aborted the REPL
    stream mid-session."""

    def test_non_dict_result(self):
        assert _handle_event({"result": "just a string"}, "plan") == "plan"

    def test_non_dict_status(self):
        assert _handle_event({"result": {"status": "running"}}, "plan") == "plan"

    def test_null_parts(self):
        event = {"result": {"status": {"state": "working", "message": {"parts": None}}}}
        assert _handle_event(event, "plan") == "plan"

    def test_scalar_json_part_text(self):
        event = {
            "result": {
                "status": {
                    "state": "working",
                    "message": {"parts": [{"kind": "text", "text": "42"}]},
                }
            }
        }
        assert _handle_event(event, "plan") == "plan"

    def test_final_with_null_data(self):
        event = {
            "result": {
                "status": {
                    "state": "completed",
                    "message": {
                        "parts": [
                            {"kind": "text", "text": '{"type": "final", "data": null}'}
                        ]
                    },
                }
            }
        }
        out = _handle_event(event, "plan")
        assert isinstance(out, CodingResult)
        assert out.summary == ""

    def test_parse_coding_result_non_dict_nested_result(self):
        out = _parse_coding_result({"result": [1, 2, 3]})
        assert isinstance(out, CodingResult)
        assert "1" in out.summary
