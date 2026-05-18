"""Unit tests for the cronworkflow suspend/restore helpers in conftest.

These pin the only-toggle-and-restore-what-we-touched contract — a
cronworkflow that the user (or a prior session) suspended manually must
remain suspended on teardown, otherwise leftover patches accumulate
across runs and the cleanup turns into noise.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from unittest import mock

import pytest

from tests.e2e.conftest import (
    _restore_cronworkflows,
    _suspend_cronworkflows_for_session,
)


def _completed(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def _cron_payload(items: list[tuple[str, bool]]) -> str:
    return json.dumps(
        {
            "items": [
                {"metadata": {"name": name}, "spec": {"suspend": suspended}}
                for name, suspended in items
            ]
        }
    )


class TestSuspendCronworkflows:
    def test_only_unsuspended_workflows_are_patched(self):
        """``spec.suspend == true`` already → fixture leaves it alone."""
        listing = _cron_payload(
            [
                ("already-paused", True),
                ("running-a", False),
                ("running-b", False),
            ]
        )
        patched: list[str] = []

        def fake_run(cmd, capture_output=True, text=True, timeout=30):
            assert cmd[0] == "kubectl"
            if cmd[1] == "get":
                return _completed(stdout=listing)
            if cmd[1] == "patch":
                patched.append(cmd[3])
                return _completed()
            raise AssertionError(f"unexpected kubectl call: {cmd!r}")

        with (
            mock.patch("shutil.which", return_value="/usr/bin/kubectl"),
            mock.patch.object(subprocess, "run", side_effect=fake_run),
        ):
            toggled = _suspend_cronworkflows_for_session()

        assert patched == ["running-a", "running-b"], (
            f"only un-suspended cronworkflows must be patched; got patched={patched}"
        )
        assert toggled == ["running-a", "running-b"]

    def test_kubectl_missing_returns_empty(self):
        """No ``kubectl`` binary → fixture is a no-op (CI without cluster)."""
        with mock.patch("shutil.which", return_value=None):
            toggled = _suspend_cronworkflows_for_session()
        assert toggled == []

    def test_crd_not_installed_returns_empty(self):
        """No CronWorkflow CRD → kubectl prints 'could not find' to stderr."""

        def fake_run(cmd, capture_output=True, text=True, timeout=30):
            return _completed(
                returncode=1,
                stderr="error: the server could not find the requested resource",
            )

        with (
            mock.patch("shutil.which", return_value="/usr/bin/kubectl"),
            mock.patch.object(subprocess, "run", side_effect=fake_run),
        ):
            toggled = _suspend_cronworkflows_for_session()
        assert toggled == []

    def test_patch_failure_skips_the_failed_one(self):
        """One bad patch must not abort suspension of the others."""
        listing = _cron_payload([("a", False), ("b", False), ("c", False)])

        def fake_run(cmd, capture_output=True, text=True, timeout=30):
            if cmd[1] == "get":
                return _completed(stdout=listing)
            assert cmd[1] == "patch"
            if cmd[3] == "b":
                return _completed(returncode=1, stderr="forbidden")
            return _completed()

        with (
            mock.patch("shutil.which", return_value="/usr/bin/kubectl"),
            mock.patch.object(subprocess, "run", side_effect=fake_run),
        ):
            toggled = _suspend_cronworkflows_for_session()

        assert toggled == ["a", "c"], (
            f"failed patch must drop the name from the restore list; "
            f"got toggled={toggled}"
        )


class TestRestoreCronworkflows:
    def test_restores_only_provided_names(self):
        """Restore patches exactly the names the fixture toggled."""
        patched: list[tuple[str, str]] = []

        def fake_run(cmd, capture_output=True, text=True, timeout=30):
            assert cmd[1] == "patch"
            patched.append((cmd[3], cmd[-1]))
            return _completed()

        with (
            mock.patch("shutil.which", return_value="/usr/bin/kubectl"),
            mock.patch.object(subprocess, "run", side_effect=fake_run),
        ):
            _restore_cronworkflows(["a", "b"])

        names = [n for n, _ in patched]
        payloads = {p for _, p in patched}
        assert names == ["a", "b"]
        assert payloads == {'{"spec":{"suspend":false}}'}

    def test_empty_list_no_kubectl_invoked(self):
        """Empty restore list short-circuits — no subprocess work."""
        with mock.patch.object(subprocess, "run") as run:
            _restore_cronworkflows([])
        assert run.call_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
