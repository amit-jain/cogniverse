"""Cluster-free tests for the e2e CronWorkflow guard."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

import pytest

import tests.e2e.conftest as e2e_conftest
import tests.e2e.cron_guard as cron_guard

# cron_guard reads this from the e2e conftest at call time; deriving it here
# means a context change fails on the real contract rather than on a literal.
KUBECTL_CTX = e2e_conftest.KUBECTL_CONTEXT


@dataclass
class _CronState:
    suspend: bool
    annotation: str | None = None

    def as_item(self, name: str) -> dict:
        annotations = {}
        if self.annotation is not None:
            annotations[cron_guard.E2E_SUSPENDED_ANNOTATION] = self.annotation
        metadata: dict = {"name": name}
        if annotations:
            metadata["annotations"] = annotations
        return {"metadata": metadata, "spec": {"suspend": self.suspend}}


class _FakeKubectl:
    def __init__(
        self,
        *,
        items: dict[str, _CronState],
        get_rc: int = 0,
        get_stderr: str = "",
        patch_failures_on_suspend: set[str] | None = None,
        annotate_failures: set[str] | None = None,
    ) -> None:
        self.items = {
            name: _CronState(state.suspend, state.annotation)
            for name, state in items.items()
        }
        self.order = list(items)
        self.get_rc = get_rc
        self.get_stderr = get_stderr
        self.patch_failures_on_suspend = patch_failures_on_suspend or set()
        self.annotate_failures = annotate_failures or set()
        self.commands: list[list[str]] = []
        self.context: str | None = None

    def _payload(self) -> str:
        return json.dumps(
            {"items": [self.items[name].as_item(name) for name in self.order]}
        )

    def run(self, argv, **kwargs):  # noqa: ANN001
        command = list(argv)
        self.commands.append(command)

        # Every kubectl call must name its context. An omitted --context acts
        # on whatever context happens to be current, which on a developer box
        # is a different cluster entirely.
        assert command[0] == "kubectl", command
        assert command[1] == "--context", command
        # An empty or flag-shaped context silently acts on whatever context is
        # current, so it must be a real name -- and the same one every call.
        assert command[2] and not command[2].startswith("-"), command
        if self.context is None:
            self.context = command[2]
        assert command[2] == self.context, command
        rest = command[3:]

        if rest[:2] == ["get", "cronworkflows"]:
            return subprocess.CompletedProcess(
                command,
                self.get_rc,
                stdout="" if self.get_rc else self._payload(),
                stderr=self.get_stderr if self.get_rc else "",
            )

        if rest[:2] == ["patch", "cronworkflow"]:
            name = rest[2]
            suspend = json.loads(command[-1])["spec"]["suspend"]
            if suspend and name in self.patch_failures_on_suspend:
                return subprocess.CompletedProcess(
                    command,
                    1,
                    stdout="",
                    stderr=f"patch failed for {name}",
                )
            self.items[name].suspend = suspend
            return subprocess.CompletedProcess(
                command, 0, stdout=f"{name}\n", stderr=""
            )

        if rest[:1] == ["annotate"]:
            name = rest[2]
            if name in self.annotate_failures:
                return subprocess.CompletedProcess(
                    command,
                    1,
                    stdout="",
                    stderr=f"annotate failed for {name}",
                )
            marker = rest[3]
            if marker.endswith("-"):
                self.items[name].annotation = None
            else:
                _, token = marker.split("=", 1)
                self.items[name].annotation = token
            return subprocess.CompletedProcess(
                command, 0, stdout=f"{name}\n", stderr=""
            )

        raise AssertionError(f"unexpected kubectl argv: {command}")


def _patch_kubectl(monkeypatch, fake: _FakeKubectl) -> None:
    monkeypatch.setattr(cron_guard.subprocess, "run", fake.run)


def test_suspend_then_restore_finally_only_touches_session_owned_cronworkflows(
    monkeypatch,
):
    fake = _FakeKubectl(
        items={
            "alpha": _CronState(suspend=False),
            "beta": _CronState(suspend=True),
        }
    )
    _patch_kubectl(monkeypatch, fake)

    session_token = "session-token-1"
    cron_restore = ()
    restore = None
    with pytest.raises(RuntimeError, match="setup failure"):
        try:
            suspend = cron_guard.suspend_cronworkflows_for_session(session_token)
            assert suspend.restore_names == ("alpha",)
            assert suspend.failures == ()
            cron_restore = suspend.restore_names
            raise RuntimeError("setup failure")
        finally:
            restore = cron_guard.restore_cronworkflows(cron_restore)

    assert restore.restored_names == ("alpha",)
    assert restore.failures == ()
    assert fake.commands == [
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "get",
            "cronworkflows",
            "-n",
            "cogniverse",
            "-o",
            "json",
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "annotate",
            "cronworkflow",
            "alpha",
            "cogniverse.io/e2e-suspended=session-token-1",
            "-n",
            "cogniverse",
            "--overwrite",
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "patch",
            "cronworkflow",
            "alpha",
            "-n",
            "cogniverse",
            "--type",
            "merge",
            "-p",
            '{"spec":{"suspend":true}}',
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "get",
            "cronworkflows",
            "-n",
            "cogniverse",
            "-o",
            "json",
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "patch",
            "cronworkflow",
            "alpha",
            "-n",
            "cogniverse",
            "--type",
            "merge",
            "-p",
            '{"spec":{"suspend":false}}',
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "annotate",
            "cronworkflow",
            "alpha",
            "cogniverse.io/e2e-suspended-",
            "-n",
            "cogniverse",
        ],
    ]


def test_already_suspended_by_user_cronworkflow_is_never_toggled_or_restored(
    monkeypatch,
):
    fake = _FakeKubectl(
        items={
            "user-paused": _CronState(suspend=True),
            "alpha": _CronState(suspend=False),
        }
    )
    _patch_kubectl(monkeypatch, fake)

    suspend = cron_guard.suspend_cronworkflows_for_session("session-token-1")
    assert suspend.restore_names == ("alpha",)
    assert suspend.failures == ()

    restore = cron_guard.restore_cronworkflows(suspend.restore_names)
    assert restore.restored_names == ("alpha",)
    assert restore.failures == ()

    assert fake.commands == [
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "get",
            "cronworkflows",
            "-n",
            "cogniverse",
            "-o",
            "json",
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "annotate",
            "cronworkflow",
            "alpha",
            "cogniverse.io/e2e-suspended=session-token-1",
            "-n",
            "cogniverse",
            "--overwrite",
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "patch",
            "cronworkflow",
            "alpha",
            "-n",
            "cogniverse",
            "--type",
            "merge",
            "-p",
            '{"spec":{"suspend":true}}',
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "get",
            "cronworkflows",
            "-n",
            "cogniverse",
            "-o",
            "json",
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "patch",
            "cronworkflow",
            "alpha",
            "-n",
            "cogniverse",
            "--type",
            "merge",
            "-p",
            '{"spec":{"suspend":false}}',
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "annotate",
            "cronworkflow",
            "alpha",
            "cogniverse.io/e2e-suspended-",
            "-n",
            "cogniverse",
        ],
    ]


def test_restore_stale_annotation_from_previous_session(monkeypatch):
    fake = _FakeKubectl(
        items={
            "stale": _CronState(
                suspend=True,
                annotation="previous-session-token",
            )
        }
    )
    _patch_kubectl(monkeypatch, fake)

    restored = cron_guard.restore_stale_cronworkflows("current-session-token")

    assert restored.restored_names == ("stale",)
    assert restored.failures == ()
    assert fake.commands == [
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "get",
            "cronworkflows",
            "-n",
            "cogniverse",
            "-o",
            "json",
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "get",
            "cronworkflows",
            "-n",
            "cogniverse",
            "-o",
            "json",
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "patch",
            "cronworkflow",
            "stale",
            "-n",
            "cogniverse",
            "--type",
            "merge",
            "-p",
            '{"spec":{"suspend":false}}',
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "annotate",
            "cronworkflow",
            "stale",
            "cogniverse.io/e2e-suspended-",
            "-n",
            "cogniverse",
        ],
    ]


def test_restore_stale_skips_cronworkflows_owned_by_this_session(monkeypatch):
    fake = _FakeKubectl(
        items={
            "owned": _CronState(
                suspend=True,
                annotation="current-session-token",
            )
        }
    )
    _patch_kubectl(monkeypatch, fake)

    restored = cron_guard.restore_stale_cronworkflows("current-session-token")

    assert restored.restored_names == ()
    assert restored.failures == ()
    assert fake.commands == [
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "get",
            "cronworkflows",
            "-n",
            "cogniverse",
            "-o",
            "json",
        ],
    ]


def test_get_failure_raises_with_rc_and_stderr(monkeypatch):
    fake = _FakeKubectl(
        items={},
        get_rc=1,
        get_stderr="boom",
    )
    _patch_kubectl(monkeypatch, fake)

    with pytest.raises(RuntimeError) as exc_info:
        cron_guard.restore_stale_cronworkflows("current-session-token")

    assert str(exc_info.value) == "kubectl get cronworkflows failed: rc=1 stderr='boom'"
    assert fake.commands == [
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "get",
            "cronworkflows",
            "-n",
            "cogniverse",
            "-o",
            "json",
        ],
    ]


def test_suspend_records_patch_failures_and_keeps_the_broken_cronworkflow(
    monkeypatch,
):
    fake = _FakeKubectl(
        items={
            "good-a": _CronState(suspend=False),
            "broken": _CronState(suspend=False),
            "good-b": _CronState(suspend=False),
        },
        patch_failures_on_suspend={"broken"},
    )
    _patch_kubectl(monkeypatch, fake)

    suspended = cron_guard.suspend_cronworkflows_for_session("session-token-1")

    assert suspended.restore_names == ("good-a", "broken", "good-b")
    assert suspended.failures == ("broken",)
    assert fake.commands == [
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "get",
            "cronworkflows",
            "-n",
            "cogniverse",
            "-o",
            "json",
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "annotate",
            "cronworkflow",
            "good-a",
            "cogniverse.io/e2e-suspended=session-token-1",
            "-n",
            "cogniverse",
            "--overwrite",
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "patch",
            "cronworkflow",
            "good-a",
            "-n",
            "cogniverse",
            "--type",
            "merge",
            "-p",
            '{"spec":{"suspend":true}}',
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "annotate",
            "cronworkflow",
            "broken",
            "cogniverse.io/e2e-suspended=session-token-1",
            "-n",
            "cogniverse",
            "--overwrite",
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "patch",
            "cronworkflow",
            "broken",
            "-n",
            "cogniverse",
            "--type",
            "merge",
            "-p",
            '{"spec":{"suspend":true}}',
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "annotate",
            "cronworkflow",
            "good-b",
            "cogniverse.io/e2e-suspended=session-token-1",
            "-n",
            "cogniverse",
            "--overwrite",
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "patch",
            "cronworkflow",
            "good-b",
            "-n",
            "cogniverse",
            "--type",
            "merge",
            "-p",
            '{"spec":{"suspend":true}}',
        ],
    ]

    restored = cron_guard.restore_cronworkflows(suspended.restore_names)

    assert restored.restored_names == ("good-a", "broken", "good-b")
    assert restored.failures == ()


def test_restore_is_idempotent_for_already_restored_cronworkflows(monkeypatch):
    fake = _FakeKubectl(
        items={
            "alpha": _CronState(
                suspend=True,
                annotation="session-token-1",
            ),
            "beta": _CronState(
                suspend=True,
                annotation="session-token-1",
            ),
        }
    )
    _patch_kubectl(monkeypatch, fake)

    first = cron_guard.restore_cronworkflows(("alpha", "beta"))
    second = cron_guard.restore_cronworkflows(("alpha", "beta"))

    assert first.restored_names == ("alpha", "beta")
    assert first.failures == ()
    assert second.restored_names == ()
    assert second.failures == ()
    assert fake.commands == [
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "get",
            "cronworkflows",
            "-n",
            "cogniverse",
            "-o",
            "json",
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "patch",
            "cronworkflow",
            "alpha",
            "-n",
            "cogniverse",
            "--type",
            "merge",
            "-p",
            '{"spec":{"suspend":false}}',
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "annotate",
            "cronworkflow",
            "alpha",
            "cogniverse.io/e2e-suspended-",
            "-n",
            "cogniverse",
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "patch",
            "cronworkflow",
            "beta",
            "-n",
            "cogniverse",
            "--type",
            "merge",
            "-p",
            '{"spec":{"suspend":false}}',
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "annotate",
            "cronworkflow",
            "beta",
            "cogniverse.io/e2e-suspended-",
            "-n",
            "cogniverse",
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CTX,
            "get",
            "cronworkflows",
            "-n",
            "cogniverse",
            "-o",
            "json",
        ],
    ]
