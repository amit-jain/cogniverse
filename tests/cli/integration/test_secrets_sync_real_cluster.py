"""Secrets the chart mounts must actually exist in the cluster after a sync.

The messaging deployment reads TELEGRAM_BOT_TOKEN from
``<release>-messaging-secrets`` key ``telegram-bot-token``. Nothing created
that Secret, so enabling messaging left the gateway pod unable to start. This
drives the real sync with real kubectl against a Kubernetes API server the
session boots itself (``ephemeral_k8s_cluster``) and reads the Secret back,
asserting the stored value is exactly the token that was supplied and that
it lives under the key and name the chart mounts — a Secret with the right
name but the wrong key fails here just as loudly as a missing one.
"""

from __future__ import annotations

import base64
import json
import subprocess
import uuid

import pytest
from cogniverse_cli.constants import NAMESPACE
from cogniverse_cli.secrets import (
    MESSAGING_SECRET,
    TELEGRAM_TOKEN_KEY,
    sync_telegram_token_to_cluster,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_docker]


@pytest.fixture(autouse=True)
def _use_test_owned_cluster(ephemeral_k8s_cluster, monkeypatch):
    """Point kubectl — ours and the sync helper's — at the session's own
    API server, never a developer's cluster."""
    monkeypatch.setenv("KUBECONFIG", ephemeral_k8s_cluster["kubeconfig"])


def _read_secret_value(name: str, key: str) -> str | None:
    out = subprocess.run(
        ["kubectl", "get", "secret", name, "-n", NAMESPACE, "-o", "json"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if out.returncode != 0:
        return None
    data = json.loads(out.stdout).get("data", {})
    if key not in data:
        return None
    return base64.b64decode(data[key]).decode()


@pytest.fixture
def restore_secret(_use_test_owned_cluster):
    """Preserve and restore whatever the cluster already had."""
    original = _read_secret_value(MESSAGING_SECRET, TELEGRAM_TOKEN_KEY)
    yield
    if original is None:
        subprocess.run(
            ["kubectl", "delete", "secret", MESSAGING_SECRET, "-n", NAMESPACE],
            capture_output=True,
            check=False,
            timeout=30,
        )
    else:
        rendered = subprocess.run(
            [
                "kubectl",
                "create",
                "secret",
                "generic",
                MESSAGING_SECRET,
                "-n",
                NAMESPACE,
                f"--from-literal={TELEGRAM_TOKEN_KEY}={original}",
                "--dry-run=client",
                "-o",
                "yaml",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=rendered.stdout,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )


class TestTelegramSecretSync:
    def test_sync_stores_the_token_the_chart_mounts(self, monkeypatch, restore_secret):
        token = f"9999999:TEST{uuid.uuid4().hex}"
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)

        assert sync_telegram_token_to_cluster(required=True) is True

        # Exactly what the messaging deployment's secretKeyRef resolves.
        assert _read_secret_value(MESSAGING_SECRET, TELEGRAM_TOKEN_KEY) == token

    def test_resync_replaces_a_rotated_token(self, monkeypatch, restore_secret):
        first = f"1111111:OLD{uuid.uuid4().hex}"
        second = f"2222222:NEW{uuid.uuid4().hex}"

        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", first)
        assert sync_telegram_token_to_cluster(required=True) is True
        assert _read_secret_value(MESSAGING_SECRET, TELEGRAM_TOKEN_KEY) == first

        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", second)
        assert sync_telegram_token_to_cluster(required=True) is True
        assert _read_secret_value(MESSAGING_SECRET, TELEGRAM_TOKEN_KEY) == second

    def test_token_read_from_the_project_env_directory(
        self, monkeypatch, tmp_path, restore_secret
    ):
        """The gitignored ./.env/TELEGRAM_BOT_TOKEN.env is the dev source."""
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        token = f"3333333:FILE{uuid.uuid4().hex}"
        env_dir = tmp_path / "project"
        env_dir.mkdir()
        (env_dir / "TELEGRAM_BOT_TOKEN.env").write_text(f"TELEGRAM_BOT_TOKEN={token}\n")
        monkeypatch.setattr("cogniverse_cli.secrets.PROJECT_ENV", env_dir)
        monkeypatch.setattr("cogniverse_cli.secrets.HOME_ENV", tmp_path / "absent")

        assert sync_telegram_token_to_cluster(required=True) is True
        assert _read_secret_value(MESSAGING_SECRET, TELEGRAM_TOKEN_KEY) == token

    def test_home_env_is_used_when_the_project_has_no_copy(
        self, monkeypatch, tmp_path, restore_secret
    ):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        token = f"4444444:HOME{uuid.uuid4().hex}"
        home_env = tmp_path / "home.env"
        home_env.write_text(f"# shared\nTELEGRAM_BOT_TOKEN={token}\n")
        monkeypatch.setattr("cogniverse_cli.secrets.PROJECT_ENV", tmp_path / "absent")
        monkeypatch.setattr("cogniverse_cli.secrets.HOME_ENV", home_env)

        assert sync_telegram_token_to_cluster(required=True) is True
        assert _read_secret_value(MESSAGING_SECRET, TELEGRAM_TOKEN_KEY) == token

    def test_project_env_overrides_home_env(
        self, monkeypatch, tmp_path, restore_secret
    ):
        """Both present: the project copy wins, per-variable."""
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        project_token = f"5555555:PROJ{uuid.uuid4().hex}"
        home_token = f"6666666:HOME{uuid.uuid4().hex}"

        env_dir = tmp_path / "project"
        env_dir.mkdir()
        (env_dir / "TELEGRAM_BOT_TOKEN.env").write_text(project_token)
        home_env = tmp_path / "home.env"
        home_env.write_text(f"TELEGRAM_BOT_TOKEN={home_token}\n")
        monkeypatch.setattr("cogniverse_cli.secrets.PROJECT_ENV", env_dir)
        monkeypatch.setattr("cogniverse_cli.secrets.HOME_ENV", home_env)

        assert sync_telegram_token_to_cluster(required=True) is True
        assert _read_secret_value(MESSAGING_SECRET, TELEGRAM_TOKEN_KEY) == project_token

    def test_environment_variable_beats_both_files(
        self, monkeypatch, tmp_path, restore_secret
    ):
        env_token = f"7777777:ENV{uuid.uuid4().hex}"
        env_dir = tmp_path / "project"
        env_dir.mkdir()
        (env_dir / "TELEGRAM_BOT_TOKEN.env").write_text("1:FROMPROJECT")
        home_env = tmp_path / "home.env"
        home_env.write_text("TELEGRAM_BOT_TOKEN=2:FROMHOME\n")
        monkeypatch.setattr("cogniverse_cli.secrets.PROJECT_ENV", env_dir)
        monkeypatch.setattr("cogniverse_cli.secrets.HOME_ENV", home_env)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", env_token)

        assert sync_telegram_token_to_cluster(required=True) is True
        assert _read_secret_value(MESSAGING_SECRET, TELEGRAM_TOKEN_KEY) == env_token

    def test_missing_token_reports_failure_and_writes_nothing(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.setattr("cogniverse_cli.secrets.PROJECT_ENV", tmp_path / "absent")
        monkeypatch.setattr("cogniverse_cli.secrets.HOME_ENV", tmp_path / "gone")
        before = _read_secret_value(MESSAGING_SECRET, TELEGRAM_TOKEN_KEY)

        assert sync_telegram_token_to_cluster(required=True) is False

        assert _read_secret_value(MESSAGING_SECRET, TELEGRAM_TOKEN_KEY) == before
