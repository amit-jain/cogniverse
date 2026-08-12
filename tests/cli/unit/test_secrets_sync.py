"""Unit tests for cogniverse_cli.secrets."""

from __future__ import annotations

import subprocess
from unittest.mock import patch


def _ok(stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=0, stdout=stdout, stderr=stderr
    )


def _err(returncode: int = 1, stderr: str = "boom") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout="", stderr=stderr
    )


def test_returns_false_when_token_missing_and_not_required(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(
        "cogniverse_cli.secrets.HF_CACHE_TOKEN_PATH", tmp_path / "missing"
    )
    from cogniverse_cli.secrets import sync_hf_token_to_cluster

    assert sync_hf_token_to_cluster(required=False) is False


def test_returns_false_when_token_missing_and_required(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(
        "cogniverse_cli.secrets.HF_CACHE_TOKEN_PATH", tmp_path / "missing"
    )
    from cogniverse_cli.secrets import sync_hf_token_to_cluster

    # Still returns False, caller decides what to do with required.
    assert sync_hf_token_to_cluster(required=True) is False


def test_hf_token_env_wins_over_cache_file(monkeypatch, tmp_path):
    cache = tmp_path / "token"
    cache.write_text("cached-token-value")
    monkeypatch.setenv("HF_TOKEN", "env-token-value")
    monkeypatch.setattr("cogniverse_cli.secrets.HF_CACHE_TOKEN_PATH", cache)

    from cogniverse_cli.secrets import _read_hf_token

    assert _read_hf_token() == "env-token-value"


def test_cache_file_used_when_no_env(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    cache = tmp_path / "token"
    cache.write_text("  cached-token-value  \n")  # whitespace gets stripped
    monkeypatch.setattr("cogniverse_cli.secrets.HF_CACHE_TOKEN_PATH", cache)

    from cogniverse_cli.secrets import _read_hf_token

    assert _read_hf_token() == "cached-token-value"


def test_sync_creates_namespace_and_applies_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "the-token")
    monkeypatch.setattr(
        "cogniverse_cli.secrets.HF_CACHE_TOKEN_PATH", tmp_path / "unused"
    )

    calls = []

    def fake_run(args, **kwargs):
        calls.append((list(args), kwargs.get("input")))
        # get namespace -> "not found" => returncode 1 triggers create
        if args[:3] == ["kubectl", "get", "namespace"]:
            return _err(returncode=1, stderr="not found")
        if args[:3] == ["kubectl", "create", "namespace"]:
            return _ok()
        if "create" in args and "secret" in args and "--dry-run=client" in args:
            return _ok(stdout="apiVersion: v1\nkind: Secret\n...")
        if args[:2] == ["kubectl", "apply"]:
            return _ok()
        return _ok()

    with patch("cogniverse_cli.secrets.subprocess.run", side_effect=fake_run):
        from cogniverse_cli.secrets import sync_hf_token_to_cluster

        assert sync_hf_token_to_cluster() is True

    create_ns = [c for c in calls if c[0][:3] == ["kubectl", "create", "namespace"]]
    assert len(create_ns) == 1

    render = [
        c
        for c in calls
        if "create" in c[0] and "secret" in c[0] and "--dry-run=client" in c[0]
    ]
    assert len(render) == 1
    # Token injected via --from-literal
    assert any("--from-literal=HF_TOKEN=the-token" in a for a in render[0][0])

    apply = [c for c in calls if c[0][:2] == ["kubectl", "apply"]]
    assert len(apply) == 1
    # apply consumed the rendered YAML via stdin
    assert apply[0][1] is not None and "kind: Secret" in apply[0][1]


def test_sync_returns_false_on_apply_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "the-token")
    monkeypatch.setattr(
        "cogniverse_cli.secrets.HF_CACHE_TOKEN_PATH", tmp_path / "unused"
    )

    def fake_run(args, **kwargs):
        if args[:3] == ["kubectl", "get", "namespace"]:
            return _ok()  # namespace exists
        if "create" in args and "secret" in args and "--dry-run=client" in args:
            return _ok(stdout="apiVersion: v1\nkind: Secret\n...")
        if args[:2] == ["kubectl", "apply"]:
            return _err(stderr="forbidden")
        return _ok()

    with patch("cogniverse_cli.secrets.subprocess.run", side_effect=fake_run):
        from cogniverse_cli.secrets import sync_hf_token_to_cluster

        assert sync_hf_token_to_cluster() is False


class TestKubectlBinaryGuard:
    def test_missing_kubectl_aborts_with_clear_message(self):
        import pytest
        from cogniverse_cli.secrets import _kubectl

        with patch(
            "cogniverse_cli.secrets.subprocess.run",
            side_effect=FileNotFoundError("kubectl"),
        ):
            with pytest.raises(SystemExit) as se:
                _kubectl(["get", "secret", "x"])
        assert "kubectl not found" in str(se.value)

    def test_hung_kubectl_aborts(self):
        import pytest
        from cogniverse_cli.secrets import _kubectl

        with patch(
            "cogniverse_cli.secrets.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="kubectl", timeout=30),
        ):
            with pytest.raises(SystemExit) as se:
                _kubectl(["get", "secret", "x"])
        assert "timed out" in str(se.value)


def test_inference_key_returns_false_when_missing_and_not_required(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)
    monkeypatch.setattr("cogniverse_cli.secrets.PROJECT_ENV", tmp_path / "absent")
    monkeypatch.setattr("cogniverse_cli.secrets.HOME_ENV", tmp_path / "gone")
    from cogniverse_cli.secrets import sync_inference_api_key_to_cluster

    assert sync_inference_api_key_to_cluster(required=False) is False


def test_inference_key_returns_false_when_missing_and_required(monkeypatch, tmp_path):
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)
    monkeypatch.setattr("cogniverse_cli.secrets.PROJECT_ENV", tmp_path / "absent")
    monkeypatch.setattr("cogniverse_cli.secrets.HOME_ENV", tmp_path / "gone")
    from cogniverse_cli.secrets import sync_inference_api_key_to_cluster

    # Still returns False, caller decides what to do with required.
    assert sync_inference_api_key_to_cluster(required=True) is False


def test_inference_key_env_wins_over_env_files(monkeypatch, tmp_path):
    env_dir = tmp_path / "project"
    env_dir.mkdir()
    (env_dir / "COGNIVERSE_INFERENCE_API_KEY.env").write_text("file-key-value")
    monkeypatch.setattr("cogniverse_cli.secrets.PROJECT_ENV", env_dir)
    monkeypatch.setattr("cogniverse_cli.secrets.HOME_ENV", tmp_path / "absent")
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "env-key-value")

    from cogniverse_cli.secrets import _read_inference_api_key

    assert _read_inference_api_key() == "env-key-value"


def test_inference_key_read_from_project_env_when_no_env_var(monkeypatch, tmp_path):
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)
    env_dir = tmp_path / "project"
    env_dir.mkdir()
    (env_dir / "COGNIVERSE_INFERENCE_API_KEY.env").write_text(
        "COGNIVERSE_INFERENCE_API_KEY=file-key-value\n"
    )
    monkeypatch.setattr("cogniverse_cli.secrets.PROJECT_ENV", env_dir)
    monkeypatch.setattr("cogniverse_cli.secrets.HOME_ENV", tmp_path / "absent")

    from cogniverse_cli.secrets import _read_inference_api_key

    assert _read_inference_api_key() == "file-key-value"


def test_inference_sync_creates_namespace_and_applies_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "the-key")

    calls = []

    def fake_run(args, **kwargs):
        calls.append((list(args), kwargs.get("input")))
        # get namespace -> "not found" => returncode 1 triggers create
        if args[:3] == ["kubectl", "get", "namespace"]:
            return _err(returncode=1, stderr="not found")
        if args[:3] == ["kubectl", "create", "namespace"]:
            return _ok()
        if "create" in args and "secret" in args and "--dry-run=client" in args:
            return _ok(stdout="apiVersion: v1\nkind: Secret\n...")
        if args[:2] == ["kubectl", "apply"]:
            return _ok()
        return _ok()

    with patch("cogniverse_cli.secrets.subprocess.run", side_effect=fake_run):
        from cogniverse_cli.secrets import sync_inference_api_key_to_cluster

        assert sync_inference_api_key_to_cluster() is True

    create_ns = [c for c in calls if c[0][:3] == ["kubectl", "create", "namespace"]]
    assert len(create_ns) == 1

    render = [
        c
        for c in calls
        if "create" in c[0] and "secret" in c[0] and "--dry-run=client" in c[0]
    ]
    assert len(render) == 1
    # The Secret the chart's secretKeyRef resolves: name + key + value
    assert "cogniverse-inference-api-key" in render[0][0]
    assert any(
        "--from-literal=COGNIVERSE_INFERENCE_API_KEY=the-key" in a for a in render[0][0]
    )

    apply = [c for c in calls if c[0][:2] == ["kubectl", "apply"]]
    assert len(apply) == 1
    # apply consumed the rendered YAML via stdin
    assert apply[0][1] is not None and "kind: Secret" in apply[0][1]


def test_inference_sync_returns_false_on_apply_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "the-key")

    def fake_run(args, **kwargs):
        if args[:3] == ["kubectl", "get", "namespace"]:
            return _ok()  # namespace exists
        if "create" in args and "secret" in args and "--dry-run=client" in args:
            return _ok(stdout="apiVersion: v1\nkind: Secret\n...")
        if args[:2] == ["kubectl", "apply"]:
            return _err(stderr="forbidden")
        return _ok()

    with patch("cogniverse_cli.secrets.subprocess.run", side_effect=fake_run):
        from cogniverse_cli.secrets import sync_inference_api_key_to_cluster

        assert sync_inference_api_key_to_cluster() is False
