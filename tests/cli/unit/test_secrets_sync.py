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
