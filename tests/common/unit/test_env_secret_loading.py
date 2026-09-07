"""Every pytest run loads ``.env/`` so remote endpoints resolve.

Without these, ``ensure_llm`` finds no configured endpoint and silently
provisions a model container on the host instead of using the remote one.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from tests.env_secrets import env_secret_dirs, load_env_secrets


def _write(directory: Path, name: str, body: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.env").write_text(body)


def test_loads_a_bare_value(tmp_path, monkeypatch):
    _write(tmp_path / ".env", "SECRET_ALPHA", "ak-plain-value\n")
    monkeypatch.delenv("SECRET_ALPHA", raising=False)
    assert load_env_secrets(tmp_path) == {"SECRET_ALPHA": "ak-plain-value"}
    assert os.environ["SECRET_ALPHA"] == "ak-plain-value"


def test_strips_a_self_named_prefix(tmp_path, monkeypatch):
    _write(tmp_path / ".env", "SECRET_BETA", "SECRET_BETA=with-prefix\n")
    monkeypatch.delenv("SECRET_BETA", raising=False)
    assert load_env_secrets(tmp_path) == {"SECRET_BETA": "with-prefix"}


def test_skips_comments_and_blank_lines(tmp_path, monkeypatch):
    _write(tmp_path / ".env", "SECRET_GAMMA", "# a comment\n\n  \nreal-value\n")
    monkeypatch.delenv("SECRET_GAMMA", raising=False)
    assert load_env_secrets(tmp_path) == {"SECRET_GAMMA": "real-value"}


def test_an_explicitly_set_variable_wins(tmp_path, monkeypatch):
    _write(tmp_path / ".env", "SECRET_DELTA", "from-file\n")
    monkeypatch.setenv("SECRET_DELTA", "from-caller")
    assert load_env_secrets(tmp_path) == {}
    assert os.environ["SECRET_DELTA"] == "from-caller"


def test_an_empty_file_sets_nothing(tmp_path, monkeypatch):
    _write(tmp_path / ".env", "SECRET_EPSILON", "\n#only a comment\n")
    monkeypatch.delenv("SECRET_EPSILON", raising=False)
    assert load_env_secrets(tmp_path) == {}
    assert "SECRET_EPSILON" not in os.environ


def test_a_missing_directory_is_not_an_error(tmp_path):
    assert load_env_secrets(tmp_path / "nothing-here") == {}


REMOTE_INFERENCE_CREDENTIALS = (
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "COGNIVERSE_INFERENCE_API_KEY",
)


def test_remote_inference_credentials_reach_the_environment(tmp_path, monkeypatch):
    """The three names a spawn decision needs are set from ``.env/`` exactly.

    Missing any one of them makes the model-list probe fail and ``ensure_llm``
    fall through to building a local sidecar.
    """
    expected = {
        name: f"value-for-{name.lower()}" for name in REMOTE_INFERENCE_CREDENTIALS
    }
    for name, value in expected.items():
        _write(tmp_path / ".env", name, f"{value}\n")
        monkeypatch.delenv(name, raising=False)
    assert load_env_secrets(tmp_path) == expected
    assert {name: os.environ[name] for name in REMOTE_INFERENCE_CREDENTIALS} == expected


def test_a_worktree_resolves_the_owning_checkouts_secrets(tmp_path, monkeypatch):
    """``.env`` is untracked and lives only in the main checkout; a worktree
    must find it through the common git dir rather than read nothing."""
    main = tmp_path / "main"
    main.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(main)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(main),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "root",
        ],
        check=True,
    )
    worktree = tmp_path / "wt"
    subprocess.run(
        ["git", "-C", str(main), "worktree", "add", "-q", str(worktree)], check=True
    )
    _write(main / ".env", "MODAL_TOKEN_ID", "from-main-checkout\n")
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)

    assert env_secret_dirs(worktree) == [worktree / ".env", main / ".env"]
    assert load_env_secrets(worktree) == {"MODAL_TOKEN_ID": "from-main-checkout"}
    assert os.environ["MODAL_TOKEN_ID"] == "from-main-checkout"
