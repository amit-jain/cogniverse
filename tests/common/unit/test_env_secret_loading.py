"""Every pytest run loads ``.env/`` so remote endpoints resolve.

Without these, ``ensure_llm`` finds no configured endpoint and silently
provisions a model container on the host instead of using the remote one.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

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


@pytest.mark.parametrize(
    "name",
    ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "COGNIVERSE_INFERENCE_API_KEY"],
)
def test_the_running_session_has_the_remote_inference_credentials(name):
    """conftest loads these at import, so a spawn decision sees the endpoints.

    Missing any one of these makes the model-list probe fail and ``ensure_llm``
    fall through to building a local sidecar.
    """
    if not any((d / f"{name}.env").is_file() for d in env_secret_dirs()):
        pytest.fail(
            f"{name}.env is absent from {[str(d) for d in env_secret_dirs()]}; "
            "every run falls back to a local spawn"
        )
    assert os.environ[name] != ""
