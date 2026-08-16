"""The runtime server routes application ``logging`` to a real handler.

Uvicorn configures only its own loggers; without a root handler every INFO
record from the ``cogniverse_*`` loggers is dropped and WARNING+ reaches
stderr through ``logging.lastResort`` with no timestamp or logger name.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from contextlib import contextmanager
from typing import Iterator

import pytest

from cogniverse_runtime import main as runtime_main

RUNTIME_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@contextmanager
def _bare_root_logger() -> Iterator[logging.Logger]:
    """Strip the root logger for the duration of a test and restore it after."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    for handler in saved_handlers:
        root.removeHandler(handler)
    try:
        yield root
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)


def _stream_handler_formats(root: logging.Logger) -> list[str]:
    return [
        handler.formatter._fmt
        for handler in root.handlers
        if type(handler) is logging.StreamHandler and handler.formatter is not None
    ]


class TestConfigureRuntimeLogging:
    def test_bare_root_gets_one_info_stream_handler(self, monkeypatch):
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        with _bare_root_logger() as root:
            runtime_main._configure_runtime_logging()
            assert root.level == logging.INFO
            assert _stream_handler_formats(root) == [RUNTIME_LOG_FORMAT]
            # Idempotent: a second call adds nothing.
            runtime_main._configure_runtime_logging()
            assert _stream_handler_formats(root) == [RUNTIME_LOG_FORMAT]

    def test_log_level_env_selects_the_root_level(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "debug")
        with _bare_root_logger() as root:
            runtime_main._configure_runtime_logging()
            assert root.level == logging.DEBUG

    def test_unknown_log_level_raises(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "LOUD")
        with _bare_root_logger():
            with pytest.raises(ValueError) as raised:
                runtime_main._configure_runtime_logging()
        assert str(raised.value) == "LOG_LEVEL='LOUD' is not a logging level name"

    def test_existing_root_handler_is_left_in_charge(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        with _bare_root_logger() as root:
            root.setLevel(logging.WARNING)
            existing = logging.NullHandler()
            root.addHandler(existing)
            runtime_main._configure_runtime_logging()
            assert root.handlers == [existing]
            assert root.level == logging.WARNING


def test_importing_the_server_module_configures_logging():
    """The uvicorn entrypoint (``cogniverse_runtime.main``) configures logging
    on import, so INFO records reach the pod log without any other setup."""
    probe = (
        "import logging, sys\n"
        "import cogniverse_runtime.main\n"
        "root = logging.getLogger()\n"
        "fmts = [h.formatter._fmt for h in root.handlers"
        " if type(h) is logging.StreamHandler and h.formatter is not None]\n"
        "sys.stdout.write(f'{logging.getLevelName(root.level)}|{fmts!r}\\n')\n"
    )
    env = {**os.environ, "LOG_LEVEL": "DEBUG"}
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == (f"DEBUG|{[RUNTIME_LOG_FORMAT]!r}")
