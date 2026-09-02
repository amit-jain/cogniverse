"""The dashboard entrypoint must configure the root logger.

Without this the root logger keeps its default WARNING level and no handler, so
every ``logger.info`` the dashboard emits is discarded: the pod's logs carry
3143 lines and zero INFO, an operator grepping them sees a server that logs
nothing, and a diagnosis that reads "this line never appeared" concludes the
code never ran when it only never logged.

The subprocess is the point: configuration only counts in a bare interpreter,
which is what ``streamlit run`` gives the app. Asserting it in-process would
pass on pytest's own root handler.
"""

import subprocess
import sys

PROBE = """
import logging
from cogniverse_dashboard import configure_dashboard_logging

configure_dashboard_logging()
logging.getLogger("cogniverse_dashboard.probe").info("hello-from-probe")
logging.getLogger("cogniverse_dashboard.probe").warning("warn-from-probe")
"""


def _run(env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    env.pop("LOG_LEVEL", None)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def test_info_reaches_the_stream_in_a_bare_interpreter():
    result = _run()
    assert result.returncode == 0, result.stderr
    # The exact formatted tail: logger name, level, message. Pins the format,
    # not merely that something was emitted.
    assert " - cogniverse_dashboard.probe - INFO - hello-from-probe" in result.stderr, (
        result.stderr
    )


def test_log_level_env_suppresses_info_but_keeps_warning():
    """The other direction: a gate that only ever passes proves nothing."""
    result = _run({"LOG_LEVEL": "WARNING"})
    assert result.returncode == 0, result.stderr
    assert "hello-from-probe" not in result.stderr, result.stderr
    assert (
        " - cogniverse_dashboard.probe - WARNING - warn-from-probe" in result.stderr
    ), result.stderr


def test_unknown_log_level_is_rejected_loudly():
    result = _run({"LOG_LEVEL": "CHATTY"})
    assert result.returncode == 1, result.stdout + result.stderr
    assert "LOG_LEVEL='CHATTY' is not a logging level name" in result.stderr


def test_existing_root_handler_is_left_in_charge():
    """pytest / an embedding host owns the root logger when it already set one."""
    probe = """
import logging, sys
logging.basicConfig(level=logging.ERROR, format="PREEXISTING %(message)s")
from cogniverse_dashboard import configure_dashboard_logging
configure_dashboard_logging()
logging.getLogger("cogniverse_dashboard.probe").error("owned")
"""
    import os

    env = dict(os.environ)
    env.pop("LOG_LEVEL", None)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "PREEXISTING owned" in result.stderr, result.stderr
