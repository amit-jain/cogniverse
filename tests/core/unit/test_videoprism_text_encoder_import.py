"""Importing the VideoPrism text encoder must not trigger config I/O.

The module resolved videoprism_repo_path via get_config at import time, which
raises when BACKEND_URL is unset — so importing the module (in tests, tooling,
or a dev shell without the env) failed, and the loader's mismatched
except ImportError missed the ValueError, silently disabling LVT text encoding.
Config resolution is now deferred to first use.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.unit
@pytest.mark.ci_fast
def test_module_imports_without_backend_url():
    code = (
        "import os\n"
        "os.environ.pop('BACKEND_URL', None)\n"
        "os.environ.pop('BACKEND_PORT', None)\n"
        "import cogniverse_core.common.models.videoprism_text_encoder as m\n"
        # Config resolution is deferred, so no load has been attempted at import.
        "assert m._VIDEOPRISM_LOAD_ATTEMPTED is False\n"
        "print('IMPORT_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "IMPORT_OK" in result.stdout


@pytest.mark.unit
@pytest.mark.ci_fast
def test_ensure_videoprism_survives_config_failure():
    """A config resolution failure at first-use must not crash — it logs and
    proceeds to attempt the import (which itself may be unavailable)."""
    from unittest.mock import patch

    import cogniverse_core.common.models.videoprism_text_encoder as m

    m._VIDEOPRISM_LOAD_ATTEMPTED = False
    m.vp = None
    with patch(
        "cogniverse_foundation.config.utils.create_default_config_manager",
        side_effect=ValueError("BACKEND_URL environment variable is required"),
    ) as build_cm:
        # Does not raise despite the config failure; return tracks vp state.
        result = m._ensure_videoprism()
        assert isinstance(result, bool)
        assert result is (m.vp is not None)
        assert m._VIDEOPRISM_LOAD_ATTEMPTED is True

        # Cached: a second call short-circuits without re-resolving config.
        calls_before = build_cm.call_count
        assert m._ensure_videoprism() is result
        assert build_cm.call_count == calls_before
    m._VIDEOPRISM_LOAD_ATTEMPTED = False
    m.vp = None
