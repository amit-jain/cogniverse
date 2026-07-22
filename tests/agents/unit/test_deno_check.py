"""Unit tests for the Deno-availability probe used by RLMInference at boot."""

import os
from pathlib import Path

import pytest

from cogniverse_agents.inference.deno_check import (
    DenoNotInstalledError,
    assert_deno_available,
    is_deno_available,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class TestIsDenoAvailable:
    def test_returns_true_when_deno_on_path(self, tmp_path, monkeypatch):
        """A binary called 'deno' on PATH satisfies the probe."""
        # Layout a fake deno binary in tmp_path and put tmp_path first on PATH.
        deno = tmp_path / "deno"
        deno.write_text("#!/bin/sh\necho deno-stub")
        deno.chmod(0o755)
        monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}/usr/bin:/bin")

        assert is_deno_available() is True

    def test_returns_true_via_home_fallback_and_amends_path(
        self, tmp_path, monkeypatch
    ):
        """When PATH lacks deno but ~/.deno/bin/deno exists, return True and amend PATH."""
        fake_home = tmp_path / "home"
        deno_bin = fake_home / ".deno" / "bin"
        deno_bin.mkdir(parents=True)
        (deno_bin / "deno").write_text("#!/bin/sh\necho deno-stub")
        (deno_bin / "deno").chmod(0o755)
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(Path, "home", lambda: fake_home)
        monkeypatch.setenv("PATH", "/usr/bin:/bin")  # no deno on PATH

        assert is_deno_available() is True
        # PATH should now include the deno bin dir
        assert str(deno_bin) in os.environ["PATH"].split(os.pathsep)

    def test_returns_false_when_missing(self, tmp_path, monkeypatch):
        """Empty PATH and no ~/.deno -> False."""
        empty_home = tmp_path / "empty_home"
        empty_home.mkdir()
        monkeypatch.setenv("HOME", str(empty_home))
        monkeypatch.setattr(Path, "home", lambda: empty_home)
        monkeypatch.setenv("PATH", str(tmp_path))  # no deno here

        assert is_deno_available() is False


class TestAssertDenoAvailable:
    def test_raises_when_missing(self, tmp_path, monkeypatch):
        """Probe raises DenoNotInstalledError with actionable message."""
        empty_home = tmp_path / "empty_home"
        empty_home.mkdir()
        monkeypatch.setenv("HOME", str(empty_home))
        monkeypatch.setattr(Path, "home", lambda: empty_home)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.delenv("COGNIVERSE_RLM_SKIP_DENO_CHECK", raising=False)

        with pytest.raises(DenoNotInstalledError) as exc:
            assert_deno_available()

        message = str(exc.value)
        assert "Deno is required" in message
        assert "COGNIVERSE_RLM_SKIP_DENO_CHECK" in message  # escape-hatch surfaced

    def test_skip_env_var_bypasses(self, tmp_path, monkeypatch):
        """Setting COGNIVERSE_RLM_SKIP_DENO_CHECK=1 lets construction proceed."""
        empty_home = tmp_path / "empty_home"
        empty_home.mkdir()
        monkeypatch.setenv("HOME", str(empty_home))
        monkeypatch.setattr(Path, "home", lambda: empty_home)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setenv("COGNIVERSE_RLM_SKIP_DENO_CHECK", "1")

        # Must not raise.
        assert_deno_available()

    @pytest.mark.parametrize("value", ["true", "yes", "TRUE", "Yes"])
    def test_skip_env_var_accepts_truthy_values(self, value, tmp_path, monkeypatch):
        """Probe honours common truthy spellings, not just '1'."""
        empty_home = tmp_path / "empty_home"
        empty_home.mkdir()
        monkeypatch.setenv("HOME", str(empty_home))
        monkeypatch.setattr(Path, "home", lambda: empty_home)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setenv("COGNIVERSE_RLM_SKIP_DENO_CHECK", value)

        assert_deno_available()  # no raise

    def test_skip_env_var_falsy_values_dont_bypass(self, tmp_path, monkeypatch):
        """0 / empty / 'no' must not bypass the probe."""
        empty_home = tmp_path / "empty_home"
        empty_home.mkdir()
        monkeypatch.setenv("HOME", str(empty_home))
        monkeypatch.setattr(Path, "home", lambda: empty_home)
        monkeypatch.setenv("PATH", str(tmp_path))
        for falsy in ("0", "", "no", "false"):
            monkeypatch.setenv("COGNIVERSE_RLM_SKIP_DENO_CHECK", falsy)
            with pytest.raises(DenoNotInstalledError):
                assert_deno_available()
