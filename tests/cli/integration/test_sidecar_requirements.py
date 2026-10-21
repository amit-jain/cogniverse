"""Each sidecar image installs exactly ``deploy/<name>/requirements.txt`` and
serves its model from that environment alone. These tests install that file
into an isolated interpreter and drive the model-load preconditions the
server's readiness probe depends on, so a missing runtime dependency fails
here instead of as a pod that never becomes Ready.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[3]

# The processor class each sidecar's model load builds for its pinned model
# (``ClapProcessor`` wraps ``ClapFeatureExtractor`` for the audio it is sent);
# constructing it with defaults runs the same backend checks as the real load
# without fetching the checkpoint.
_SIDECAR_PROCESSOR_PROBES = {
    "clap_embed": textwrap.dedent(
        """
        import librosa
        import transformers
        extractor = transformers.ClapFeatureExtractor()
        print(type(extractor).__name__, extractor.sampling_rate)
        """
    ),
}

_EXPECTED_PROBE_OUTPUT = {
    "clap_embed": "ClapFeatureExtractor 48000",
}


def _run_in_sidecar_env(sidecar: str, code: str) -> subprocess.CompletedProcess:
    requirements = _REPO_ROOT / "deploy" / sidecar / "requirements.txt"
    assert requirements.is_file(), requirements
    return subprocess.run(
        [
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--with-requirements",
            str(requirements),
            "python",
            "-c",
            code,
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.integration
@pytest.mark.parametrize("sidecar", sorted(_SIDECAR_PROCESSOR_PROBES))
def test_sidecar_requirements_satisfy_the_model_processor(sidecar: str) -> None:
    probe = _run_in_sidecar_env(sidecar, _SIDECAR_PROCESSOR_PROBES[sidecar])
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert probe.stdout.strip().splitlines()[-1] == _EXPECTED_PROBE_OUTPUT[sidecar]
