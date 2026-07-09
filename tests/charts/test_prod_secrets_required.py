"""Production installs must not inherit the repo's dev secret defaults.

The MinIO root password and the OpenShell HMAC handshake secret shipped as
committed dev defaults and values.prod.yaml did not override them. They are now
emptied in the prod values and guarded by Helm `required`, so a prod install
fails loud until the operator supplies a real value.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART = REPO_ROOT / "charts" / "cogniverse"
PROD_VALUES = CHART / "values.prod.yaml"
K3S_VALUES = CHART / "values.k3s.yaml"
MINIO_TMPL = CHART / "templates" / "minio.yaml"


def _prod():
    return yaml.safe_load(PROD_VALUES.read_text())


@pytest.mark.unit
def test_prod_values_empty_the_dev_secrets():
    prod = _prod()
    assert prod["minio"]["rootPassword"] == "", "prod must not ship a minio password"
    assert prod["openshell"]["server"]["sshHandshakeSecret"] == "", (
        "prod must not ship the openshell handshake secret"
    )


@pytest.mark.unit
def test_minio_template_requires_password():
    assert "required" in MINIO_TMPL.read_text()
    assert ".Values.minio.rootPassword" in MINIO_TMPL.read_text()


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")
def test_prod_install_fails_loud_without_minio_password():
    result = subprocess.run(
        [
            "helm",
            "template",
            str(CHART),
            "-f",
            str(PROD_VALUES),
            # Prod also requires a redis password; give one so this isolates
            # the minio required check.
            "--set",
            "redis.auth.password=x",
            "--show-only",
            "templates/minio.yaml",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "minio.rootPassword must be set" in result.stderr


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")
def test_dev_install_still_renders_minio_secret():
    result = subprocess.run(
        [
            "helm",
            "template",
            str(CHART),
            "-f",
            str(K3S_VALUES),
            "--show-only",
            "templates/minio.yaml",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "rootPassword:" in result.stdout
