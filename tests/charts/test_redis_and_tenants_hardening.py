"""In-cluster Redis must support auth, and the base chart must not ship example
tenants.

Redis shipped with no authentication, so any pod reaching its Service could
read or flush the queue and session state. The base values also auto-provisioned
example tenants (acme_corp / globex_inc). Redis auth is now opt-in (off for dev,
required-with-secret in prod) and the example tenants are removed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART = REPO_ROOT / "charts" / "cogniverse"
VALUES = CHART / "values.yaml"
K3S = CHART / "values.k3s.yaml"
PROD = CHART / "values.prod.yaml"

_helm = pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")


def _values():
    return yaml.safe_load(VALUES.read_text())


def _render(*sets, values_file=K3S, show="templates/redis.yaml"):
    cmd = ["helm", "template", str(CHART), "-f", str(values_file), "--show-only", show]
    for s in sets:
        cmd += ["--set", s]
    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.mark.unit
def test_base_values_ship_no_example_tenants():
    tenants = _values()["config"]["tenants"]
    ids = {t.get("id") for t in tenants} if tenants else set()
    assert "acme_corp" not in ids and "globex_inc" not in ids, (
        f"base chart must not auto-provision example tenants; got {ids}"
    )


@pytest.mark.unit
def test_redis_auth_defaults_off():
    assert _values()["redis"]["auth"]["enabled"] is False


@pytest.mark.unit
@_helm
def test_dev_redis_has_no_requirepass():
    out = _render().stdout
    assert "--requirepass" not in out
    assert "REDIS_PASSWORD" not in out


@pytest.mark.unit
@_helm
def test_prod_redis_auth_fails_loud_without_password():
    result = _render(values_file=PROD)
    assert result.returncode != 0
    assert "redis.auth.password must be set" in result.stderr


@pytest.mark.unit
@_helm
def test_auth_on_puts_password_only_in_secret():
    out = _render("redis.auth.enabled=true", "redis.auth.password=s3cret").stdout
    # Server enforces auth via an env-expanded ref, not a literal.
    assert "--requirepass" in out
    assert '"$(REDIS_PASSWORD)"' in out
    # The literal password appears only in the Secret's stringData.
    assert out.count("s3cret") == 1
    assert "redis-password: " in out


@pytest.mark.unit
@_helm
def test_auth_on_consumer_url_uses_password_env():
    out = _render(
        "redis.auth.enabled=true",
        "redis.auth.password=s3cret",
        show="templates/ingestor.yaml",
    ).stdout
    assert "redis://:$(REDIS_PASSWORD)@" in out
    assert "s3cret" not in out  # password not baked into the consumer URL
