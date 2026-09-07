"""Multi-schema tenants, and teardown of a fixture that fails after minting.

Both run against the deployed cluster. Each mints its own eight-hex org, so
every assertion is scoped to schemas and rows this module created; peer tenants
come and go around it.

The one-activation property of a multi-schema create is not observable from
here — the config server's generation counter is cluster-wide and the seam that
performs the activation runs inside the runtime pod. That claim is pinned in
tests/core/integration/test_schema_intent_recovery.py; what this module pins is
the outcome: every requested schema live, each one the shipped definition under
the tenant's suffix, and all of them gone at teardown.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
import pytest

from cogniverse_vespa.json_schema_parser import JsonSchemaParser
from tests.e2e.conftest import (
    _VESPA_SCHEMAS_LIST_URL,
    RUNTIME,
    TENANT_DEPLOY_TIMEOUT_S,
    _deployed_schema_names_strict,
    _tenant_schema_name,
    own_tenant,
    register_tenant_and_wait,
    unique_id,
)

MULTI_BASES = ("wiki_pages", "provenance", "agent_memories")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_DIR = _REPO_ROOT / "configs" / "schemas"

# The inner session creates a tenant and deletes it; both are bounded by the
# helpers' own caps. The startup allowance covers importing tests.e2e.conftest
# in a fresh interpreter, measured at 7.5 s on the e2e host.
_REGISTER_TIMEOUT_S = float(
    inspect.signature(register_tenant_and_wait).parameters["timeout_s"].default
)
_INNER_STARTUP_BUDGET_S = 60.0
_INNER_SESSION_TIMEOUT_S = (
    _REGISTER_TIMEOUT_S + TENANT_DEPLOY_TIMEOUT_S + _INNER_STARTUP_BUDGET_S
)


def _shipped(base: str) -> dict:
    return json.loads((_SCHEMA_DIR / f"{base}_schema.json").read_text())


def _rendered_sd(base: str, deployed_name: str) -> str:
    """The .sd the shipped definition for ``base`` renders to as ``deployed_name``.

    The deploy path is SchemaRegistry.deploy_schemas setting ``name`` on the
    shipped JSON (registries/schema_registry.py:407) and the Vespa backend
    handing that to JsonSchemaParser (vespa/backend.py:869); this renders
    through the same two calls, so the comparison is whole text rather than a
    restatement of the fields a reader thought to list.
    """
    definition = _shipped(base)
    definition["name"] = deployed_name
    return JsonSchemaParser().parse_schema(definition).schema_to_text


def _live_sd(deployed_name: str) -> str:
    resp = httpx.get(f"{_VESPA_SCHEMAS_LIST_URL}{deployed_name}.sd", timeout=30.0)
    assert resp.status_code == 200, (
        f"{deployed_name}.sd: HTTP {resp.status_code} {resp.text[:400]}"
    )
    return resp.text


class TestMultiSchemaTenant:
    """A tenant minted with three base schemas gets exactly those three."""

    @pytest.fixture(scope="class")
    def tenant(self) -> dict:
        org = unique_id("mschema")
        tenant_id = f"{org}:multi"
        row = register_tenant_and_wait(
            tenant_id,
            created_by="e2e-multi-schema",
            base_schemas=list(MULTI_BASES),
        )
        return {"org": org, "tenant_id": tenant_id, "row": row}

    def test_requested_bases_name_the_shipped_definitions(self):
        assert {base: _shipped(base)["name"] for base in MULTI_BASES} == {
            base: base for base in MULTI_BASES
        }

    def test_persisted_row_records_every_requested_base(self, tenant):
        row = tenant["row"]
        assert (
            row["tenant_full_id"],
            row["org_id"],
            row["tenant_name"],
            row["status"],
            row["created_by"],
            row["schemas_deployed"],
        ) == (
            tenant["tenant_id"],
            tenant["org"],
            "multi",
            "active",
            "e2e-multi-schema",
            list(MULTI_BASES),
        )

    def test_vespa_deploys_exactly_the_requested_schemas_for_this_org(self, tenant):
        org = tenant["org"]
        assert {
            name for name in _deployed_schema_names_strict() if f"_{org}_" in name
        } == {_tenant_schema_name(base, tenant["tenant_id"]) for base in MULTI_BASES}

    def test_each_deployed_schema_is_its_shipped_definition(self, tenant):
        names = {
            base: _tenant_schema_name(base, tenant["tenant_id"]) for base in MULTI_BASES
        }
        assert {base: _live_sd(name) for base, name in names.items()} == {
            base: _rendered_sd(base, name) for base, name in names.items()
        }


_INNER_CONFTEST = """
from tests.e2e import conftest as e2e_conftest

pytest_fixture_setup = e2e_conftest.pytest_fixture_setup
pytest_runtest_call = e2e_conftest.pytest_runtest_call
"""

_INNER_MODULE = """
import json
from pathlib import Path

import pytest

from tests.e2e.conftest import register_tenant_and_wait, unique_id

RECORD = Path(__file__).parent / "minted.json"


@pytest.fixture(scope="module")
def tenant_then_failure():
    org = unique_id("teardown")
    tenant_id = f"{org}:t1"
    row = register_tenant_and_wait(tenant_id, created_by="e2e-teardown-contract")
    RECORD.write_text(json.dumps({"org": org, "tenant_id": tenant_id, "row": row}))
    raise RuntimeError("setup failed after the mint")


def test_body_never_runs(tenant_then_failure):
    raise AssertionError("the fixture fails before the body")
"""


def test_a_fixture_that_fails_after_minting_still_deletes_its_tenant(tmp_path):
    """The mint happens, the setup then fails, and the tenant is gone anyway.

    The failure has to be real for the guarantee to be under test, so it runs
    in its own pytest session as a subprocess: that session reports an error,
    this one reads its report and the live cluster.
    """
    session = tmp_path / "inner"
    session.mkdir()
    (session / "conftest.py").write_text(_INNER_CONFTEST)
    (session / "test_inner.py").write_text(_INNER_MODULE)
    # Its own config, so the inner session's rootdir is this directory and the
    # repo's addopts and testpaths do not reach it.
    (session / "pytest.ini").write_text("[pytest]\n")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(session),
            "-c",
            str(session / "pytest.ini"),
            "-p",
            "no:cacheprovider",
            "-q",
            "--tb=long",
            f"--junitxml={session / 'report.xml'}",
        ],
        cwd=_REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=_INNER_SESSION_TIMEOUT_S,
        check=False,
    )

    record = session / "minted.json"
    assert record.is_file(), (
        f"the inner session never minted a tenant, so nothing was torn down:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    minted = json.loads(record.read_text())
    org, tenant_id, row = minted["org"], minted["tenant_id"], minted["row"]
    own_tenant(org)

    root = ET.parse(session / "report.xml").getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    case = suite.find("testcase")
    assert (
        proc.returncode,
        suite.get("tests"),
        suite.get("errors"),
        suite.get("failures"),
        suite.get("skipped"),
        case.get("classname"),
        case.get("name"),
        case.find("error").get("message"),
    ) == (
        1,
        "1",
        "1",
        "0",
        "0",
        "test_inner",
        "test_body_never_runs",
        'failed on setup with "RuntimeError: setup failed after the mint"',
    ), proc.stdout

    assert re.fullmatch(r"teardown_[0-9a-f]{8}", org), org
    assert (
        row["tenant_full_id"],
        row["org_id"],
        row["status"],
        row["created_by"],
    ) == (tenant_id, org, "active", "e2e-teardown-contract")

    with httpx.Client(timeout=TENANT_DEPLOY_TIMEOUT_S) as client:
        tenant_get = client.get(f"{RUNTIME}/admin/tenants/{tenant_id}")
        org_get = client.get(f"{RUNTIME}/admin/organizations/{org}")
    leftover = {name for name in _deployed_schema_names_strict() if f"_{org}_" in name}
    assert (tenant_get.status_code, org_get.status_code, leftover) == (404, 404, set())
