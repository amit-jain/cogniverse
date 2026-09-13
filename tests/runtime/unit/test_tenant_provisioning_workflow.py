"""The tenant-provisioning Argo template deploys and verifies a Vespa schema
per profile. Two bugs lived here:

* The deploy step referenced ``configs/schemas/<profile>.json`` while the files
  are named ``<schema_name>_schema.json`` — ``deploy_json_schema.py`` failed on
  the missing path while the loop still reported success, so a provisioned
  tenant got no schemas.
* The schema file/id are named after the profile's ``schema_name``, which is
  not always the profile name (``audio_clap_semantic`` -> ``audio_content``),
  so even the ``_schema.json`` suffix wasn't enough — the step must resolve
  schema_name from config.

These pin the template's schema-path expression, the resolver, and the
profile→schema_name→file chain against the real files on disk.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / "workflows" / "tenant-provisioning.yaml"
SCHEMAS_DIR = REPO_ROOT / "configs" / "schemas"
CONFIG = REPO_ROOT / "configs" / "config.json"
RESOLVER = REPO_ROOT / "scripts" / "resolve_profile_schema.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from resolve_profile_schema import resolve_profile_schema  # noqa: E402


def _default_profiles() -> dict:
    return json.loads(CONFIG.read_text()).get("backend", {}).get("profiles", {})


@pytest.mark.unit
def test_template_resolves_schema_name_not_profile_name():
    text = WORKFLOW.read_text(encoding="utf-8")
    # Deploy + verify must resolve schema_name via the resolver, then use the
    # ``<schema_name>_schema.json`` file. The bare ``<profile>.json`` and the
    # unresolved ``<profile>_schema.json`` forms must be gone.
    assert "resolve_profile_schema.py" in text
    assert "configs/schemas/${schema_name}_schema.json" in text
    assert 'configs/schemas/${profile}.json"' not in text
    assert 'configs/schemas/${profile}_schema.json"' not in text


@pytest.mark.unit
def test_resolver_maps_every_default_profile_to_an_existing_schema_file():
    """The whole point: each default profile resolves to a schema file that
    actually exists — the deploy step won't no-op on a missing path."""
    profiles = _default_profiles()
    assert profiles, "default config must ship backend profiles"

    missing = []
    for name in profiles:
        schema_name = resolve_profile_schema(name, CONFIG)
        if not (SCHEMAS_DIR / f"{schema_name}_schema.json").exists():
            missing.append((name, schema_name))
    assert not missing, f"profiles whose schema file is missing: {missing}"


@pytest.mark.unit
def test_resolver_handles_name_schema_mismatch():
    # audio_clap_semantic's schema is audio_content — the exact case the
    # profile-name form got wrong.
    assert resolve_profile_schema("audio_clap_semantic", CONFIG) == "audio_content"


@pytest.mark.unit
def test_resolver_cli_prints_schema_name_and_fails_on_unknown():
    ok = subprocess.run(
        [sys.executable, str(RESOLVER), "audio_clap_semantic", str(CONFIG)],
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0
    assert ok.stdout.strip() == "audio_content"

    bad = subprocess.run(
        [sys.executable, str(RESOLVER), "no_such_profile", str(CONFIG)],
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 1
    # Nothing printed to stdout on failure — command substitution must not get
    # a bogus schema name.
    assert bad.stdout.strip() == ""


def _provisioning_shape(document: dict) -> dict:
    """What the template runs, in order, and how its tier step is wired."""
    template = document["spec"]
    (pipeline,) = [
        t for t in template["templates"] if t["name"] == "provisioning-pipeline"
    ]
    steps = [step["template"] for group in pipeline["steps"] for step in group]
    (tier_step,) = [t for t in template["templates"] if t["name"] == "set-tier"]
    (script,) = tier_step["container"]["args"]
    parameters = {
        entry["name"]: entry["value"] for entry in template["arguments"]["parameters"]
    }
    return {
        "steps": steps,
        "tier_invocation": " ".join(script.replace("\\\n", " ").split()),
        "tier_default": parameters["tier"],
    }


@pytest.mark.unit
def test_provisioning_sets_the_router_tier_before_verifying():
    (document,) = [
        d
        for d in yaml.safe_load_all(WORKFLOW.read_text())
        if d["kind"] == "WorkflowTemplate"
    ]
    shape = _provisioning_shape(document)
    assert shape["steps"] == [
        "validate-tenant",
        "create-namespace",
        "deploy-schemas",
        "create-phoenix-project",
        "setup-resource-quotas",
        "create-storage",
        "initialize-memory",
        "set-tier",
        "verify-tenant",
        "notify-completion",
    ]
    assert (
        "scripts/provision_tenant.py --step tier "
        '--tier "{{workflow.parameters.tier}}" '
        '--tenant-id "{{workflow.parameters.tenant-id}}"'
    ) in shape["tier_invocation"]
    assert shape["tier_default"] == "default"
