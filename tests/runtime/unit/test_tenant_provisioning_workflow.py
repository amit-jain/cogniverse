"""The tenant-provisioning Argo template deploys and verifies a Vespa schema
per profile, from inside the runtime image.

The image ships the installed packages, ``configs/`` and nothing else: no
``uv``, no ``scripts/`` and no ``kubectl``. Every step therefore runs
``python -m cogniverse_runtime.provision_tenant``, whose schema step goes
through ``SchemaRegistry`` — the seam ``POST /admin/profiles/{name}/deploy``
uses — rather than posting a one-schema application package of its own.

These pin the step commands, their Vespa environment, and the
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


ENTRYPOINT = ["python", "-m", "cogniverse_runtime.provision_tenant"]
VESPA_ENV = {
    "BACKEND_URL": "http://cogniverse-vespa",
    "BACKEND_PORT": "8080",
    "VESPA_CONFIG_PORT": "19071",
}


def _templates() -> dict:
    (document,) = [
        d
        for d in yaml.safe_load_all(WORKFLOW.read_text())
        if d["kind"] == "WorkflowTemplate"
    ]
    return {t["name"]: t for t in document["spec"]["templates"]}


@pytest.mark.unit
def test_every_runtime_step_runs_the_installed_entrypoint():
    templates = _templates()
    for name in (
        "deploy-schemas",
        "create-phoenix-project",
        "initialize-memory",
        "set-tier",
        "verify-tenant",
    ):
        assert templates[name]["container"]["command"] == ENTRYPOINT, name
    # The runtime image has no uv, no scripts/ and no kubectl, so a step
    # naming any of them fails before it reaches Vespa.
    text = WORKFLOW.read_text(encoding="utf-8")
    for absent in ("uv run", "scripts/", "kubectl "):
        assert absent not in text, absent


@pytest.mark.unit
def test_schema_steps_name_the_profiles_and_the_vespa_endpoint():
    templates = _templates()
    for name, step in (("deploy-schemas", "schemas"), ("verify-tenant", "verify")):
        container = templates[name]["container"]
        assert container["args"] == [
            "--step",
            step,
            "--tenant-id",
            "{{workflow.parameters.tenant-id}}",
            "--profiles",
            "{{workflow.parameters.profiles}}",
        ], name
        # The step reads the data endpoint the rest of the stack reads and
        # the config-server port the deploy posts to; the old VESPA_URL was
        # read by nothing.
        assert {
            entry["name"]: entry["value"] for entry in container["env"]
        } == VESPA_ENV, name


@pytest.mark.unit
def test_namespace_and_claim_are_verified_through_argo_resource_reads():
    templates = _templates()
    assert templates["verify-namespace"]["resource"]["action"] == "get"
    assert "kind: Namespace" in templates["verify-namespace"]["resource"]["manifest"]
    assert templates["verify-storage"]["resource"]["action"] == "get"
    assert (
        "kind: PersistentVolumeClaim"
        in templates["verify-storage"]["resource"]["manifest"]
    )


@pytest.mark.unit
def test_schema_deployment_resolves_the_profiles_schema_name():
    """The module deploys ``profile.schema_name``, not the profile name.

    ``audio_clap_semantic`` deploys ``audio_content``; a step that passed the
    profile name through would register a schema no reader queries.
    """
    from cogniverse_runtime import provision_tenant

    calls = []

    class _Registry:
        def deploy_schema(self, *, tenant_id, base_schema_name):
            calls.append((tenant_id, base_schema_name))
            return f"{base_schema_name}_{tenant_id.replace(':', '_')}"

    class _Backend:
        schema_registry = _Registry()

    profile_names = ["audio_clap_semantic", "video_colpali_smol500_mv_frame"]
    expected = [resolve_profile_schema(name, CONFIG) for name in profile_names]
    original = provision_tenant._resolve
    provision_tenant._resolve = lambda tenant_id, profiles: (
        None,
        _Backend(),
        [resolve_profile_schema(name, CONFIG) for name in profiles],
    )
    try:
        deployed = provision_tenant.deploy_schemas("acme", profile_names)
    finally:
        provision_tenant._resolve = original

    assert calls == [("acme:acme", schema) for schema in expected]
    assert deployed == [f"{schema}_acme_acme" for schema in expected]


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
    parameters = {
        entry["name"]: entry["value"] for entry in template["arguments"]["parameters"]
    }
    return {
        "steps": steps,
        "tier_invocation": tier_step["container"]["command"]
        + tier_step["container"]["args"],
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
        "verify-namespace",
        "verify-storage",
        "verify-tenant",
        "notify-completion",
    ]
    assert shape["tier_invocation"] == ENTRYPOINT + [
        "--step",
        "tier",
        "--tier",
        "{{workflow.parameters.tier}}",
        "--tenant-id",
        "{{workflow.parameters.tenant-id}}",
    ]
    assert shape["tier_default"] == "default"
