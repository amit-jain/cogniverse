"""The tenant-provisioning Argo template deploys and verifies a Vespa schema
per profile, from inside the runtime image.

The image ships the installed packages, ``configs/`` and nothing else: no
``uv``, no ``scripts/`` and no ``kubectl``. Every step therefore runs
``python -m cogniverse_runtime.provision_tenant``, whose schema step goes
through ``SchemaRegistry`` — the seam ``POST /admin/profiles/{name}/deploy``
uses — rather than posting a one-schema application package of its own.

These pin the step commands, the environment each step needs to reach its
own work, and the profile→schema_name→file chain against the real files on
disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / "workflows" / "tenant-provisioning.yaml"
SCHEMAS_DIR = REPO_ROOT / "configs" / "schemas"
CONFIG = REPO_ROOT / "configs" / "config.json"

ENTRYPOINT = ["python", "-m", "cogniverse_runtime.provision_tenant"]
VESPA_ENV = {
    "BACKEND_URL": "http://cogniverse-vespa",
    "BACKEND_PORT": "8080",
    "VESPA_CONFIG_PORT": "19071",
}
TELEMETRY_ENV = {
    "TELEMETRY_OTLP_ENDPOINT": "cogniverse-phoenix:4317",
    "TELEMETRY_HTTP_ENDPOINT": "http://cogniverse-phoenix:6006",
}
RUNTIME_STEP_ENV = {
    "deploy-schemas": VESPA_ENV,
    "create-phoenix-project": VESPA_ENV | TELEMETRY_ENV,
    "initialize-memory": VESPA_ENV,
    "set-tier": VESPA_ENV,
    "verify-tenant": VESPA_ENV,
}


def _shipped_profiles() -> dict:
    """The cluster-wide profile catalog the provisioning steps resolve from."""
    return json.loads(CONFIG.read_text()).get("backend", {}).get("profiles", {})


def _shipped_schema_name(profile_name: str) -> str:
    return _shipped_profiles()[profile_name]["schema_name"]


class _TenantWithoutRows:
    """A ConfigManager for a tenant that owns no stored backend config."""

    def __init__(self) -> None:
        self.reads: list[str] = []

    def get_backend_config(self, tenant_id, service="backend"):
        from cogniverse_foundation.config.unified_config import BackendConfig

        self.reads.append(tenant_id)
        return BackendConfig(tenant_id=tenant_id)

    def get_system_config(self):
        return object()

    def get_routing_config(self, tenant_id):
        return object()

    def get_telemetry_config(self, tenant_id):
        return object()


def _templates() -> dict:
    (document,) = [
        d
        for d in yaml.safe_load_all(WORKFLOW.read_text())
        if d["kind"] == "WorkflowTemplate"
    ]
    return {t["name"]: t for t in document["spec"]["templates"]}


def _step_env(name: str) -> dict:
    return {
        entry["name"]: entry["value"]
        for entry in _templates()[name]["container"]["env"]
    }


@pytest.mark.unit
def test_every_runtime_step_runs_the_installed_entrypoint():
    templates = _templates()
    for name in RUNTIME_STEP_ENV:
        assert templates[name]["container"]["command"] == ENTRYPOINT, name
    # The runtime image has no uv, no scripts/ and no kubectl, so a step
    # naming any of them fails before it reaches Vespa.
    text = WORKFLOW.read_text(encoding="utf-8")
    for absent in ("uv run", "scripts/", "kubectl "):
        assert absent not in text, absent


@pytest.mark.unit
def test_every_runtime_step_declares_exactly_its_environment():
    """The telemetry step needs the Vespa endpoint too: its telemetry manager
    is built from the config store, which is reached through ``BACKEND_URL``."""
    assert {name: _step_env(name) for name in RUNTIME_STEP_ENV} == RUNTIME_STEP_ENV


@pytest.mark.unit
def test_every_runtime_step_can_bootstrap_its_config_store():
    """Run the real bootstrap under each step's declared environment and
    nothing else. A step missing ``BACKEND_URL`` raises here, which is what
    every step does the moment it builds a ConfigManager."""
    from cogniverse_foundation.config.bootstrap import BootstrapConfig

    resolved = {}
    for name in RUNTIME_STEP_ENV:
        with patch.dict(os.environ, _step_env(name), clear=True):
            bootstrap = BootstrapConfig.from_environment(config_path=CONFIG)
        resolved[name] = (
            bootstrap.backend_type,
            bootstrap.backend_url,
            bootstrap.backend_port,
        )
    assert resolved == {
        name: ("vespa", "http://cogniverse-vespa", 8080) for name in RUNTIME_STEP_ENV
    }


@pytest.mark.unit
def test_schema_steps_name_the_profiles_parameter():
    templates = _templates()
    for name, step in (("deploy-schemas", "schemas"), ("verify-tenant", "verify")):
        assert templates[name]["container"]["args"] == [
            "--step",
            step,
            "--tenant-id",
            "{{workflow.parameters.tenant-id}}",
            "--profiles",
            "{{workflow.parameters.profiles}}",
        ], name


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
def test_schema_deployment_sends_every_profiles_schema_as_one_package():
    """The module deploys ``profile.schema_name``, not the profile name, and
    deploys all of them in a single application package.

    ``audio_clap_semantic`` deploys ``audio_content``; a step that passed the
    profile name through would register a schema no reader queries. One
    package per run means a profile that fails leaves none of them registered
    instead of a half-provisioned tenant.
    """
    from cogniverse_runtime import provision_tenant

    calls = []

    class _Registry:
        def deploy_schemas(self, *, tenant_id, base_schema_names):
            calls.append((tenant_id, list(base_schema_names)))
            return [
                f"{base}_{tenant_id.replace(':', '_')}" for base in base_schema_names
            ]

    class _Backend:
        schema_registry = _Registry()

    profile_names = ["audio_clap_semantic", "video_colpali_smol500_mv_frame"]
    expected = [_shipped_schema_name(name) for name in profile_names]
    original = provision_tenant._resolve
    provision_tenant._resolve = lambda tenant_id, profiles: (
        None,
        _Backend(),
        [_shipped_schema_name(name) for name in profiles],
    )
    try:
        deployed = provision_tenant.deploy_schemas("acme", profile_names)
    finally:
        provision_tenant._resolve = original

    assert calls == [("acme:acme", expected)]
    assert deployed == [f"{schema}_acme_acme" for schema in expected]


@pytest.mark.unit
def test_profiles_resolve_from_the_shipped_catalog_when_the_tenant_has_no_rows():
    """A tenant registered a minute ago owns no backend rows, so provisioning
    resolves its profiles from the cluster catalog in ``config.json`` merged
    with whatever the tenant has overridden."""
    from cogniverse_runtime import provision_tenant

    profile_names = ["audio_clap_semantic", "video_colpali_smol500_mv_frame"]
    manager = _TenantWithoutRows()

    with patch.dict(os.environ, {"COGNIVERSE_CONFIG": str(CONFIG)}):
        resolved = provision_tenant.resolve_schema_names(
            manager, "acme:prod", profile_names
        )
    assert resolved == [_shipped_schema_name(name) for name in profile_names]
    assert manager.reads == ["acme:prod"]


@pytest.mark.unit
def test_an_unknown_profile_names_the_catalog_it_was_looked_up_in():
    from cogniverse_runtime import provision_tenant

    with patch.dict(os.environ, {"COGNIVERSE_CONFIG": str(CONFIG)}):
        with pytest.raises(RuntimeError) as raised:
            provision_tenant.resolve_schema_names(
                _TenantWithoutRows(), "acme:prod", ["no_such_profile"]
            )
    assert str(raised.value) == (
        "Provisioning failed for tenant acme:prod: profile 'no_such_profile' "
        f"is not configured. Configured profiles: {sorted(_shipped_profiles())}"
    )


@pytest.mark.unit
def test_every_default_profile_maps_to_an_existing_schema_file():
    """The whole point: each default profile resolves to a schema file that
    actually exists — the deploy step won't no-op on a missing path."""
    profiles = _shipped_profiles()
    assert profiles, "default config must ship backend profiles"

    missing = []
    for name in profiles:
        schema_name = _shipped_schema_name(name)
        if not (SCHEMAS_DIR / f"{schema_name}_schema.json").exists():
            missing.append((name, schema_name))
    assert not missing, f"profiles whose schema file is missing: {missing}"


@pytest.mark.unit
def test_the_workflows_default_profiles_are_shipped_profiles():
    """The template's ``profiles`` default must name profiles the catalog
    carries, or every run of the shipped defaults fails at the schema step."""
    (document,) = [
        d
        for d in yaml.safe_load_all(WORKFLOW.read_text())
        if d["kind"] == "WorkflowTemplate"
    ]
    parameters = {
        entry["name"]: entry["value"]
        for entry in document["spec"]["arguments"]["parameters"]
    }
    defaults = parameters["profiles"].split(",")
    assert defaults == ["video_colpali_smol500_mv_frame", "video_xclip_sv_chunk_6s"]
    assert [_shipped_schema_name(name) for name in defaults] == defaults


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
