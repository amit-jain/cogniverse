"""The rendered schema-deployment hook against the runtime's admin router.

The hook deploys ``config.defaultProfiles.video`` for each ``config.tenants``
entry. The runtime here reads the chart's own rendered config.json and a real
Vespa, so the request the hook makes is answered by the code a helm upgrade
reaches.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
import yaml
from cogniverse_cli.config import LLM_SERVING_MODAL, compose_values_files
from fastapi import FastAPI

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"
_RUNTIME_URL_LINE = 'RUNTIME_URL="http://cogniverse-runtime:8000"'

pytestmark = [
    pytest.mark.skipif(
        shutil.which("helm") is None,
        reason="helm CLI not installed — chart tests require helm",
    ),
    pytest.mark.requires_docker,
    pytest.mark.requires_vespa,
]


def _render_e2e_stack() -> list[dict]:
    """The chart as ``cogniverse up`` renders it for the k3d ROCm stack."""
    cmd = ["helm", "template", "cogniverse", str(CHART_PATH)]
    for values_file in compose_values_files(
        use_k3d=True,
        backend="rocm",
        serving=LLM_SERVING_MODAL,
        project_root=REPO_ROOT,
    ):
        cmd.extend(["-f", str(values_file)])
    cmd.extend(["--set", "runtime.qualityMonitor.tenantId=test-tenant"])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return [d for d in yaml.safe_load_all(result.stdout) if d is not None]


def _hook_script(docs: list[dict]) -> str:
    (job,) = [
        d
        for d in docs
        if d.get("kind") == "Job"
        and d["metadata"]["name"] == "cogniverse-schema-deployment"
    ]
    return job["spec"]["template"]["spec"]["containers"][0]["command"][-1]


def _rendered_config(docs: list[dict]) -> dict:
    (configmap,) = [
        d
        for d in docs
        if d.get("kind") == "ConfigMap" and "config.json" in (d.get("data") or {})
    ]
    return json.loads(configmap["data"]["config.json"])


@pytest.fixture
def runtime_admin(shared_vespa, tmp_path, monkeypatch):
    """The admin router on a live server, reading the rendered config.json
    and the shipped schemas, with its config store and schemas on Vespa."""
    import cogniverse_vespa.backend  # noqa: F401  (registers the backend)
    from cogniverse_core.registries import schema_registry as schema_registry_module
    from cogniverse_core.registries.backend_registry import BackendRegistry
    from cogniverse_core.registries.schema_registry import SchemaRegistry
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import SystemConfig
    from cogniverse_runtime.routers import admin
    from cogniverse_vespa.config.config_store import VespaConfigStore

    docs = _render_e2e_stack()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_rendered_config(docs)))
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(config_path))

    def reset_registries() -> None:
        BackendRegistry._instance = None
        BackendRegistry._backend_instances.clear()
        BackendRegistry._shared_schema_registry = None
        SchemaRegistry._instance = None
        schema_registry_module._registry_instance = None

    reset_registries()
    port = shared_vespa["http_port"]
    config_manager = ConfigManager(
        store=VespaConfigStore(backend_url="http://localhost", backend_port=port)
    )
    config_manager.set_system_config(
        SystemConfig(backend_url="http://localhost", backend_port=port)
    )
    schema_loader = FilesystemSchemaLoader(REPO_ROOT / "configs" / "schemas")
    admin.set_config_manager(config_manager)
    admin.set_schema_loader(schema_loader)
    created: list[str] = []

    app = FastAPI()

    @app.get("/health")
    def health() -> dict:
        return {"status": "healthy"}

    app.include_router(admin.router, prefix="/admin")

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        server_port = probe.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=server_port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        assert time.monotonic() < deadline, "admin router did not start"
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{server_port}", docs, shared_vespa, created
    finally:
        server.should_exit = True
        thread.join(timeout=30)
        try:
            for schema in created:
                backend = BackendRegistry.get_instance().get_ingestion_backend(
                    "vespa",
                    tenant_id="default",
                    config_manager=config_manager,
                    schema_loader=schema_loader,
                )
                assert backend.delete_schema(schema, tenant_id="default") == [
                    f"{schema}_default_default"
                ]
        finally:
            admin.reset_dependencies()
            reset_registries()


def test_the_rendered_hook_deploys_the_selected_profile_for_every_listed_tenant(
    runtime_admin,
):
    url, docs, vespa, created = runtime_admin
    config = _rendered_config(docs)
    profile = config["backend"]["default_profiles"]["video"]["profile"]
    schema = config["backend"]["profiles"][profile]["schema_name"]
    tenants = [
        tenant["id"]
        for tenant in yaml.safe_load((CHART_PATH / "values.k3s.yaml").read_text())[
            "config"
        ]["tenants"]
    ]
    script = _hook_script(docs)
    assert tenants == ["default"]
    assert re.findall(
        r'/admin/profiles/([^/"]+)/deploy" \\\n.*\n\s*-d \'\{"tenant_id": "([^"]+)"',
        script,
    ) == [(profile, "default")]
    assert script.count(_RUNTIME_URL_LINE) == 1, script

    env = {
        name: value
        for name, value in os.environ.items()
        if name.lower() not in {"http_proxy", "https_proxy", "all_proxy"}
    }
    result = subprocess.run(
        ["/bin/sh", "-c", script.replace(_RUNTIME_URL_LINE, f'RUNTIME_URL="{url}"')],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    (response,) = re.findall(r"^  Response: (.*)$", result.stdout, re.MULTILINE)
    body = json.loads(response)
    body.pop("deployed_at")
    status = body.pop("deployment_status")
    if status == "success":
        created.append(schema)
    assert status in {"success", "already_deployed"}, status
    assert body == {
        "profile_name": profile,
        "tenant_id": "default",
        "schema_name": schema,
        "tenant_schema_name": f"{schema}_default_default",
        "error_message": None,
    }
    assert result.stdout.rstrip().endswith("Schema deployment completed!")

    search = httpx.post(
        f"http://localhost:{vespa['http_port']}/search/",
        json={"yql": f"select * from sources {schema}_default_default where true"},
        timeout=30,
    )
    assert search.status_code == 200, search.text
    if created:
        assert search.json()["root"]["fields"]["totalCount"] == 0
