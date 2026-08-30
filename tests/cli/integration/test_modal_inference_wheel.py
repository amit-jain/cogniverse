from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[3]

_INSTALLED_WHEEL_PROBE = textwrap.dedent(
    """
    from importlib.util import find_spec
    from types import SimpleNamespace

    import httpx

    from cogniverse_cli.inference_endpoints import EndpointCredentials
    from cogniverse_foundation.inference_specs import INFERENCE_SERVICE_SPECS
    from cogniverse_cli.modal_inference_lifecycle import (
        ModalInferenceLifecycle,
        _DEPLOYMENT_MODULES,
        _load_deployment,
    )

    deployment_modules = (
        "cogniverse_cli.modal_inference.serving",
        "cogniverse_cli.modal_inference.vllm",
        "cogniverse_cli.modal_inference.pylate",
        *_DEPLOYMENT_MODULES.values(),
    )
    server_modules = (
        "cogniverse_cli.modal_inference.servers.clap",
        "cogniverse_cli.modal_inference.servers.face",
        "cogniverse_cli.modal_inference.servers.gliner",
        "cogniverse_cli.modal_inference.servers.pylate",
        "cogniverse_cli.modal_inference.servers.videoprism",
    )

    for module_name in (*deployment_modules, *server_modules):
        spec = find_spec(module_name)
        assert spec is not None and spec.origin is not None, module_name
        assert "site-packages" in spec.origin, (module_name, spec.origin)

    def module_is_absent(module_name):
        try:
            return find_spec(module_name) is None
        except ModuleNotFoundError:
            return True

    assert module_is_absent("deploy.modal_inference.vllm")
    assert module_is_absent("cogniverse_runtime.sidecars.clap_embed")

    deployed = []

    class Deployment:
        def __init__(self, service):
            self.service = service

        def deploy(self, *, name):
            deployed.append((self.service, name))

    class Function:
        def __init__(self, service):
            self.service = service
            self.active = 0
            self.updates = []

        def update_autoscaler(self, *, min_containers, scaledown_window):
            self.active = min_containers
            self.updates.append((min_containers, scaledown_window))

        def get_web_url(self):
            return f"https://{self.service.replace('_', '-')}.modal.run"

        def get_current_stats(self):
            return SimpleNamespace(num_total_runners=self.active)

    services = tuple(INFERENCE_SERVICE_SPECS)
    functions = {
        contract.modal_app: Function(name)
        for name, contract in INFERENCE_SERVICE_SPECS.items()
    }

    def load(contract):
        assert _load_deployment(contract) is not None
        return Deployment(contract.name)

    target = INFERENCE_SERVICE_SPECS["vllm_colpali"]

    def handle(request):
        assert request.headers["Authorization"] == "Bearer wheel-secret"
        if request.url.path == target.health_path:
            return httpx.Response(200, json={"status": "ok"})
        assert request.url.path == target.models_path
        return httpx.Response(
            200,
            json={
                "data": [{
                    "id": target.model_id,
                    "revision": target.model_revision,
                }]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with ModalInferenceLifecycle(
            credentials=EndpointCredentials(bearer_token="wheel-secret"),
            client=client,
            deployment_loader=load,
            function_from_name=lambda app_name, _: functions[app_name],
            readiness_poll_interval=0,
        ) as lifecycle:
            lifecycle.deploy(services)
            endpoint = lifecycle.warm(["vllm_colpali"])[0]
            lifecycle.release(["vllm_colpali"])

    assert tuple(service for service, _ in deployed) == services
    assert (endpoint.model_id, endpoint.model_revision) == (
        target.model_id,
        target.model_revision,
    )
    assert functions[target.modal_app].updates == [(1, 300), (0, 300)]
    print("installed cogniverse-cli Modal deployment lifecycle verified")
    """
)


def test_installed_cli_wheel_owns_modal_deployments_and_servers(tmp_path):
    distribution_dir = tmp_path / "dist"
    build = subprocess.run(
        [
            "uv",
            "build",
            "--package",
            "cogniverse-cli",
            "--wheel",
            "--out-dir",
            str(distribution_dir),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    wheels = tuple(distribution_dir.glob("cogniverse_cli-*.whl"))
    assert len(wheels) == 1, tuple(path.name for path in distribution_dir.iterdir())

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("UV_NO_SYNC", None)
    probe = subprocess.run(
        [
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--with",
            str(wheels[0]),
            "python",
            "-c",
            _INSTALLED_WHEEL_PROBE,
        ],
        cwd=run_dir,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert probe.stdout.strip() == (
        "installed cogniverse-cli Modal deployment lifecycle verified"
    )
