from __future__ import annotations

from importlib import import_module

import pytest
from cogniverse_cli.modal_inference.vllm import _vllm_command
from cogniverse_cli.modal_inference_config import get_inference_service_spec

SERVICE_MODULES = {
    "vllm_colpali": "cogniverse_cli.modal_inference.vllm_colpali",
    "denseon": "cogniverse_cli.modal_inference.denseon",
}

EXPECTED_DIMENSIONS = {
    "vllm_colpali": 320,
    "denseon": 768,
}

EXPECTED_COMMANDS = {
    "vllm_colpali": (
        "vllm",
        "serve",
        "TomoroAI/tomoro-colqwen3-embed-4b",
        "--revision",
        "bf790bd8780b098b86453444632a184bb770be1a",
        "--served-model-name",
        "TomoroAI/tomoro-colqwen3-embed-4b",
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
        "--max-model-len",
        "4096",
        "--runner",
        "pooling",
        "--convert",
        "embed",
        "--limit-mm-per-prompt",
        '{"video":0,"image":1}',
    ),
    "denseon": (
        "vllm",
        "serve",
        "lightonai/DenseOn",
        "--revision",
        "cb9947ebccb33862d24e3c7ca2edb25e51acd887",
        "--served-model-name",
        "lightonai/DenseOn",
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
        "--runner",
        "pooling",
        "--convert",
        "embed",
        "--dtype",
        "float32",
    ),
}


@pytest.mark.parametrize(("service", "module_name"), SERVICE_MODULES.items())
def test_service_module_registers_only_its_canonical_inference_app(
    service: str,
    module_name: str,
):
    spec = get_inference_service_spec(service)

    module = import_module(module_name)
    app = module.app
    function = app.registered_functions[spec.modal_object]

    assert app.name == spec.modal_app
    assert app.registered_web_endpoints == [spec.modal_object]
    assert set(app.registered_functions) == {spec.modal_object}
    assert function.tag == spec.modal_object
    assert function.spec.gpus == list(spec.gpu_candidates)
    assert list(function.spec.volumes) == ["/root/.cache/huggingface"]
    assert repr(function.spec.volumes["/root/.cache/huggingface"]) == (
        "modal.Volume.from_name('cogniverse-huggingface-cache')"
    )
    assert [repr(secret) for secret in function.spec.secrets] == [
        "modal.Secret.from_name('cogniverse-inference-api-key')"
    ]
    assert spec.output_dimension == EXPECTED_DIMENSIONS[service]
    assert spec.min_containers == 0
    assert spec.scaledown_window == 300


@pytest.mark.parametrize("service", SERVICE_MODULES)
def test_service_launch_command_is_exact_and_revision_pinned(service: str):
    spec = get_inference_service_spec(service)

    assert _vllm_command(spec) == EXPECTED_COMMANDS[service]


def test_service_modules_create_independent_modal_apps():
    apps = {
        service: import_module(module_name).app
        for service, module_name in SERVICE_MODULES.items()
    }

    assert {service: app.name for service, app in apps.items()} == {
        "vllm_colpali": "cogniverse-vllm-colpali",
        "denseon": "cogniverse-denseon",
    }
    assert len({id(app) for app in apps.values()}) == 2


def test_lateon_services_have_no_vllm_launch_contract():
    """LateOn retrieval requires PyLate's exact encode (query expansion over
    masked padding); the public vLLM request schema carries no attention
    mask, so these services must not resolve a vLLM launch command."""
    for service in ("colbert_pylate", "code_colbert_pylate"):
        spec = get_inference_service_spec(service)
        with pytest.raises(ValueError, match="no canonical vLLM launch contract"):
            _vllm_command(spec)
