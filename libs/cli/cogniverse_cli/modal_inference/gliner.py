"""Authenticated scale-to-zero Modal App for the pinned GLiNER service."""

import modal

from cogniverse_cli.modal_inference_config import get_inference_service_spec

_SPEC = get_inference_service_spec("gliner")
_CACHE_PATH = "/root/.cache/huggingface"

_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi==0.135.3",
        "gliner==0.2.26",
        "pydantic==2.13.0",
        "torch==2.5.1",
    )
    .env({"DEVICE": "cuda", "HF_HOME": _CACHE_PATH, "MODEL_NAME": _SPEC.model_id})
    .add_local_python_source(
        "cogniverse_cli.modal_inference",
        "cogniverse_cli.modal_inference_config",
        copy=True,
    )
)
_volume = modal.Volume.from_name(
    "cogniverse-huggingface-cache",
    create_if_missing=True,
)
_secrets = [
    modal.Secret.from_name(
        "cogniverse-inference-api-key",
        required_keys=["COGNIVERSE_INFERENCE_API_KEY"],
    )
]

app = modal.App(_SPEC.modal_app)


@app.function(
    image=_image,
    gpu=list(_SPEC.gpu_candidates),
    volumes={_CACHE_PATH: _volume},
    secrets=_secrets,
    min_containers=_SPEC.min_containers,
    scaledown_window=_SPEC.scaledown_window,
    timeout=900,
    startup_timeout=900,
    serialized=True,
    name=_SPEC.modal_object,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def inference():
    from cogniverse_cli.modal_inference.servers.gliner import app as production_app
    from cogniverse_cli.modal_inference.serving import build_authenticated_asgi_app

    return build_authenticated_asgi_app(
        production_app,
        model_id=_SPEC.model_id,
        model_revision=_SPEC.model_revision,
    )
