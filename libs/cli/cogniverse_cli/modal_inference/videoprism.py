"""Authenticated scale-to-zero Modal App for the pinned VideoPrism service."""

import modal

from cogniverse_cli.modal_inference_config import get_inference_service_spec

_SPEC = get_inference_service_spec("videoprism_jax")
_CACHE_PATH = "/root/.cache/huggingface"

_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install(
        "einops==0.8.1",
        "einshape==1.0",
        "fastapi==0.135.3",
        "flax==0.12.0",
        "huggingface-hub==0.36.2",
        "jax[cuda12]==0.9.2",
        "numpy==2.2.6",
        "opencv-python-headless==4.13.0.92",
        "pydantic==2.13.0",
        "sentencepiece==0.2.1",
        "tensorflow==2.20.0",
        "tf-keras==2.20.1",
    )
    .run_commands(
        "pip install --no-deps 'videoprism @ "
        "git+https://github.com/google-deepmind/videoprism.git@"
        f"{_SPEC.source_revision}'"
    )
    .env(
        {
            "HF_HOME": _CACHE_PATH,
            "JAX_PLATFORM_NAME": "gpu",
            "JAX_PLATFORMS": "cuda",
            "MODEL_NAME": _SPEC.model_id,
            "NUM_FRAMES": "16",
        }
    )
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
    timeout=1200,
    startup_timeout=1200,
    serialized=True,
    name=_SPEC.modal_object,
)
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def inference():
    from cogniverse_cli.modal_inference.servers.videoprism import app as production_app
    from cogniverse_cli.modal_inference.serving import build_authenticated_asgi_app

    return build_authenticated_asgi_app(
        production_app,
        model_id=_SPEC.model_id,
        model_revision=_SPEC.model_revision,
    )
