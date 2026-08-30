"""Authenticated scale-to-zero Modal App for pinned CLAP embeddings."""

import modal

from cogniverse_foundation.inference_specs import get_inference_service_spec

_SPEC = get_inference_service_spec("clap_embed")
_CACHE_PATH = "/root/.cache/huggingface"

_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgomp1", "libsndfile1")
    .pip_install(
        "fastapi==0.135.3",
        "librosa==0.11.0",
        "numpy==1.26.4",
        "pydantic==2.13.0",
        "soundfile==0.13.1",
        "torch==2.8.0",
        "transformers==4.56.2",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env(
        {
            "CLAP_EMBED_DEVICE": "cuda",
            "CLAP_EMBED_MODEL": _SPEC.model_id,
            "CLAP_EMBED_MODEL_REVISION": _SPEC.model_revision,
            "HF_HOME": _CACHE_PATH,
        }
    )
    .add_local_python_source(
        "cogniverse_cli.modal_inference",
        "cogniverse_foundation.inference_specs",
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
@modal.concurrent(max_inputs=32)
@modal.asgi_app()
def inference():
    from cogniverse_cli.modal_inference.servers.clap import (
        ClapEmbedConfig,
        build_app,
    )
    from cogniverse_cli.modal_inference.serving import build_authenticated_asgi_app

    production_app = build_app(
        ClapEmbedConfig(
            model_name=_SPEC.model_id,
            model_revision=_SPEC.model_revision,
            device="cuda",
        )
    )
    return build_authenticated_asgi_app(
        production_app,
        model_id=_SPEC.model_id,
        model_revision=_SPEC.model_revision,
    )
