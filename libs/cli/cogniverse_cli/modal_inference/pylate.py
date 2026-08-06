"""Modal App factory for exact authenticated PyLate ColBERT services."""

from __future__ import annotations

from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from fastapi import FastAPI

    from cogniverse_cli.modal_inference_config import InferenceServiceSpec

_API_KEY_SECRET = "cogniverse-inference-api-key"
_HF_CACHE_NAME = "cogniverse-huggingface-cache"
_HF_CACHE_PATH = "/root/.cache/huggingface"

# Package pins match the workspace lock so the served encode reproduces the
# in-process PyLate oracle. ``pylate`` installs without dependencies because
# its indexing extra (fast-plaid) is not needed on the serve path.
_PYLATE_PACKAGES = (
    "fastapi==0.135.3",
    "datasets==4.8.4",
    "numpy==2.4.4",
    "pydantic==2.13.0",
    "sentence-transformers==5.1.1",
    "torch==2.8.0",
    "transformers==4.56.2",
)
_PYLATE_PIN = "pylate==1.4.0"


def _pylate_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(*_PYLATE_PACKAGES)
        .pip_install(_PYLATE_PIN, extra_options="--no-deps")
        .env({"HF_HOME": _HF_CACHE_PATH})
        .add_local_python_source(
            "cogniverse_cli.modal_inference",
            "cogniverse_cli.modal_inference_config",
            copy=True,
        )
    )


def build_pylate_app(spec: InferenceServiceSpec) -> modal.App:
    """Declare one scale-to-zero Modal App for a pinned PyLate service."""

    image = _pylate_image()
    volume = modal.Volume.from_name(_HF_CACHE_NAME, create_if_missing=True)
    secrets = [
        modal.Secret.from_name(
            _API_KEY_SECRET,
            required_keys=["COGNIVERSE_INFERENCE_API_KEY"],
        )
    ]

    app = modal.App(spec.modal_app)

    @app.function(
        image=image,
        gpu=list(spec.gpu_candidates),
        volumes={_HF_CACHE_PATH: volume},
        secrets=secrets,
        min_containers=spec.min_containers,
        scaledown_window=spec.scaledown_window,
        timeout=900,
        startup_timeout=900,
        serialized=True,
        name=spec.modal_object,
    )
    @modal.concurrent(max_inputs=100)
    @modal.asgi_app()
    def inference() -> FastAPI:
        from cogniverse_cli.modal_inference.servers.pylate import build_app
        from cogniverse_cli.modal_inference.serving import (
            build_authenticated_asgi_app,
        )

        return build_authenticated_asgi_app(
            build_app(spec.model_id, spec.model_revision, "cuda"),
            model_id=spec.model_id,
            model_revision=spec.model_revision,
        )

    return app
