"""Authentication and model identity for Modal-hosted inference ASGI apps."""

from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

DEFAULT_API_KEY_ENV = "COGNIVERSE_INFERENCE_API_KEY"


def require_bearer_key(
    request: Request,
    *,
    expected_key: str,
) -> None:
    """Require the pinned bearer key without exposing credential values."""

    authorizations = request.headers.getlist("authorization")
    if not authorizations:
        raise HTTPException(
            status_code=401,
            detail="Bearer authorization required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if len(authorizations) != 1:
        raise HTTPException(
            status_code=401,
            detail="Exactly one Bearer authorization header is required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    authorization = authorizations[0]
    scheme, separator, supplied = authorization.partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not supplied
        or supplied != supplied.strip()
        or not hmac.compare_digest(supplied, expected_key)
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def models_response(model_id: str, model_revision: str) -> dict[str, object]:
    """Build the single-model identity contract used during discovery."""

    if not model_id or model_id != model_id.strip():
        raise ValueError("model_id must be a non-empty canonical identifier")
    if (
        not model_revision
        or model_revision != model_revision.strip()
        or model_revision in {"latest", "main", "master"}
    ):
        raise ValueError("model_revision must identify an immutable artifact")
    return {
        "data": [
            {
                "created": 0,
                "id": model_id,
                "object": "model",
                "owned_by": "cogniverse",
                "revision": model_revision,
            }
        ],
        "object": "list",
    }


def build_authenticated_asgi_app(
    production_app: FastAPI,
    *,
    model_id: str,
    model_revision: str,
    api_key_env: str = DEFAULT_API_KEY_ENV,
) -> FastAPI:
    """Wrap a production app with bearer authentication and pinned identity."""

    expected_key = os.environ.get(api_key_env)
    if not expected_key or expected_key != expected_key.strip():
        raise RuntimeError(
            f"Inference authentication is not configured in {api_key_env}"
        )
    identity = models_response(model_id, model_revision)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async with production_app.router.lifespan_context(production_app):
            yield

    app = FastAPI(lifespan=lifespan)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        try:
            require_bearer_key(request, expected_key=expected_key)
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=exc.headers,
            )

        request.scope["headers"] = [
            (name, value)
            for name, value in request.scope["headers"]
            if name.lower() != b"authorization"
        ]
        return await call_next(request)

    @app.get("/v1/models")
    async def list_models() -> dict[str, object]:
        return identity

    app.mount("/", production_app)
    return app
