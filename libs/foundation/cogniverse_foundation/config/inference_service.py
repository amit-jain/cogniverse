"""Availability contract for inference services with an in-process fallback.

Several embedders run either against a sidecar named in
``INFERENCE_SERVICE_URLS`` or, when no URL is configured, by loading the
model in-process. The deployed runtime image carries no deep-learning
runtime, so the in-process branch cannot execute there. Callers gate that
branch on :func:`require_in_process_backend` so an unconfigured service
fails naming itself instead of raising from inside a model loader.
"""

import importlib.util

__all__ = ["InferenceServiceUnavailableError", "require_in_process_backend"]


class InferenceServiceUnavailableError(RuntimeError):
    """A service has no configured sidecar and cannot run in-process."""

    def __init__(self, service: str, module: str):
        self.service = service
        self.module = module
        super().__init__(
            f"{service} inference service is not configured and its in-process "
            f"backend is unavailable in this image (no module named {module!r}). "
            f"Set INFERENCE_SERVICE_URLS[{service!r}] to the {service} sidecar "
            f"URL, or install {module!r} to run {service} in-process."
        )


def require_in_process_backend(service: str, *, module: str) -> None:
    """Raise :class:`InferenceServiceUnavailableError` when ``module`` is absent."""
    try:
        found = importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        found = False
    if not found:
        raise InferenceServiceUnavailableError(service, module)
