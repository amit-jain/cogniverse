"""Bearer credentials for authenticated inference endpoints."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping
from urllib.parse import urlsplit

from cogniverse_foundation.config.bootstrap import (
    inference_api_key_from_environment,
)

_EMPTY_HEADERS: Mapping[str, str] = MappingProxyType({})


def is_modal_inference_url(base_url: str) -> bool:
    """Return whether ``base_url`` is a root HTTPS Modal inference URL."""

    parsed = urlsplit(base_url)
    return (
        parsed.scheme == "https"
        and parsed.hostname is not None
        and parsed.hostname.endswith(".modal.run")
        and parsed.username is None
        and parsed.password is None
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
    )


def inference_headers(base_url: str) -> Mapping[str, str]:
    """Return immutable headers for one canonical inference root URL."""

    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("inference endpoint must be a root HTTP(S) URL")

    if not parsed.hostname.endswith(".modal.run"):
        return _EMPTY_HEADERS
    if not is_modal_inference_url(base_url):
        raise ValueError("Modal inference endpoints require HTTPS")
    api_key = inference_api_key_from_environment()
    return MappingProxyType({"Authorization": f"Bearer {api_key}"})
