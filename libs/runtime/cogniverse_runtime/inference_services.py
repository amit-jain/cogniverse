"""Validation for environment-configured inference service endpoints."""

from __future__ import annotations

import json
from urllib.parse import urlsplit


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate service name {name!r}")
        result[name] = value
    return result


def parse_inference_service_urls(raw: str | None) -> dict[str, str] | None:
    """Parse an explicit service-name to HTTP(S) endpoint mapping.

    ``None`` means the environment variable was absent and therefore does not
    override the caller's normal discovery or persisted configuration. Every
    explicitly supplied value, including an empty string, is validated.
    """
    if raw is None:
        return None

    try:
        parsed = json.loads(raw, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError("INFERENCE_SERVICE_URLS must be a valid JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError("INFERENCE_SERVICE_URLS must be a JSON object")

    service_urls: dict[str, str] = {}
    for name, url in parsed.items():
        if not name or name != name.strip():
            raise ValueError(
                "INFERENCE_SERVICE_URLS service name must be non-empty "
                "and contain no surrounding whitespace"
            )
        if not isinstance(url, str):
            raise ValueError(f"INFERENCE_SERVICE_URLS[{name!r}] URL must be a string")
        if not url:
            raise ValueError(f"INFERENCE_SERVICE_URLS[{name!r}] URL must not be empty")
        if any(character.isspace() for character in url):
            raise ValueError(
                f"INFERENCE_SERVICE_URLS[{name!r}] URL must not contain whitespace"
            )

        try:
            parts = urlsplit(url)
            port = parts.port
        except ValueError as exc:
            raise ValueError(
                f"INFERENCE_SERVICE_URLS[{name!r}] URL must use a valid port"
            ) from exc
        if port is not None and not 1 <= port <= 65535:
            raise ValueError(
                f"INFERENCE_SERVICE_URLS[{name!r}] URL must use a valid port"
            )
        if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
            raise ValueError(
                f"INFERENCE_SERVICE_URLS[{name!r}] must be an absolute HTTP or HTTPS URL"
            )
        if parts.username is not None or parts.password is not None:
            raise ValueError(
                f"INFERENCE_SERVICE_URLS[{name!r}] URL must not include credentials"
            )
        if parts.fragment:
            raise ValueError(
                f"INFERENCE_SERVICE_URLS[{name!r}] URL must not include a fragment"
            )
        service_urls[name] = url

    return service_urls
