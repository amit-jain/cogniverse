"""Which collected e2e tests run: opt-in deselections and the Modal provider gate."""

from __future__ import annotations

import os

import pytest
from cogniverse_cli.secrets import read_secret


def _modal_inference_deselections(config, items):
    explicit = os.environ.get("RUN_MODAL_INFERENCE_E2E") == "1" or (
        "requires_modal_inference" in (config.option.markexpr or "")
    )
    if explicit:
        return []
    return [
        item
        for item in items
        if any(item.iter_markers(name="requires_modal_inference"))
    ]


def _telegram_real_flow_deselections(items):
    # read_secret is the shared lookup: env var, then ./.env, then ~/.env, each
    # of which may be a directory of per-key <VAR>.env files. Reading os.environ
    # here instead would ignore a secret provisioned the documented way.
    token = read_secret("TELEGRAM_BOT_TOKEN")
    chat_id = read_secret("TELEGRAM_TEST_CHAT_ID")
    if token and chat_id:
        return [], None
    deselected = [
        item for item in items if any(item.iter_markers(name="requires_telegram_bot"))
    ]
    if not deselected:
        return [], None
    missing = []
    if not token:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not chat_id:
        missing.append("TELEGRAM_TEST_CHAT_ID")
    return deselected, "missing " + " and ".join(missing)


def _require_modal_inference_endpoints(items, endpoints) -> None:
    for item in items:
        for marker in item.iter_markers(name="requires_modal_inference"):
            if len(marker.args) != 1 or not isinstance(marker.args[0], str):
                raise pytest.UsageError(
                    "requires_modal_inference must name exactly one inference service"
                )
            service = marker.args[0]
            endpoint = endpoints.get(service)
            provider = endpoint.provider if endpoint is not None else None
            if provider != "modal":
                pytest.fail(
                    f"{item.nodeid} requires Modal provider for {service!r}; "
                    f"resolved {provider!r}",
                    pytrace=False,
                )
