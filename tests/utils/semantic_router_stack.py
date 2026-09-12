"""Envoy data plane and readiness for a local semantic-router stack.

The stack's Envoy runs the chart's own data-plane config with the in-cluster
peers substituted for the local containers, so every routing knob production
depends on is present in the stack the tests route through. A hand-maintained
twin of that config drifts: the chart's per-message ext_proc deadline is what
keeps Envoy from cancelling the router mid-decision, and without it every
chat completion through the stack answers 504 while ``/v1/models`` still
answers 200.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_DIR = REPO_ROOT / "charts" / "cogniverse"
CHART_ENVOY_TEMPLATE = CHART_DIR / "files" / "semantic-router" / "envoy.yaml"
CHART_VALUES = CHART_DIR / "values.yaml"

# The release the chart is deployed under: ``cogniverse.fullname`` renders to
# it, so the router's address is ``<release>-semantic-router`` in the cluster
# and on the local docker network alike.
RELEASE = "cogniverse"
ROUTER_ALIAS = f"{RELEASE}-semantic-router"
UPSTREAM_ALIAS = "stub-upstream"
UPSTREAM_PORT = 8000

_EXPRESSION = re.compile(r"\{\{-?\s*(?P<body>.*?)\s*-?\}\}")
_CONTROL = re.compile(
    r"^\s*\{\{-?\s*(?P<keyword>if|else|end)\b\s*(?P<rest>.*?)\s*-?\}\}\s*$"
)
_EQUALITY = re.compile(r'^eq\s+\((?P<expression>.+)\)\s+"(?P<literal>[^"]*)"$')


class ChartRenderError(RuntimeError):
    """A chart expression the local substitution does not know how to render."""


def _unwrap(expression: str) -> str:
    text = expression.strip()
    if text.startswith("int "):
        text = text[len("int ") :].strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    return text


def _resolve(expression: str, helpers: dict[str, str], values: dict[str, Any]) -> str:
    text = _unwrap(expression)
    if text.startswith(".Values."):
        node: Any = values
        for key in text[len(".Values.") :].split("."):
            if not isinstance(node, dict) or key not in node:
                raise ChartRenderError(
                    f"{CHART_ENVOY_TEMPLATE.name} reads {{{{ {expression} }}}}, "
                    f"absent from {CHART_VALUES}"
                )
            node = node[key]
        return str(node)
    if text in helpers:
        return helpers[text]
    raise ChartRenderError(
        f"{CHART_ENVOY_TEMPLATE.name} uses {{{{ {expression} }}}}, which the local "
        f"stack has no substitution for; add one in {Path(__file__).name}"
    )


def _condition(rest: str, helpers: dict[str, str], values: dict[str, Any]) -> bool:
    match = _EQUALITY.match(rest)
    if match is None:
        raise ChartRenderError(
            f"{CHART_ENVOY_TEMPLATE.name} has condition {{{{ if {rest} }}}}, which "
            f"the local stack cannot evaluate"
        )
    return _resolve(match["expression"], helpers, values) == match["literal"]


def render_envoy_config(
    *,
    upstream_host: str = UPSTREAM_ALIAS,
    upstream_port: int = UPSTREAM_PORT,
    release: str = RELEASE,
) -> str:
    """The chart's Envoy config with the local stack's peers substituted.

    The local upstream is a plain-HTTP stub, so the chart's TLS-only blocks
    (upstream TLS context, ``auto_host_rewrite``) render out exactly as they
    do for an in-cluster HTTP backend.
    """
    values = yaml.safe_load(CHART_VALUES.read_text())
    helpers = {
        'include "cogniverse.srUpstreamHost" .': upstream_host,
        'include "cogniverse.srUpstreamPort" .': str(upstream_port),
        'include "cogniverse.srUpstreamProtocol" .': "http",
        'include "cogniverse.fullname" .': release,
    }
    emit: list[bool] = []
    rendered: list[str] = []
    for line in CHART_ENVOY_TEMPLATE.read_text().splitlines():
        control = _CONTROL.match(line)
        if control is not None:
            keyword = control["keyword"]
            if keyword == "if":
                emit.append(_condition(control["rest"], helpers, values))
            elif keyword == "else":
                emit[-1] = not emit[-1]
            else:
                emit.pop()
            continue
        if all(emit):
            rendered.append(
                _EXPRESSION.sub(
                    lambda match: _resolve(match["body"], helpers, values), line
                )
            )
    text = "\n".join(rendered) + "\n"
    yaml.safe_load(text)
    return text


def envoy_listener_port() -> int:
    """The container port the chart's Envoy listens on."""
    values = yaml.safe_load(CHART_VALUES.read_text())
    return int(values["semanticRouter"]["envoy"]["service"]["port"])


def routed_chat(
    base_url: str,
    *,
    tenant_id: str,
    tenant_tier: str,
    prompt: str = "semantic router readiness probe",
    timeout_s: float = 30.0,
) -> tuple[int, str, dict[str, Any] | None]:
    """One chat completion through the stack: status, detail, reflection."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    try:
        response = requests.post(
            url,
            json={"model": "auto", "messages": [{"role": "user", "content": prompt}]},
            headers={
                "x-authz-user-id": tenant_id,
                "x-authz-user-groups": tenant_tier,
            },
            timeout=timeout_s,
        )
    except requests.RequestException as error:
        return 0, f"{type(error).__name__}: {error}", None
    if response.status_code != 200:
        return response.status_code, repr(response.text[:400]), None
    try:
        reflection = json.loads(response.json()["choices"][0]["message"]["content"])
    except (ValueError, KeyError, IndexError, TypeError) as error:
        return 200, f"unreadable reflection ({error}): {response.text[:400]!r}", None
    return 200, "", reflection


def wait_for_routed_chat(
    base_url: str,
    *,
    tenant_id: str,
    tenant_tier: str,
    expected_model: str,
    budget_s: float,
    poll_s: float = 2.0,
    request_timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Block until a real chat completion routes end to end; raise if it never does.

    ``/v1/models`` answers from Envoy long before the ext_proc hop can carry a
    body, so waiting on it hands the tests a stack whose every completion
    times out. This waits on the completion itself, and on the router having
    rewritten the model, which is the decision the stack exists to make.

    ``request_timeout_s`` bounds each attempt, so a stack that accepts the
    connection and never answers still fails inside ``budget_s``.
    """
    deadline = time.monotonic() + budget_s
    attempts = 0
    while True:
        attempts += 1
        status, detail, reflection = routed_chat(
            base_url,
            tenant_id=tenant_id,
            tenant_tier=tenant_tier,
            timeout_s=request_timeout_s,
        )
        if reflection is not None:
            served = reflection.get("served_model")
            if served == expected_model:
                return reflection
            last = f"HTTP 200 but served_model={served!r}, expected {expected_model!r}"
        else:
            last = f"HTTP {status} {detail}" if status else detail
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"semantic-router stack never completed a routed chat completion: "
                f"POST {base_url.rstrip('/')}/chat/completions as tenant "
                f"{tenant_id!r} (groups {tenant_tier!r}) did not return "
                f"served_model {expected_model!r} in {budget_s}s over {attempts} "
                f"attempts. Last response: {last}"
            )
        time.sleep(poll_s)
