"""Bootstrap helpers for cluster secrets created from local machine state.

The chart references Secret objects by name (e.g. ``hf-token``) and
expects them to exist before ``helm upgrade``. On a fresh machine or
after a cluster rebuild they need to be (re)materialized without manual
``kubectl create secret`` steps.

Each helper here:

* Reads material from a well-known local source (env var, HF cache file).
* Applies it to the target namespace with ``kubectl apply`` via a
  ``--dry-run=client`` pipeline, so re-running is idempotent.
* Returns ``True`` on success, ``False`` on skippable miss (material not
  available). Hard errors raise.

Follows the same shape as ``sandbox.sync_gateway_certs_to_cluster()``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console

NAMESPACE = "cogniverse"
HF_TOKEN_SECRET = "hf-token"
HF_CACHE_TOKEN_PATH = Path.home() / ".cache" / "huggingface" / "token"

console = Console()


def _kubectl(args: list[str], input_data: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["kubectl", *args],
        input=input_data,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _read_hf_token() -> Optional[str]:
    """Resolve the HuggingFace token from the usual places.

    Order:
      1. ``HF_TOKEN`` env var (CI / explicit override)
      2. ``HUGGING_FACE_HUB_TOKEN`` env var (legacy)
      3. ``~/.cache/huggingface/token`` (populated by ``huggingface-cli login``)
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(var)
        if val:
            return val.strip()
    if HF_CACHE_TOKEN_PATH.exists():
        text = HF_CACHE_TOKEN_PATH.read_text().strip()
        if text:
            return text
    return None


def sync_hf_token_to_cluster(required: bool = False) -> bool:
    """Create or update the ``hf-token`` Secret in the cogniverse namespace.

    Required for deployments that pull gated HuggingFace models
    (EmbeddingGemma, Gemma chat weights, etc.). Safe no-op when the
    token isn't available *unless* ``required=True`` — callers set that
    when the chart definitely needs the token (e.g. ``inference.embed``
    enabled with a gated model).
    """
    token = _read_hf_token()
    if not token:
        if required:
            console.print(
                "[red]HF_TOKEN not found. Run `huggingface-cli login`, "
                "or export HF_TOKEN, before `cogniverse up`.[/red]"
            )
            return False
        console.print(
            "[yellow]HF_TOKEN not found — skipping hf-token secret. "
            "Gated HuggingFace models (e.g. EmbeddingGemma) will fail to load.[/yellow]"
        )
        return False

    ns_check = _kubectl(["get", "namespace", NAMESPACE])
    if ns_check.returncode != 0:
        _kubectl(["create", "namespace", NAMESPACE])

    rendered = _kubectl([
        "create", "secret", "generic", HF_TOKEN_SECRET,
        "-n", NAMESPACE,
        f"--from-literal=HF_TOKEN={token}",
        "--dry-run=client", "-o", "yaml",
    ])
    if rendered.returncode != 0:
        console.print(f"[red]Failed to render hf-token Secret: {rendered.stderr}[/red]")
        return False

    applied = _kubectl(["apply", "-f", "-"], input_data=rendered.stdout)
    if applied.returncode != 0:
        console.print(f"[red]Failed to apply hf-token Secret: {applied.stderr}[/red]")
        return False

    console.print(f"[green]HuggingFace token synced to {NAMESPACE}/{HF_TOKEN_SECRET}[/green]")
    return True
