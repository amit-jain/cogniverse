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

from cogniverse_cli.constants import NAMESPACE, RELEASE_NAME  # noqa: F401

HF_TOKEN_SECRET = "hf-token"
HF_CACHE_TOKEN_PATH = Path.home() / ".cache" / "huggingface" / "token"

# The messaging deployment mounts TELEGRAM_BOT_TOKEN from this Secret; the
# name must track the chart's "<fullname>-messaging-secrets".
MESSAGING_SECRET = f"{RELEASE_NAME}-messaging-secrets"
TELEGRAM_TOKEN_KEY = "telegram-bot-token"

# Every secret resolves the same way, most specific first:
#   1. the environment variable          (CI, or an explicit one-off override)
#   2. ./.env                            (project-local, gitignored)
#   3. ~/.env                            (shared across checkouts on this machine)
# and only then any tool-specific location (e.g. the HuggingFace cache file).
# Each .env may be a DIRECTORY holding one <VAR>.env file per secret, or a
# single FILE of KEY=value lines; both shapes are read, so the project copy
# overrides the home copy per-variable rather than wholesale.
PROJECT_ENV = Path(".env")
HOME_ENV = Path.home() / ".env"

console = Console()


def _kubectl(
    args: list[str], input_data: Optional[str] = None
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["kubectl", *args],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        raise SystemExit(
            "kubectl not found on PATH — install kubectl to sync secrets"
        ) from None
    except subprocess.TimeoutExpired:
        raise SystemExit("kubectl timed out after 30s") from None


def _value_from_env_line(line: str, var: str) -> Optional[str]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith(f"{var}="):
        return line.split("=", 1)[1].strip().strip("'\"")
    return None


def _read_from_env_location(location: Path, var: str) -> Optional[str]:
    """Read ``var`` from one .env location, directory or single file."""
    if location.is_dir():
        candidate = location / f"{var}.env"
        if not candidate.exists():
            return None
        for line in candidate.read_text().splitlines():
            found = _value_from_env_line(line, var)
            if found:
                return found
            # A per-variable file may hold just the bare value.
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" not in stripped:
                return stripped
        return None

    if location.is_file():
        for line in location.read_text().splitlines():
            found = _value_from_env_line(line, var)
            if found:
                return found
    return None


def read_secret(*names: str, extra_paths: tuple[Path, ...] = ()) -> Optional[str]:
    """Resolve a secret through the shared lookup order.

    Accepts several ``names`` so aliases (HF_TOKEN / HUGGING_FACE_HUB_TOKEN)
    share one path. ``extra_paths`` are tool-specific files holding a bare
    value, consulted last.
    """
    for var in names:
        val = os.environ.get(var)
        if val:
            return val.strip()

    for location in (PROJECT_ENV, HOME_ENV):
        for var in names:
            val = _read_from_env_location(location, var)
            if val:
                return val

    for path in extra_paths:
        if path.exists():
            text = path.read_text().strip()
            if text:
                return text
    return None


def _read_hf_token() -> Optional[str]:
    """HuggingFace token: shared lookup, then the ``huggingface-cli login`` cache."""
    return read_secret(
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        extra_paths=(HF_CACHE_TOKEN_PATH,),
    )


def sync_hf_token_to_cluster(required: bool = False) -> bool:
    """Create or update the ``hf-token`` Secret in the cogniverse namespace.

    Required for deployments that pull gated HuggingFace models
    (the Gemma LLM chat weights, etc.). Safe no-op when the
    token isn't available *unless* ``required=True`` — callers set that
    when the chart definitely needs the token (e.g. ``inference.dense_vllm``
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
            "Gated HuggingFace models (e.g. the Gemma LLM) will fail to load.[/yellow]"
        )
        return False

    ns_check = _kubectl(["get", "namespace", NAMESPACE])
    if ns_check.returncode != 0:
        _kubectl(["create", "namespace", NAMESPACE])

    rendered = _kubectl(
        [
            "create",
            "secret",
            "generic",
            HF_TOKEN_SECRET,
            "-n",
            NAMESPACE,
            f"--from-literal=HF_TOKEN={token}",
            "--dry-run=client",
            "-o",
            "yaml",
        ]
    )
    if rendered.returncode != 0:
        console.print(f"[red]Failed to render hf-token Secret: {rendered.stderr}[/red]")
        return False

    applied = _kubectl(["apply", "-f", "-"], input_data=rendered.stdout)
    if applied.returncode != 0:
        console.print(f"[red]Failed to apply hf-token Secret: {applied.stderr}[/red]")
        return False

    console.print(
        f"[green]HuggingFace token synced to {NAMESPACE}/{HF_TOKEN_SECRET}[/green]"
    )
    return True


def _read_telegram_token() -> Optional[str]:
    """Telegram bot token via the shared lookup order."""
    return read_secret("TELEGRAM_BOT_TOKEN")


def sync_telegram_token_to_cluster(required: bool = False) -> bool:
    """Create or update the messaging Secret holding the Telegram bot token.

    The chart's messaging deployment reads TELEGRAM_BOT_TOKEN from
    ``<release>-messaging-secrets`` key ``telegram-bot-token``; without this
    the gateway pod cannot start. Safe no-op when no token is available unless
    ``required=True`` (callers set that when messaging is being enabled).
    """
    token = _read_telegram_token()
    if not token:
        message = (
            "TELEGRAM_BOT_TOKEN not found. Export it, or put it in "
            f"{PROJECT_ENV}/TELEGRAM_BOT_TOKEN.env or {HOME_ENV}, "
            "before enabling messaging."
        )
        if required:
            console.print(f"[red]{message}[/red]")
            return False
        console.print(
            f"[yellow]{message} Skipping — the messaging gateway will not "
            f"start without it.[/yellow]"
        )
        return False

    ns_check = _kubectl(["get", "namespace", NAMESPACE])
    if ns_check.returncode != 0:
        _kubectl(["create", "namespace", NAMESPACE])

    rendered = _kubectl(
        [
            "create",
            "secret",
            "generic",
            MESSAGING_SECRET,
            "-n",
            NAMESPACE,
            f"--from-literal={TELEGRAM_TOKEN_KEY}={token}",
            "--dry-run=client",
            "-o",
            "yaml",
        ]
    )
    if rendered.returncode != 0:
        console.print(
            f"[red]Failed to render {MESSAGING_SECRET} Secret: {rendered.stderr}[/red]"
        )
        return False

    applied = _kubectl(["apply", "-f", "-"], input_data=rendered.stdout)
    if applied.returncode != 0:
        console.print(
            f"[red]Failed to apply {MESSAGING_SECRET} Secret: {applied.stderr}[/red]"
        )
        return False

    console.print(
        f"[green]Telegram bot token synced to {NAMESPACE}/{MESSAGING_SECRET}[/green]"
    )
    return True
