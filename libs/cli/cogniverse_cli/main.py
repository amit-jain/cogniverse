"""Cogniverse CLI entrypoint.

Provides ``up``, ``down``, ``status``, and ``logs`` commands for
deploying and managing the multi-agent RAG stack.
"""

from __future__ import annotations

import os
import subprocess
import sys

import click
import httpx
from rich.console import Console
from rich.table import Table

from cogniverse_cli.argo import deploy_workflow_templates, install_argo_controller
from cogniverse_cli.cluster import (
    CLUSTER_NAME,
    NAMESPACE,
    check_prerequisites,
    cluster_exists,
    create_cluster,
    delete_cluster,
    get_install_commands,
    has_existing_k8s,
    install_missing_prerequisites,
)
from cogniverse_cli.config import (
    get_chart_path,
    get_values_file,
    get_workflows_path,
    resolve_project_root,
)
from cogniverse_cli.deploy import helm_install, helm_uninstall
from cogniverse_cli.health import check_service_health, wait_for_url
from cogniverse_cli.images import (
    build_images,
    has_workspace_source,
    import_images,
    pull_and_import_third_party,
)

console = Console()

SERVICE_HEALTH_URLS: dict[str, str] = {
    "Vespa": "http://localhost:19071/state/v1/health",
    "Runtime": "http://localhost:8000/health",
    "Dashboard": "http://localhost:8501/_stcore/health",
    "Phoenix": "http://localhost:6006/health",
    "LLM": "http://localhost:11434/api/tags",
    "Argo": "https://localhost:2746/api/v1/info",
}

SERVICE_ENDPOINTS: dict[str, str] = {
    "Vespa": "http://localhost:8080",
    "Runtime": "http://localhost:8000",
    "Dashboard": "http://localhost:8501",
    "Phoenix": "http://localhost:6006",
    "LLM": "http://localhost:11434",
    "Argo": "http://localhost:2746",
}

# Maps service CLI argument to the kubectl resource type and name suffix.
_SERVICE_KUBECTL_RESOURCE: dict[str, str] = {
    "vespa": "statefulset/cogniverse-vespa",
    "phoenix": "statefulset/cogniverse-phoenix",
    "llm": "statefulset/cogniverse-llm",
    "runtime": "deployment/cogniverse-runtime",
    "dashboard": "deployment/cogniverse-dashboard",
    "argo": "deployment/argo-server",
}


def _probe_host_llm(url: str = "http://localhost:11434/api/tags") -> bool:
    """Return True if a local LLM endpoint responds at *url* within 3 seconds."""
    try:
        resp = httpx.get(url, timeout=3)
        return resp.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def _llm_statefulset_exists() -> bool:
    """Return True if the cogniverse-llm statefulset exists in the cluster."""
    result = subprocess.run(
        [
            "kubectl",
            "get",
            "statefulset",
            "cogniverse-llm",
            "-n",
            NAMESPACE,
        ],
        capture_output=True,
        check=False,
        timeout=10,
    )
    return result.returncode == 0


def _print_status_table() -> None:
    """Print a Rich table showing health of all services."""
    health = check_service_health(SERVICE_HEALTH_URLS)
    table = Table(title="Cogniverse Stack Status")
    table.add_column("Service", style="bold")
    table.add_column("Health")
    table.add_column("Endpoint")
    for name, url in SERVICE_ENDPOINTS.items():
        is_healthy = health.get(name, False)
        status = "[green]healthy[/green]" if is_healthy else "[red]down[/red]"
        table.add_row(name, status, url)
    console.print(table)


@click.group()
def cli() -> None:
    """Cogniverse -- deploy and manage the multi-agent RAG stack."""


@cli.command()
@click.option(
    "--llm",
    "llm_mode",
    type=click.Choice(["auto", "builtin", "external"]),
    default="auto",
    help="LLM deployment mode.",
)
@click.option("--llm-url", default=None, help="External LLM endpoint URL.")
@click.option(
    "--image-source",
    default=None,
    help="Override image source (directory with workspace source).",
)
@click.option(
    "--messaging/--no-messaging",
    default=False,
    help="Enable Telegram messaging gateway (requires TELEGRAM_BOT_TOKEN env var).",
)
def up(llm_mode: str, llm_url: str | None, image_source: str | None, messaging: bool) -> None:
    """Deploy the full Cogniverse stack."""
    # 1. Detect environment — a running k3d cluster counts as local, not prod
    k3d_running = cluster_exists()
    if k3d_running:
        use_k3d = True
    else:
        existing_k8s = has_existing_k8s()
        use_k3d = not existing_k8s

    # 2. Check prerequisites (require k3d only if no existing K8s)
    missing = check_prerequisites(require_k3d=use_k3d)
    if missing:
        commands = get_install_commands(missing)
        console.print("[yellow]Missing prerequisites:[/yellow]")
        for tool, cmd in commands:
            console.print(f"  [bold]{tool}[/bold]: {cmd}")

        if not click.confirm("\nInstall these now?", default=True):
            console.print("[red]Cannot proceed without prerequisites.[/red]")
            sys.exit(1)

        still_missing = install_missing_prerequisites(missing)
        if still_missing:
            console.print(f"[red]Failed to install: {', '.join(still_missing)}[/red]")
            console.print("[red]Please install manually using the commands above.[/red]")
            sys.exit(1)
        console.print("[green]Prerequisites installed[/green]")

    # 3. Detect LLM BEFORE creating cluster (cluster binds port 11434)
    host_llm_detected = False
    if llm_mode == "auto" and use_k3d and not k3d_running:
        host_llm_detected = _probe_host_llm()
        if host_llm_detected:
            console.print("[cyan]Detected host LLM, will use external mode.[/cyan]")

    # 4. Create k3d cluster if needed (local mode only)
    if use_k3d:
        if not k3d_running:
            console.print("[cyan]Creating k3d cluster...[/cyan]")
            # Exclude LLM port if host LLM is running (avoids port conflict)
            exclude = [11434] if host_llm_detected else None
            create_cluster(exclude_ports=exclude)
        else:
            console.print("[cyan]Using existing k3d cluster.[/cyan]")

    # 4. Build/import images if workspace source available
    project_root = resolve_project_root()
    if project_root and has_workspace_source(project_root):
        console.print("[cyan]Building container images...[/cyan]")
        tags = build_images(project_root)
        if use_k3d:
            console.print("[cyan]Importing images into k3d...[/cyan]")
            import_images(CLUSTER_NAME, tags)

    # 6. Detect LLM mode and build Helm set_values overrides
    set_values: dict[str, str] = {}
    if llm_mode == "auto":
        if host_llm_detected or _probe_host_llm():
            console.print(
                "[cyan]Using host LLM endpoint (external mode).[/cyan]"
            )
            external_url = (
                "http://host.k3d.internal:11434" if use_k3d else "http://localhost:11434"
            )
            set_values["llm.builtin.enabled"] = "false"
            set_values["llm.external.enabled"] = "true"
            set_values["llm.external.url"] = external_url
        else:
            console.print("[cyan]No local Ollama detected, using builtin LLM.[/cyan]")
    elif llm_mode == "external":
        if llm_url:
            resolved_url = llm_url
        elif use_k3d:
            resolved_url = "http://host.k3d.internal:11434"
        else:
            console.print(
                "[red]--llm-url is required when using --llm=external "
                "with an existing Kubernetes cluster.[/red]"
            )
            sys.exit(1)
        set_values["llm.builtin.enabled"] = "false"
        set_values["llm.external.enabled"] = "true"
        set_values["llm.external.url"] = resolved_url
    # llm_mode == "builtin" requires no overrides (chart defaults)

    # Messaging gateway
    if messaging:
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not bot_token:
            console.print(
                "[red]--messaging requires TELEGRAM_BOT_TOKEN env var.[/red]"
            )
            sys.exit(1)
        set_values["messaging.enabled"] = "true"
        console.print("[cyan]Messaging gateway enabled (Telegram).[/cyan]")

    # 5a. Resolve chart and values file
    llm_is_external = set_values.get("llm.external.enabled") == "true"
    chart_path = get_chart_path()
    values_file = get_values_file(prod=not use_k3d)

    # 5b. Pre-pull third-party images into k3d (reads image refs from values file)
    if use_k3d:
        console.print("[cyan]Pre-pulling third-party images...[/cyan]")
        pull_and_import_third_party(
            CLUSTER_NAME, values_file, skip_llm=llm_is_external,
        )
    console.print("[cyan]Deploying Helm release...[/cyan]")
    try:
        helm_install(chart_path, values_file, set_values=set_values or None)
        console.print("[green]Helm release deployed[/green]")
    except RuntimeError as exc:
        console.print(f"[yellow]Helm install warning: {exc}[/yellow]")
        console.print("[yellow]Continuing — pods may still come up[/yellow]")

    # 7. Install Argo controller (before waiting, so pods can start in parallel)
    console.print("[cyan]Installing Argo Workflows controller...[/cyan]")
    try:
        install_argo_controller()
        console.print("[green]Argo Workflows installed[/green]")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        console.print(f"[yellow]Argo install failed: {exc}[/yellow]")

    # 8. Deploy workflow templates
    try:
        workflows_path = get_workflows_path()
        console.print("[cyan]Deploying workflow templates...[/cyan]")
        deploy_workflow_templates(workflows_path)
        console.print("[green]Workflow templates deployed[/green]")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        console.print(f"[yellow]Workflow deploy skipped: {exc}[/yellow]")

    # 9. Wait for ALL pods to be ready (cogniverse namespace)
    console.print("[cyan]Waiting for all pods to be ready (up to 5 min)...[/cyan]")
    try:
        subprocess.run(
            ["kubectl", "wait", "--for=condition=ready", "pod",
             "-l", "app.kubernetes.io/instance=cogniverse",
             "-n", NAMESPACE, "--timeout=300s"],
            check=True, capture_output=True, timeout=310,
        )
        console.print("[green]All cogniverse pods ready[/green]")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", NAMESPACE],
            capture_output=True, text=True, timeout=10,
        )
        console.print(f"[yellow]Some pods not ready:[/yellow]\n{result.stdout}")

    # 10. Verify services via HTTP health checks
    # With NodePort services on k3d, ports are directly reachable via loadbalancer.
    console.print("[cyan]Verifying service health...[/cyan]")
    health_urls = {
        name: url
        for name, url in SERVICE_HEALTH_URLS.items()
        if not (llm_is_external and name == "LLM")
    }
    for name, url in health_urls.items():
        ok = wait_for_url(url, timeout=60, interval=5)
        if ok:
            console.print(f"  [green]{name}[/green] ready")
        else:
            console.print(f"  [yellow]{name}[/yellow] not reachable")

    console.print("[cyan]Setting up coding agent sandbox...[/cyan]")
    from cogniverse_cli.sandbox import ensure_sandbox_ready

    if ensure_sandbox_ready():
        console.print("  [green]Sandbox[/green] ready")
    else:
        console.print("  [yellow]Sandbox[/yellow] disabled (coding agent execution unavailable)")

    console.print()
    _print_status_table()


@cli.command()
@click.option(
    "--keep-data",
    is_flag=True,
    default=False,
    help="Keep persistent data (only remove workloads).",
)
def down(keep_data: bool) -> None:
    """Tear down the Cogniverse stack."""
    console.print("[cyan]Removing Helm release...[/cyan]")
    helm_uninstall()

    if not keep_data:
        # Delete the cogniverse namespace to clean up PVCs and other resources
        console.print(f"[cyan]Deleting namespace {NAMESPACE}...[/cyan]")
        subprocess.run(
            ["kubectl", "delete", "namespace", NAMESPACE, "--ignore-not-found"],
            check=False,
            timeout=120,
        )

        # Delete the argo namespace
        console.print("[cyan]Deleting namespace argo...[/cyan]")
        subprocess.run(
            ["kubectl", "delete", "namespace", "argo", "--ignore-not-found"],
            check=False,
            timeout=120,
        )

        # Delete k3d cluster if one exists
        if cluster_exists():
            console.print("[cyan]Deleting k3d cluster...[/cyan]")
            delete_cluster()

    console.print("[green]Cogniverse stack removed.[/green]")


@cli.command()
def status() -> None:
    """Show status of the Cogniverse stack."""
    _print_status_table()


@cli.group()
def sandbox() -> None:
    """Manage the OpenShell sandbox gateway for the coding agent."""


@sandbox.command(name="sync")
def sandbox_sync() -> None:
    """Sync openshell gateway certs into the cluster (after rotation)."""
    from cogniverse_cli.sandbox import sync_gateway_certs_to_cluster

    if not sync_gateway_certs_to_cluster():
        raise click.ClickException("Failed to sync openshell certs")
    console.print("[green]Sandbox certs synced. Restart runtime to pick up changes.[/green]")


@sandbox.command(name="status")
def sandbox_status() -> None:
    """Show openshell gateway status and cluster sync state."""
    from cogniverse_cli.sandbox import (
        gateway_running,
        get_active_gateway_dir,
        openshell_installed,
    )

    if not openshell_installed():
        console.print("[red]openshell CLI not installed[/red]")
        return

    gateway_dir = get_active_gateway_dir()
    if gateway_dir is None:
        console.print("[yellow]No active openshell gateway[/yellow]")
        return

    console.print(f"Active gateway: [bold]{gateway_dir.name}[/bold]")
    console.print(f"  Config: {gateway_dir}")
    console.print(f"  Running: {'[green]yes[/green]' if gateway_running() else '[red]no[/red]'}")

    result = subprocess.run(
        ["kubectl", "get", "secret", "openshell-mtls", "-n", NAMESPACE],
        capture_output=True, text=True, timeout=10,
    )
    synced = result.returncode == 0
    console.print(f"  Synced to cluster: {'[green]yes[/green]' if synced else '[red]no[/red]'}")


@cli.command()
@click.option("--tenant", default=None, help="Tenant ID.")
@click.option("--language", "-l", default="python", help="Primary language.")
@click.option("--iterations", "-n", default=5, type=int, help="Max iterations.")
@click.option("--codebase", "-c", default="", help="Indexed codebase path for context.")
def code(tenant: str | None, language: str, iterations: int, codebase: str) -> None:
    """Interactive coding agent REPL."""
    from cogniverse_cli.code import run_repl

    tenant_id = tenant or os.environ.get("COGNIVERSE_TENANT_ID", "default")
    run_repl(
        tenant_id=tenant_id,
        language=language,
        max_iterations=iterations,
        codebase_path=codebase,
    )


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--type", "content_type",
    type=click.Choice(["code", "docs", "video"]),
    default="code",
    help="Content type to index.",
)
@click.option("--tenant", default=None, help="Tenant ID.")
@click.option("--profile", default=None, help="Override Vespa profile.")
def index(path: str, content_type: str, tenant: str | None, profile: str | None) -> None:
    """Index a directory into Vespa for agent context search."""
    from pathlib import Path as P

    from cogniverse_cli.index import index_files

    if content_type != "code":
        console.print(f"[yellow]--type {content_type} is not yet implemented. Only 'code' is supported.[/yellow]")
        return

    tenant_id = tenant or os.environ.get("COGNIVERSE_TENANT_ID", "default")
    index_files(
        root=P(path).resolve(),
        content_type=content_type,
        tenant_id=tenant_id,
        profile=profile,
    )


@cli.command()
@click.argument(
    "service",
    type=click.Choice(
        ["runtime", "dashboard", "vespa", "phoenix", "llm", "argo"]
    ),
)
@click.option("--follow", "-f", is_flag=True, help="Follow log output.")
def logs(service: str, follow: bool) -> None:
    """View logs for a service."""
    # Guard: if service is "llm", check if the builtin statefulset exists
    if service == "llm" and not _llm_statefulset_exists():
        console.print(
            "[yellow]No builtin LLM pod found (external LLM mode).[/yellow]"
        )
        return

    resource = _SERVICE_KUBECTL_RESOURCE[service]
    # Argo server lives in the "argo" namespace
    namespace = "argo" if service == "argo" else NAMESPACE
    cmd = ["kubectl", "logs", resource, "-n", namespace]
    if follow:
        cmd.append("-f")
    subprocess.run(cmd, check=False)
