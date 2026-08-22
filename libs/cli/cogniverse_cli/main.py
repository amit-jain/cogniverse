"""Cogniverse CLI entrypoint.

Provides ``up``, ``down``, ``status``, and ``logs`` commands for
deploying and managing the multi-agent RAG stack.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.table import Table

from cogniverse_cli.argo import deploy_workflow_templates, install_argo_controller
from cogniverse_cli.cluster import (
    CLUSTER_NAME,
    ClusterStartError,
    check_prerequisites,
    cluster_exists,
    create_cluster,
    delete_cluster,
    get_install_commands,
    has_existing_k8s,
    install_missing_prerequisites,
    list_cluster_states,
    start_cluster,
    start_port_forwards,
    stop_cluster,
    stop_port_forwards,
)
from cogniverse_cli.config import (
    get_chart_path,
    get_device_values_file,
    get_values_file,
    get_workflows_path,
    resolve_project_root,
)
from cogniverse_cli.constants import NAMESPACE
from cogniverse_cli.deploy import helm_install, helm_uninstall
from cogniverse_cli.health import check_service_health, wait_for_url
from cogniverse_cli.images import (
    build_images,
    detect_torch_backend,
    dev_image_set_values,
    dev_version,
    has_workspace_source,
    import_images,
    prune_superseded_images,
    pull_and_import_third_party,
    verify_local_images_cover_deploy,
)

console = Console()


def _build_modal_inference_lifecycle():
    """Build the Modal lifecycle with the process-scoped inference key."""

    from cogniverse_cli.inference_endpoints import EndpointCredentials
    from cogniverse_cli.modal_inference_lifecycle import ModalInferenceLifecycle

    return ModalInferenceLifecycle(
        credentials=EndpointCredentials(
            bearer_token=os.environ.get("COGNIVERSE_INFERENCE_API_KEY")
        )
    )


def _resolve_cli_tenant(tenant: str | None) -> str:
    """Pick the tenant_id to use for a CLI command.

    Order:
      1. Explicit ``--tenant`` flag.
      2. ``$COGNIVERSE_TENANT_ID`` env var.
      3. Error out with a clear pointer.

    The runtime no longer falls back to a "default" tenant — every request
    needs an explicit tenant_id that maps to a registered tenant. CLI
    callers can register one with ``cogniverse tenant create <id>`` (when
    that command is available) or POST /admin/tenants directly.
    """
    tid = tenant or os.environ.get("COGNIVERSE_TENANT_ID")
    if tid:
        return tid
    raise click.ClickException(
        "No tenant configured. Pass --tenant <id>, set "
        "$COGNIVERSE_TENANT_ID, or register a tenant via the runtime: "
        "curl -X POST http://localhost:28000/admin/tenants -H "
        "'Content-Type: application/json' -d '{\"tenant_id\":\"<id>\"}'"
    )


def _cluster_name_from_probe(cluster_probe: object) -> str | None:
    """Normalize a cluster probe to a concrete cluster name."""
    if isinstance(cluster_probe, str):
        return cluster_probe
    if cluster_probe:
        return CLUSTER_NAME
    return None


SERVICE_HEALTH_URLS: dict[str, str] = {
    "Vespa": "http://localhost:19071/state/v1/health",
    "Runtime": "http://localhost:28000/health",
    "Dashboard": "http://localhost:28501/_stcore/health",
    "Phoenix": "http://localhost:26006/health",
    "LLM": "http://localhost:11434/api/tags",
    "Argo": "https://localhost:2746/api/v1/info",
}

SERVICE_ENDPOINTS: dict[str, str] = {
    "Vespa": "http://localhost:8080",
    "Runtime": "http://localhost:28000",
    "Dashboard": "http://localhost:28501",
    "Phoenix": "http://localhost:26006",
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


def _probe_host_llm(base: str = "http://localhost:11434") -> bool:
    """Return True if any OAI-compatible LM endpoint responds at *base*.

    Probes the native ``/api/tags`` listing endpoint (common on local LM
    servers) and falls back to ``/v1/models`` (pure OAI-compat). Either
    returning HTTP 200 within 3 seconds is treated as "local LM is up".
    """
    base = base.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    for path in ("/api/tags", "/v1/models"):
        try:
            resp = httpx.get(f"{base}{path}", timeout=3)
            if resp.status_code == 200:
                return True
        except (httpx.HTTPError, OSError):
            continue
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


def _run_kubectl_logs(cmd: list[str]) -> int:
    """Run kubectl logs, mapping a missing binary to a clear exit."""
    try:
        result = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        console.print("[red]kubectl not found on PATH[/red]")
        return 127
    return result.returncode


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
@click.option(
    "--sandbox",
    type=click.Choice(["in-cluster", "external", "off"]),
    default="in-cluster",
    help=(
        "Coding-agent sandbox mode. 'in-cluster' self-hosts the OpenShell "
        "gateway + agent-sandbox operator (default, portable). 'external' "
        "points at a managed gateway (set runtime.sandbox.external.endpoint). "
        "'off' disables the coding agent."
    ),
)
@click.option(
    "--sandbox-endpoint",
    default=None,
    help="External OpenShell gateway endpoint (host:port) for --sandbox external.",
)
def up(
    llm_mode: str,
    llm_url: str | None,
    image_source: str | None,
    messaging: bool,
    sandbox: str,
    sandbox_endpoint: str | None,
) -> None:
    """Deploy the full Cogniverse stack."""
    # 1. Detect environment — a running k3d cluster counts as local, not prod
    try:
        k3d_cluster_name = _cluster_name_from_probe(cluster_exists())
    except ClusterStartError as exc:
        console.print(f"[red]{exc}[/red]", soft_wrap=True)
        sys.exit(1)
    k3d_running = bool(k3d_cluster_name)
    if k3d_running:
        use_k3d = True
    else:
        existing_k8s = has_existing_k8s()
        use_k3d = not existing_k8s
    resolved_cluster_name = k3d_cluster_name or CLUSTER_NAME

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
            console.print(
                "[red]Please install manually using the commands above.[/red]"
            )
            sys.exit(1)
        console.print("[green]Prerequisites installed[/green]")

    # 3. Detect LLM BEFORE creating cluster (cluster binds port 11434)
    host_llm_detected = False
    if llm_mode == "auto" and use_k3d and not k3d_running:
        host_llm_detected = _probe_host_llm()
        if host_llm_detected:
            console.print("[cyan]Detected host LLM, will use external mode.[/cyan]")

    # 4. Resolve project root (needed for cluster volume mount and image build)
    project_root = resolve_project_root()
    build_root = Path(image_source).resolve() if image_source else project_root
    if image_source and not has_workspace_source(build_root):
        console.print(
            f"[red]--image-source {image_source} has no buildable workspace "
            f"source (libs/runtime not found).[/red]"
        )
        sys.exit(1)

    # 5. Create k3d cluster if needed (local mode only)
    if use_k3d:
        if not k3d_running:
            console.print("[cyan]Creating k3d cluster...[/cyan]")
            # Exclude LLM port if host LLM is running (avoids port conflict)
            exclude = [11434] if host_llm_detected else None
            try:
                create_cluster(
                    name=resolved_cluster_name,
                    exclude_ports=exclude,
                    workspace_path=str(project_root) if project_root else None,
                )
            except ClusterStartError as exc:
                console.print(f"[red]{exc}[/red]", soft_wrap=True)
                sys.exit(1)
        else:
            console.print("[cyan]Using existing k3d cluster.[/cyan]")
        # Label the node with amd.com/gpu.present / nvidia.com/gpu.present
        # so the chart's nodeSelector schedules GPU pods.
        host_backend_for_label = detect_torch_backend()
        if host_backend_for_label == "rocm":
            subprocess.run(
                [
                    "kubectl",
                    "label",
                    "node",
                    "--all",
                    "amd.com/gpu.present=true",
                    "--overwrite",
                ],
                capture_output=True,
                timeout=30,
                check=False,
            )
        elif host_backend_for_label == "cuda":
            subprocess.run(
                [
                    "kubectl",
                    "label",
                    "node",
                    "--all",
                    "nvidia.com/gpu.present=true",
                    "--overwrite",
                ],
                capture_output=True,
                timeout=30,
                check=False,
            )
    # Compose the deploy values (device overrides on the k3s/prod base) up
    # front so the image build can gate optional sidecars on
    # inference.<svc>.enabled.
    chart_path = get_chart_path()
    base_values_file = get_values_file(prod=not use_k3d)
    values_files: list[Path] = [base_values_file]
    if use_k3d:
        host_backend = detect_torch_backend()
        device_values = get_device_values_file(host_backend)
        if device_values is not None:
            values_files.append(device_values)
            console.print(
                f"[cyan]Composing device overrides:[/cyan] {device_values.name}"
            )

    dev_image_overrides: dict[str, str] = {}
    image_version: str | None = None
    if build_root and has_workspace_source(build_root):
        console.print("[cyan]Building container images...[/cyan]")
        # Derive the git version once so the built image tag, the deployed
        # --set override, and the stamped chart version are all identical.
        image_version = dev_version(build_root)
        tags = build_images(
            build_root, values_files=values_files, version=image_version
        )
        if use_k3d:
            console.print("[cyan]Importing images into k3d...[/cyan]")
            import_images(resolved_cluster_name, tags)
        # Reclaim the superseded generation's ~25GB of images (host + k3d
        # node) so repeated deploys don't fill the disk into Vespa's feed
        # block; keeps the current build and one previous for rollback.
        try:
            prune_superseded_images(
                image_version,
                node_container=(
                    f"k3d-{resolved_cluster_name}-server-0" if use_k3d else None
                ),
            )
        except Exception as exc:
            console.print(f"[yellow]Image prune skipped: {exc}[/yellow]")
        dev_image_overrides = dev_image_set_values(
            project_root, values_files=values_files, version=image_version
        )
        verify_local_images_cover_deploy(
            build_root, values_files, built_tags=tags, version=image_version
        )

    # 6. Detect LLM mode and build Helm set_values overrides. The dev-image
    # overrides point the chart at the git-tagged images just built + imported.
    set_values: dict[str, str] = dict(dev_image_overrides)
    if llm_mode == "auto":
        if host_llm_detected or _probe_host_llm():
            console.print("[cyan]Using host LLM endpoint (external mode).[/cyan]")
            external_url = (
                "http://host.k3d.internal:11434"
                if use_k3d
                else "http://localhost:11434"
            )
            # Don't override llm.engine — the chart helper renders the
            # canonical litellm model id (and api_base) per engine choice.
            set_values["llm.builtin.enabled"] = "false"
            set_values["llm.external.enabled"] = "true"
            set_values["llm.external.url"] = external_url
        else:
            console.print(
                "[cyan]No local LM detected on :11434, using builtin LLM.[/cyan]"
            )
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
        set_values["llm.engine"] = "external"
        set_values["llm.builtin.enabled"] = "false"
        set_values["llm.external.enabled"] = "true"
        set_values["llm.external.url"] = resolved_url
    # llm_mode == "builtin" requires no overrides (chart defaults)

    # Messaging gateway
    if messaging:
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not bot_token:
            console.print("[red]--messaging requires TELEGRAM_BOT_TOKEN env var.[/red]")
            sys.exit(1)
        set_values["messaging.enabled"] = "true"
        console.print("[cyan]Messaging gateway enabled (Telegram).[/cyan]")

    # 5a. Device overrides were composed above (before the image build); keep
    # the llm-external flag for the third-party pre-pull below.
    llm_is_external = set_values.get("llm.external.enabled") == "true"

    # 5b. Pre-pull third-party images from every values file so GB-scale
    # GPU images don't blow the helm-install timeout at pod start.
    if use_k3d:
        console.print("[cyan]Pre-pulling third-party images...[/cyan]")
        for vf in values_files:
            pull_and_import_third_party(
                resolved_cluster_name,
                vf,
                skip_llm=llm_is_external,
            )
    # 5c. Bootstrap secrets the chart references by name. Must happen
    # BEFORE helm install so gated-model pods (e.g. the vLLM Gemma LLM
    # student/teacher pods) find hf-token at startup.
    from cogniverse_cli.secrets import (
        sync_hf_token_to_cluster,
        sync_inference_api_key_to_cluster,
    )

    console.print("[cyan]Syncing cluster secrets...[/cyan]")
    sync_hf_token_to_cluster(required=False)
    sync_inference_api_key_to_cluster(required=False)

    # 6. Install Argo controller FIRST so its CRDs exist before the main chart
    # renders CronWorkflow/WorkflowTemplate manifests. The main chart then opts
    # out of sub-chart CRD install to avoid ownership conflicts.
    console.print("[cyan]Installing Argo Workflows controller...[/cyan]")
    install_argo_controller()
    console.print("[green]Argo Workflows installed[/green]")

    # 7. Wire the coding-agent sandbox. In-cluster (default) self-hosts the
    # OpenShell gateway + agent-sandbox operator via the chart — no host
    # dependency, reproducible on any cluster. External points at a managed
    # gateway. Off disables the coding agent.
    subprocess.run(
        ["kubectl", "create", "namespace", NAMESPACE],
        capture_output=True,
        timeout=10,
        check=False,
    )
    if sandbox == "in-cluster":
        set_values["runtime.sandbox.enabled"] = "true"
        set_values["runtime.sandbox.inCluster.enabled"] = "true"
        console.print("  [green]Coding sandbox[/green]: in-cluster (self-hosted)")
    elif sandbox == "external":
        endpoint = sandbox_endpoint or ""
        if not endpoint:
            console.print(
                "[red]--sandbox external requires --sandbox-endpoint host:port[/red]"
            )
            sys.exit(1)
        set_values["runtime.sandbox.enabled"] = "true"
        set_values["runtime.sandbox.external.enabled"] = "true"
        set_values["runtime.sandbox.external.endpoint"] = endpoint
        console.print(f"  [green]Coding sandbox[/green]: external gateway ({endpoint})")
    else:
        set_values["runtime.sandbox.enabled"] = "false"
        console.print("  [yellow]Coding sandbox[/yellow]: disabled")

    # 8. Deploy the main Helm release. Backend mirrors host detection so
    # the chart picks the matching imagesByBackend entry instead of the
    # default cpu (which would ErrImageNeverPull on a non-cpu host).
    set_values["argo-workflows.crds.install"] = "false"
    if use_k3d:
        set_values["runtime.backend"] = host_backend
        set_values["dashboard.backend"] = host_backend
    console.print("[cyan]Deploying Helm release...[/cyan]")
    helm_install(
        chart_path,
        values_files,
        set_values=set_values or None,
        chart_version=image_version,
    )
    console.print("[green]Helm release deployed[/green]")

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
            [
                "kubectl",
                "wait",
                "--for=condition=ready",
                "pod",
                "-l",
                "app.kubernetes.io/instance=cogniverse",
                "-n",
                NAMESPACE,
                "--timeout=300s",
            ],
            check=True,
            capture_output=True,
            timeout=310,
        )
        console.print("[green]All cogniverse pods ready[/green]")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", NAMESPACE],
            capture_output=True,
            text=True,
            timeout=10,
        )
        console.print(f"[yellow]Some pods not ready:[/yellow]\n{result.stdout}")

    # 10. Bridge services with no NodePort (argo runs in its own namespace) to
    # localhost. Reaps any daemon a prior up left so repeated runs don't orphan
    # a restart-loop still retrying its bind.
    console.print("[cyan]Starting service port-forwards...[/cyan]")
    start_port_forwards()

    # 11. Verify services via HTTP health checks
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
    console.print("[cyan]Stopping service port-forwards...[/cyan]")
    stop_port_forwards()

    console.print("[cyan]Removing Helm release...[/cyan]")
    helm_uninstall()

    if keep_data:
        console.print("[green]Cogniverse stack removed.[/green]")
        return

    # kubectl delete rc was previously ignored, so a forbidden/unreachable
    # delete still printed "stack removed" and exited 0 — invisible to any
    # script wrapping `cogniverse down`. Surface each failure and exit nonzero.
    failed: list[str] = []
    for ns in (NAMESPACE, "argo"):
        console.print(f"[cyan]Deleting namespace {ns}...[/cyan]")
        result = subprocess.run(
            ["kubectl", "delete", "namespace", ns, "--ignore-not-found"],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            detail = (result.stderr or "").strip() or f"exit {result.returncode}"
            console.print(f"[red]Failed to delete namespace {ns}: {detail}[/red]")
            failed.append(ns)

    # Delete k3d cluster if one exists
    try:
        cluster_name = _cluster_name_from_probe(cluster_exists())
    except ClusterStartError as exc:
        console.print(f"[red]{exc}[/red]", soft_wrap=True)
        raise SystemExit(1) from None
    if cluster_name:
        console.print("[cyan]Deleting k3d cluster...[/cyan]")
        delete_cluster(cluster_name)

    if failed:
        raise SystemExit(
            f"Teardown incomplete — namespace delete failed: {', '.join(failed)}"
        )

    console.print("[green]Cogniverse stack removed.[/green]")


@cli.command()
def status() -> None:
    """Show status of the Cogniverse stack."""
    try:
        clusters = list_cluster_states()
    except Exception as exc:  # noqa: BLE001 — degrade, but say why
        # Rendering a docker/k3d outage as "no clusters" hid the actual
        # problem — an operator cannot tell "nothing deployed" from
        # "docker daemon down".
        console.print(f"[yellow]Could not list k3d clusters: {exc}[/yellow]")
        clusters = []
    if clusters:
        cluster_table = Table(title="k3d Clusters")
        cluster_table.add_column("Cluster", style="bold")
        cluster_table.add_column("State")
        for cluster in clusters:
            running = cluster["servers_running"] >= max(cluster["servers_count"], 1)
            state = "[green]running[/green]" if running else "[yellow]stopped[/yellow]"
            cluster_table.add_row(cluster["name"], state)
        console.print(cluster_table)
    _print_status_table()


@cli.command()
@click.option(
    "--name",
    default=None,
    help="k3d cluster to stop. Omit to resolve the active cogniverse* cluster.",
)
def stop(name: str | None) -> None:
    """Stop a cluster's containers, keeping all data (frees RAM/GPU)."""
    resolved_name = name
    if resolved_name is None:
        try:
            resolved_name = _cluster_name_from_probe(cluster_exists())
        except ClusterStartError as exc:
            console.print(f"[red]{exc}[/red]", soft_wrap=True)
            raise SystemExit(1) from None
        if resolved_name is None:
            console.print("[red]No cogniverse k3d cluster found.[/red]")
            raise SystemExit(1)
    if not cluster_exists(resolved_name):
        console.print(f"[red]No k3d cluster named {resolved_name!r}.[/red]")
        raise SystemExit(1)
    console.print(f"[cyan]Stopping cluster {resolved_name}...[/cyan]")
    stop_cluster(resolved_name)
    if resolved_name == CLUSTER_NAME:
        # The dev stack's kubectl port-forwards are dead once the cluster
        # halts; reap their restart-loops so a later start rebinds cleanly.
        stop_port_forwards()
    console.print(
        f"[green]Cluster {resolved_name} stopped — data preserved; "
        f"resume with `cogniverse start --name {resolved_name}`.[/green]"
    )


@cli.command()
@click.option(
    "--name",
    default=None,
    help="k3d cluster to start. Omit to resolve the active cogniverse* cluster.",
)
def start(name: str | None) -> None:
    """Start a previously stopped cluster (volumes intact)."""
    resolved_name = name
    if resolved_name is None:
        try:
            resolved_name = _cluster_name_from_probe(cluster_exists())
        except ClusterStartError as exc:
            console.print(f"[red]{exc}[/red]", soft_wrap=True)
            raise SystemExit(1) from None
        if resolved_name is None:
            console.print("[red]No cogniverse k3d cluster found.[/red]")
            raise SystemExit(1)
    if not cluster_exists(resolved_name):
        console.print(f"[red]No k3d cluster named {resolved_name!r}.[/red]")
        raise SystemExit(1)
    console.print(f"[cyan]Starting cluster {resolved_name}...[/cyan]")
    try:
        start_cluster(resolved_name)
    except ClusterStartError as exc:
        console.print(f"[red]{exc}[/red]", soft_wrap=True)
        raise SystemExit(1) from None
    if resolved_name == CLUSTER_NAME:
        # The dev stack is reached through kubectl port-forwards (29xxx);
        # the e2e cluster maps NodePorts directly (33xxx) and needs none.
        try:
            start_port_forwards()
        except Exception as exc:
            console.print(
                f"[yellow]Cluster started but port-forwards failed ({exc}); "
                "re-run `cogniverse start` once pods are ready.[/yellow]"
            )
    console.print(f"[green]Cluster {resolved_name} started.[/green]")


@cli.group()
def inference() -> None:
    """Manage external inference services."""


@inference.group(name="modal")
def inference_modal() -> None:
    """Manage canonical Modal inference deployments."""


def _modal_status_operation(operation: str, services: tuple[str, ...]) -> None:
    from cogniverse_cli.modal_inference_lifecycle import ModalLifecycleError

    try:
        with _build_modal_inference_lifecycle() as lifecycle:
            statuses = getattr(lifecycle, operation)(services)
    except (ModalLifecycleError, ValueError) as exc:
        raise click.ClickException(str(exc)) from None
    for status in statuses:
        click.echo(
            f"{status.service}: {status.web_url} "
            f"(active_containers={status.active_containers})"
        )


@inference_modal.command(name="deploy")
@click.argument("services", nargs=-1, required=True)
def inference_modal_deploy(services: tuple[str, ...]) -> None:
    """Deploy one or more canonical Modal services."""

    _modal_status_operation("deploy", services)


@inference_modal.command(name="warm")
@click.argument("services", nargs=-1, required=True)
def inference_modal_warm(services: tuple[str, ...]) -> None:
    """Warm services and verify their authenticated model contracts."""

    from cogniverse_cli.modal_inference_lifecycle import ModalLifecycleError

    try:
        with _build_modal_inference_lifecycle() as lifecycle:
            endpoints = lifecycle.warm(services)
            statuses = {status.service: status for status in lifecycle.status(services)}
    except (ModalLifecycleError, ValueError) as exc:
        raise click.ClickException(str(exc)) from None
    for endpoint in endpoints:
        status = statuses[endpoint.service]
        click.echo(
            f"{endpoint.service}: {endpoint.base_url} "
            f"(model={endpoint.model_id}, "
            f"active_containers={status.active_containers})"
        )


@inference_modal.command(name="release")
@click.argument("services", nargs=-1, required=True)
def inference_modal_release(services: tuple[str, ...]) -> None:
    """Return services to scale-to-zero without stopping their apps."""

    _modal_status_operation("release", services)


@inference_modal.command(name="status")
@click.argument("services", nargs=-1, required=True)
def inference_modal_status(services: tuple[str, ...]) -> None:
    """Show Modal endpoint and live runner count."""

    _modal_status_operation("status", services)


@inference_modal.command(name="qualify")
@click.argument("service")
@click.option(
    "--gpu",
    "gpu_candidates",
    multiple=True,
    required=True,
    help="Candidate Modal GPU type; repeat in any order.",
)
def inference_modal_qualify(
    service: str,
    gpu_candidates: tuple[str, ...],
) -> None:
    """Choose the earliest configured GPU from supplied candidates."""

    from cogniverse_cli.modal_inference_lifecycle import ModalLifecycleError

    try:
        with _build_modal_inference_lifecycle() as lifecycle:
            result = lifecycle.qualify(service, gpu_candidates)
    except (ModalLifecycleError, ValueError) as exc:
        raise click.ClickException(str(exc)) from None
    click.echo(
        f"{result.service}: selected {result.selected_gpu} from "
        f"{', '.join(result.considered_gpus)}"
    )


@inference_modal.command(name="undeploy")
@click.argument("service")
@click.option(
    "--confirm-service",
    required=True,
    help="Exact service name required before destructive undeploy.",
)
def inference_modal_undeploy(service: str, confirm_service: str) -> None:
    """Permanently stop a Modal app after exact confirmation."""

    from cogniverse_cli.modal_inference_lifecycle import ModalLifecycleError

    try:
        with _build_modal_inference_lifecycle() as lifecycle:
            lifecycle.undeploy(service, confirm_service)
    except (ModalLifecycleError, ValueError) as exc:
        raise click.ClickException(str(exc)) from None
    click.echo(f"{service}: undeployed")


@cli.group()
def graph() -> None:
    """Query the knowledge graph built by `cogniverse index`."""


@graph.command(name="stats")
@click.option("--tenant", default=None, help="Tenant ID.")
def graph_stats(tenant: str | None) -> None:
    """Show graph statistics: node/edge counts and top-degree nodes."""
    from cogniverse_cli.graph import cmd_stats

    tenant_id = _resolve_cli_tenant(tenant)
    code = cmd_stats(tenant_id)
    if code:
        raise SystemExit(code)


@graph.command(name="search")
@click.argument("query")
@click.option("--tenant", default=None, help="Tenant ID.")
@click.option("-k", "--top-k", default=10, type=int, help="Max results.")
def graph_search(query: str, tenant: str | None, top_k: int) -> None:
    """Semantic search over graph nodes."""
    from cogniverse_cli.graph import cmd_search

    tenant_id = _resolve_cli_tenant(tenant)
    code = cmd_search(tenant_id, query, top_k=top_k)
    if code:
        raise SystemExit(code)


@graph.command(name="neighbors")
@click.argument("node")
@click.option("--tenant", default=None, help="Tenant ID.")
@click.option("-d", "--depth", default=1, type=int, help="Traversal depth (1-3).")
def graph_neighbors(node: str, tenant: str | None, depth: int) -> None:
    """Show direct neighbors of a node."""
    from cogniverse_cli.graph import cmd_neighbors

    tenant_id = _resolve_cli_tenant(tenant)
    code = cmd_neighbors(tenant_id, node, depth=depth)
    if code:
        raise SystemExit(code)


@graph.command(name="path")
@click.argument("source")
@click.argument("target")
@click.option("--tenant", default=None, help="Tenant ID.")
@click.option("-d", "--max-depth", default=4, type=int, help="Max path depth.")
def graph_path(source: str, target: str, tenant: str | None, max_depth: int) -> None:
    """Find the shortest path between two nodes."""
    from cogniverse_cli.graph import cmd_path

    tenant_id = _resolve_cli_tenant(tenant)
    code = cmd_path(tenant_id, source, target, max_depth=max_depth)
    if code:
        raise SystemExit(code)


@cli.group()
def secrets() -> None:
    """Manage cluster secrets referenced by the Helm chart."""


@secrets.command(name="sync")
@click.option(
    "--required/--optional",
    default=False,
    help="Fail if the token is missing instead of warning.",
)
def secrets_sync(required: bool) -> None:
    """Re-sync cluster Secrets from local credentials.

    hf-token comes from HF_TOKEN, HUGGING_FACE_HUB_TOKEN, or
    ~/.cache/huggingface/token (populated by `huggingface-cli login`).
    The messaging Secret holding the Telegram bot token comes from
    TELEGRAM_BOT_TOKEN or .env/TELEGRAM_BOT_TOKEN.env. The inference
    Secret holding the Modal bearer key comes from
    COGNIVERSE_INFERENCE_API_KEY or .env/COGNIVERSE_INFERENCE_API_KEY.env.
    All are applied into the cogniverse namespace.
    """
    from cogniverse_cli.secrets import (
        sync_hf_token_to_cluster,
        sync_inference_api_key_to_cluster,
        sync_telegram_token_to_cluster,
    )

    hf_ok = sync_hf_token_to_cluster(required=required)
    # Messaging and Modal-hosted inference are optional in the chart, so a
    # missing bot token or bearer key is only fatal when --required asks for
    # a fully-provisioned cluster.
    telegram_ok = sync_telegram_token_to_cluster(required=required)
    inference_ok = sync_inference_api_key_to_cluster(required=required)

    if required and not (hf_ok and telegram_ok and inference_ok):
        failed = ", ".join(
            name
            for name, ok in (
                ("hf-token", hf_ok),
                ("telegram-bot-token", telegram_ok),
                ("cogniverse-inference-api-key", inference_ok),
            )
            if not ok
        )
        raise click.ClickException(f"Failed to sync: {failed}")


@cli.group()
def admin() -> None:
    """Admin-side operations: tenant lifecycle, orphan reconciliation."""


@admin.command(name="reconcile-orphans")
@click.option(
    "--confirm",
    is_flag=True,
    default=False,
    help=(
        "Actually drop the orphan schemas. Default is dry-run (list only). "
        "All orphan tenants are dropped in one Vespa redeploy."
    ),
)
@click.option(
    "--runtime-url",
    default="http://localhost:28000",
    show_default=True,
    help="Runtime endpoint to call /admin/reconcile-orphans on.",
)
def admin_reconcile_orphans(confirm: bool, runtime_url: str) -> None:
    """Find and drop Vespa-only schema orphans.

    Diffs the deployed Vespa schemas against the SchemaRegistry's active
    set. Anything in Vespa but not in the registry is an orphan from an
    interrupted deploy path. Default mode (dry-run) lists them; pass
    --confirm to drop them all in one atomic redeploy.
    """
    from cogniverse_cli.admin import run

    run(runtime_url, confirm=confirm)


@admin.command(name="invite")
@click.argument("tenant_id")
@click.option(
    "--expires-in-hours",
    default=24,
    show_default=True,
    type=int,
    help="How long the token stays valid.",
)
@click.option(
    "--runtime-url",
    default="http://localhost:28000",
    show_default=True,
    help="Runtime endpoint to call /admin/messaging/invite on.",
)
def admin_invite(tenant_id: str, expires_in_hours: int, runtime_url: str) -> None:
    """Mint a messaging invite token for TENANT_ID.

    The user sends the printed ``/start <token>`` to the bot to link their
    chat account to this tenant. Tokens are single-use and expire.
    """
    from cogniverse_cli.admin import run_invite

    run_invite(runtime_url, tenant_id, expires_in_hours=expires_in_hours)


@cli.group()
def sandbox() -> None:
    """Manage the OpenShell sandbox gateway for the coding agent."""


@sandbox.command(name="sync")
def sandbox_sync() -> None:
    """Sync openshell gateway certs into the cluster (after rotation)."""
    from cogniverse_cli.sandbox import sync_gateway_certs_to_cluster

    if not sync_gateway_certs_to_cluster():
        raise click.ClickException("Failed to sync openshell certs")
    console.print(
        "[green]Sandbox certs synced. Restart runtime to pick up changes.[/green]"
    )


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
    console.print(
        f"  Running: {'[green]yes[/green]' if gateway_running() else '[red]no[/red]'}"
    )

    try:
        result = subprocess.run(
            ["kubectl", "get", "secret", "openshell-mtls", "-n", NAMESPACE],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        console.print(f"  Synced to cluster: [red]unknown (kubectl: {exc})[/red]")
        return
    stderr = (result.stderr or "").lower()
    if result.returncode == 0:
        console.print("  Synced to cluster: [green]yes[/green]")
    elif "not found" in stderr or "notfound" in stderr:
        console.print("  Synced to cluster: [red]no[/red]")
    else:
        detail = (result.stderr or "").strip() or f"exit {result.returncode}"
        console.print(f"  Synced to cluster: [yellow]unknown ({detail})[/yellow]")


@cli.command()
@click.option("--tenant", default=None, help="Tenant ID.")
@click.option("--language", "-l", default="python", help="Primary language.")
@click.option("--iterations", "-n", default=5, type=int, help="Max iterations.")
@click.option("--codebase", "-c", default="", help="Indexed codebase path for context.")
def code(tenant: str | None, language: str, iterations: int, codebase: str) -> None:
    """Interactive coding agent REPL."""
    from cogniverse_cli.code import run_repl

    tenant_id = _resolve_cli_tenant(tenant)
    run_repl(
        tenant_id=tenant_id,
        language=language,
        max_iterations=iterations,
        codebase_path=codebase,
    )


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--type",
    "content_type",
    type=click.Choice(["code", "docs", "video"]),
    default="code",
    help="Content type to index.",
)
@click.option("--tenant", default=None, help="Tenant ID.")
@click.option("--profile", default=None, help="Override Vespa profile.")
@click.option(
    "--gliner-url",
    default=None,
    help=(
        "GLiNER inference service used for text graph extraction "
        "(default: the URL in system configuration)."
    ),
)
def index(
    path: str,
    content_type: str,
    tenant: str | None,
    profile: str | None,
    gliner_url: str | None,
) -> None:
    """Index a directory into Vespa for agent context search."""
    from pathlib import Path as P

    from cogniverse_cli.index import index_files

    if content_type == "video":
        console.print(
            "[yellow]--type video is not yet implemented. "
            "Use 'code' or 'docs'.[/yellow]"
        )
        return

    tenant_id = _resolve_cli_tenant(tenant)
    index_files(
        root=P(path).resolve(),
        content_type=content_type,
        tenant_id=tenant_id,
        profile=profile,
        gliner_url=gliner_url,
    )


@cli.command()
@click.argument(
    "service",
    type=click.Choice(["runtime", "dashboard", "vespa", "phoenix", "llm", "argo"]),
)
@click.option("--follow", "-f", is_flag=True, help="Follow log output.")
def logs(service: str, follow: bool) -> None:
    """View logs for a service."""
    # Guard: if service is "llm", check if the builtin statefulset exists
    if service == "llm" and not _llm_statefulset_exists():
        console.print("[yellow]No builtin LLM pod found (external LLM mode).[/yellow]")
        return

    resource = _SERVICE_KUBECTL_RESOURCE[service]
    # Argo server lives in the "argo" namespace
    namespace = "argo" if service == "argo" else NAMESPACE
    cmd = ["kubectl", "logs", resource, "-n", namespace]
    if follow:
        cmd.append("-f")
    # Propagate kubectl's exit code — a NotFound previously exited 0, so
    # scripts wrapping `cogniverse logs` could not detect failure.
    code = _run_kubectl_logs(cmd)
    if code:
        raise SystemExit(code)
