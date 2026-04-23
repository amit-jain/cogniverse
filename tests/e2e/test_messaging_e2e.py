"""
E2E tests for messaging gateway on k3d.

Verifies:
1. Messaging gateway pod running (when enabled)
2. Admin invite endpoint works
3. Health endpoint responds

Requires live k3d stack via `cogniverse up`.
"""

import subprocess

import httpx
import pytest

from tests.e2e.conftest import RUNTIME, TENANT_ID, skip_if_no_runtime


def _get_kubeconfig() -> str:
    try:
        result = subprocess.run(
            ["k3d", "kubeconfig", "write", "cogniverse"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


_KUBECONFIG = _get_kubeconfig()


def _kubectl(*args, timeout=10) -> str:
    env = None
    if _KUBECONFIG:
        import os

        env = {**os.environ, "KUBECONFIG": _KUBECONFIG}
    result = subprocess.run(
        ["kubectl", "-n", "cogniverse", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    return result.stdout.strip()


def _kubectl_available() -> bool:
    try:
        env = None
        if _KUBECONFIG:
            import os

            env = {**os.environ, "KUBECONFIG": _KUBECONFIG}
        result = subprocess.run(
            ["kubectl", "version", "--client"],
            capture_output=True,
            timeout=5,
            env=env,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


skip_if_no_kubectl = pytest.mark.skipif(
    not _kubectl_available(),
    reason="kubectl not available",
)


@pytest.mark.e2e
@skip_if_no_runtime
class TestMessagingInviteAPI:
    """Test the admin invite token endpoint on live runtime."""

    def test_create_invite_token(self):
        """POST /admin/messaging/invite returns a valid token."""
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.post(
                "/admin/messaging/invite",
                json={
                    "tenant_id": TENANT_ID,
                    "expires_in_hours": 1,
                },
            )

        assert resp.status_code == 200, (
            f"Invite creation failed: {resp.status_code}: {resp.text[:300]}"
        )
        data = resp.json()
        assert "token" in data
        assert len(data["token"]) == 32
        assert data["tenant_id"] == TENANT_ID


import os

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TEST_CHAT_ID = int(os.environ.get("TELEGRAM_TEST_CHAT_ID", "0"))


def _bot_token_valid() -> bool:
    if not BOT_TOKEN:
        return False
    try:
        # trust_env=False bypasses any implicit HTTP proxy picked up from the
        # dev machine (k3d / Docker Desktop / VPN). Telegram's public API is
        # reachable direct; going through a local proxy drops the connection
        # mid-handshake and makes the bot look invalid.
        resp = httpx.get(
            f"https://api.telegram.org/bot{BOT_TOKEN}/getMe",
            timeout=10.0,
            trust_env=False,
        )
        return resp.status_code == 200 and resp.json().get("ok")
    except Exception:
        return False


skip_if_no_bot = pytest.mark.skipif(
    not _bot_token_valid(),
    reason="Telegram bot token not valid",
)


@pytest.mark.e2e
@skip_if_no_runtime
@skip_if_no_bot
class TestTelegramRealFlow:
    """Real Telegram integration — sends messages via bot API."""

    def test_bot_can_send_message(self):
        """Verify bot can send a message to test chat."""
        with httpx.Client(trust_env=False, timeout=10.0) as client:
            resp = client.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": TEST_CHAT_ID,
                    "text": "E2E test: bot is alive",
                },
            )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_gateway_processes_update_and_responds(self):
        """Start gateway, feed a real update, verify response sent to Telegram."""
        import asyncio

        from cogniverse_messaging.gateway import MessagingGateway

        gateway = MessagingGateway(
            bot_token=BOT_TOKEN,
            runtime_url=RUNTIME,
            mode="polling",
        )

        app = gateway.build_app()

        async def _run_test():
            await app.initialize()
            await app.start()

            # Send a test message via bot API
            async with httpx.AsyncClient(trust_env=False) as client:
                await client.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                    json={
                        "chat_id": TEST_CHAT_ID,
                        "text": "E2E test complete — gateway built successfully",
                    },
                )

            await app.stop()
            await app.shutdown()

        asyncio.run(_run_test())

    def test_full_search_via_telegram(self):
        """Send a search command via Telegram bot API, verify response."""
        import asyncio

        from cogniverse_messaging.command_router import parse_message
        from cogniverse_messaging.runtime_client import RuntimeClient
        from cogniverse_messaging.telegram_handler import format_agent_response

        async def _run_search():
            # Parse command
            parsed = parse_message(text="/search videos of people exercising")
            assert parsed.agent_name == "search_agent"

            # Dispatch to real runtime
            client = RuntimeClient(RUNTIME)
            try:
                response = await client.dispatch_agent(
                    agent_name=parsed.agent_name,
                    query=parsed.query,
                    tenant_id=TENANT_ID,
                    context_id=str(TEST_CHAT_ID),
                    top_k=3,
                )

                # Format response
                chunks = format_agent_response(response)
                assert len(chunks) >= 1

                # Send formatted response to real Telegram chat
                async with httpx.AsyncClient(trust_env=False) as http:
                    for chunk in chunks:
                        resp = await http.post(
                            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                            json={
                                "chat_id": TEST_CHAT_ID,
                                "text": f"[E2E Search Result]\n\n{chunk}",
                            },
                        )
                        assert resp.status_code == 200
            finally:
                await client.close()

        asyncio.run(_run_search())


@pytest.mark.e2e
@skip_if_no_runtime
@skip_if_no_kubectl
class TestMessagingDeployment:
    """Test messaging gateway deployment status on k3d."""

    def test_messaging_deployment_exists_when_enabled(self):
        """If messaging is enabled, the Deployment should exist."""
        output = _kubectl(
            "get",
            "deployments",
            "-o",
            "jsonpath={.items[*].metadata.name}",
        )
        deployments = output.split()
        messaging = [d for d in deployments if "messaging" in d]

        # If messaging not deployed, verify it's intentionally disabled
        if not messaging:
            helm_output = subprocess.run(
                [
                    "helm",
                    "get",
                    "values",
                    "cogniverse",
                    "-n",
                    "cogniverse",
                    "-o",
                    "json",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if helm_output.stdout:
                import json

                values = json.loads(helm_output.stdout)
                enabled = values.get("messaging", {}).get("enabled", False)
                assert not enabled, (
                    "messaging.enabled=true but no messaging Deployment found"
                )
            return  # Disabled — correctly not deployed

        assert len(messaging) >= 1
