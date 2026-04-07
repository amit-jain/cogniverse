"""Unit tests for RuntimeClient wiki/instructions/memories/jobs CRUD methods.

Audit fix #5 — runtime_client previously had no methods for /wiki/*,
/admin/tenant/*/instructions, /admin/tenant/*/memories, or
/admin/tenant/*/jobs, so even if the gateway dispatched these commands
correctly there was no way to call them. These tests verify each method
hits the right URL with the right payload.

All HTTP calls are mocked at the httpx level so the tests run without a
live runtime.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from cogniverse_messaging.runtime_client import RuntimeClient


def _make_response(status_code: int = 200, json_body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_body or {})
    resp.text = "" if json_body else "(empty)"
    return resp


@pytest.fixture
def client_with_mock_http():
    """RuntimeClient whose underlying httpx client is replaced with a mock.

    ``is_closed`` must be set to False explicitly because ``_get_client()``
    checks it and would otherwise see a truthy MagicMock and rebuild a real
    httpx client.
    """
    rc = RuntimeClient("http://runtime")
    mock_http = AsyncMock()
    mock_http.is_closed = False
    rc._client = mock_http
    return rc, mock_http


@pytest.mark.unit
@pytest.mark.ci_fast
class TestWikiCRUD:
    @pytest.mark.asyncio
    async def test_save_wiki_session_posts_correct_payload(
        self, client_with_mock_http
    ):
        rc, http = client_with_mock_http
        http.post = AsyncMock(
            return_value=_make_response(json_body={"status": "saved", "doc_id": "x"})
        )

        result = await rc.save_wiki_session(
            tenant_id="acme",
            query="test",
            response={"answer": "hi"},
            entities=["foo"],
            agent_name="search_agent",
        )

        http.post.assert_awaited_once()
        call = http.post.call_args
        assert call[0][0] == "/wiki/save"
        body = call[1]["json"]
        assert body["tenant_id"] == "acme"
        assert body["query"] == "test"
        assert body["response"] == {"answer": "hi"}
        assert body["entities"] == ["foo"]
        assert body["agent_name"] == "search_agent"
        assert result["status"] == "saved"

    @pytest.mark.asyncio
    async def test_search_wiki_posts_correct_payload(
        self, client_with_mock_http
    ):
        rc, http = client_with_mock_http
        http.post = AsyncMock(
            return_value=_make_response(json_body={"results": [], "count": 0})
        )

        result = await rc.search_wiki(tenant_id="acme", query="hi", top_k=3)

        call = http.post.call_args
        assert call[0][0] == "/wiki/search"
        assert call[1]["json"] == {
            "query": "hi",
            "tenant_id": "acme",
            "top_k": 3,
        }
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_get_wiki_topic_uses_query_param(self, client_with_mock_http):
        rc, http = client_with_mock_http
        http.get = AsyncMock(return_value=_make_response(json_body={"slug": "foo"}))

        await rc.get_wiki_topic(tenant_id="acme", slug="foo")

        call = http.get.call_args
        assert call[0][0] == "/wiki/topic/foo"
        assert call[1]["params"]["tenant_id"] == "acme"

    @pytest.mark.asyncio
    async def test_get_wiki_index_passes_tenant_id(self, client_with_mock_http):
        rc, http = client_with_mock_http
        http.get = AsyncMock(return_value=_make_response(json_body={"content": "x"}))

        await rc.get_wiki_index(tenant_id="acme")

        call = http.get.call_args
        assert call[0][0] == "/wiki/index"
        assert call[1]["params"]["tenant_id"] == "acme"

    @pytest.mark.asyncio
    async def test_lint_wiki_passes_tenant_id(self, client_with_mock_http):
        rc, http = client_with_mock_http
        http.get = AsyncMock(return_value=_make_response(json_body={"issues": []}))

        await rc.lint_wiki(tenant_id="acme")

        call = http.get.call_args
        assert call[0][0] == "/wiki/lint"
        assert call[1]["params"]["tenant_id"] == "acme"

    @pytest.mark.asyncio
    async def test_delete_wiki_topic_uses_query_param(
        self, client_with_mock_http
    ):
        rc, http = client_with_mock_http
        http.delete = AsyncMock(
            return_value=_make_response(json_body={"status": "deleted"})
        )

        await rc.delete_wiki_topic(tenant_id="acme", slug="foo")

        call = http.delete.call_args
        assert call[0][0] == "/wiki/topic/foo"
        assert call[1]["params"]["tenant_id"] == "acme"


@pytest.mark.unit
@pytest.mark.ci_fast
class TestInstructionsCRUD:
    @pytest.mark.asyncio
    async def test_set_instructions_puts_to_tenant_path(
        self, client_with_mock_http
    ):
        rc, http = client_with_mock_http
        http.put = AsyncMock(
            return_value=_make_response(json_body={"text": "x", "updated_at": "now"})
        )

        await rc.set_instructions(tenant_id="acme", text="be terse")

        call = http.put.call_args
        assert call[0][0] == "/admin/tenant/acme/instructions"
        assert call[1]["json"] == {"text": "be terse"}

    @pytest.mark.asyncio
    async def test_get_instructions_gets_from_tenant_path(
        self, client_with_mock_http
    ):
        rc, http = client_with_mock_http
        http.get = AsyncMock(
            return_value=_make_response(json_body={"text": "be terse"})
        )

        result = await rc.get_instructions(tenant_id="acme")

        call = http.get.call_args
        assert call[0][0] == "/admin/tenant/acme/instructions"
        assert result["text"] == "be terse"


@pytest.mark.unit
@pytest.mark.ci_fast
class TestMemoriesCRUD:
    @pytest.mark.asyncio
    async def test_list_memories_no_filter(self, client_with_mock_http):
        rc, http = client_with_mock_http
        http.get = AsyncMock(
            return_value=_make_response(json_body={"memories": [], "count": 0})
        )

        await rc.list_memories(tenant_id="acme")

        call = http.get.call_args
        assert call[0][0] == "/admin/tenant/acme/memories"
        assert call[1]["params"] == {}

    @pytest.mark.asyncio
    async def test_list_memories_with_agent_filter(self, client_with_mock_http):
        rc, http = client_with_mock_http
        http.get = AsyncMock(
            return_value=_make_response(json_body={"memories": [], "count": 0})
        )

        await rc.list_memories(tenant_id="acme", agent_name="search_agent")

        call = http.get.call_args
        assert call[1]["params"]["agent_name"] == "search_agent"

    @pytest.mark.asyncio
    async def test_clear_memories_with_agent_filter(self, client_with_mock_http):
        rc, http = client_with_mock_http
        http.delete = AsyncMock(
            return_value=_make_response(json_body={"status": "cleared"})
        )

        await rc.clear_memories(tenant_id="acme", agent_name="search_agent")

        call = http.delete.call_args
        assert call[0][0] == "/admin/tenant/acme/memories"
        assert call[1]["params"]["agent_name"] == "search_agent"


@pytest.mark.unit
@pytest.mark.ci_fast
class TestJobsCRUD:
    @pytest.mark.asyncio
    async def test_list_jobs_gets_from_tenant_path(self, client_with_mock_http):
        rc, http = client_with_mock_http
        http.get = AsyncMock(
            return_value=_make_response(json_body={"jobs": []})
        )

        await rc.list_jobs(tenant_id="acme")

        call = http.get.call_args
        assert call[0][0] == "/admin/tenant/acme/jobs"

    @pytest.mark.asyncio
    async def test_create_job_posts_full_payload(self, client_with_mock_http):
        rc, http = client_with_mock_http
        http.post = AsyncMock(
            return_value=_make_response(
                json_body={"job_id": "abc12345", "name": "weekly", "status": "created"}
            )
        )

        result = await rc.create_job(
            tenant_id="acme",
            name="weekly_news",
            schedule="0 9 * * 1",
            query="latest AI news",
            post_actions=["save to wiki"],
        )

        call = http.post.call_args
        assert call[0][0] == "/admin/tenant/acme/jobs"
        body = call[1]["json"]
        assert body["name"] == "weekly_news"
        assert body["schedule"] == "0 9 * * 1"
        assert body["query"] == "latest AI news"
        assert body["post_actions"] == ["save to wiki"]
        assert result["job_id"] == "abc12345"

    @pytest.mark.asyncio
    async def test_delete_job_deletes_from_tenant_path(
        self, client_with_mock_http
    ):
        rc, http = client_with_mock_http
        http.delete = AsyncMock(
            return_value=_make_response(json_body={"status": "deleted"})
        )

        await rc.delete_job(tenant_id="acme", job_id="abc12345")

        call = http.delete.call_args
        assert call[0][0] == "/admin/tenant/acme/jobs/abc12345"


@pytest.mark.unit
@pytest.mark.ci_fast
class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_non_2xx_returns_structured_error(self, client_with_mock_http):
        rc, http = client_with_mock_http
        resp = MagicMock()
        resp.status_code = 503
        resp.text = "service unavailable"
        http.get = AsyncMock(return_value=resp)

        result = await rc.list_jobs(tenant_id="acme")

        assert result["status"] == "error"
        assert result["status_code"] == 503
        assert "service unavailable" in result["message"]

    @pytest.mark.asyncio
    async def test_2xx_with_empty_body_returns_ok(self, client_with_mock_http):
        rc, http = client_with_mock_http
        resp = MagicMock()
        resp.status_code = 204
        resp.text = ""
        resp.json = MagicMock(side_effect=ValueError("no json"))
        http.delete = AsyncMock(return_value=resp)

        result = await rc.delete_job(tenant_id="acme", job_id="abc")
        assert result == {"status": "ok"}
