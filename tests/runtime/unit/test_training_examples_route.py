"""The training-example upload route refuses an upload before storing any of it."""

import asyncio

import httpx
import pytest
from fastapi import FastAPI

from cogniverse_runtime.routers import training_examples
from cogniverse_synthetic.approval.uploads import (
    MAX_UPLOADED_EXAMPLES,
    upload_templates,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

EXAMPLE = {
    "query": "kiln firing",
    "enhanced_query": "stoneware kiln firing schedule",
    "reasoning": "names the ware and the schedule",
}


def _call(method: str, path: str, body=None) -> httpx.Response:
    app = FastAPI()
    app.include_router(training_examples.router, prefix="/admin/tenant")

    async def call():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://runtime"
        ) as client:
            return await client.request(method, path, json=body)

    return asyncio.run(call())


def _upload(body) -> httpx.Response:
    return _call("POST", "/admin/tenant/acme:prod/training-examples", body)


def test_the_templates_are_each_optimizers_schema_and_the_upload_limit():
    response = _call("GET", "/admin/tenant/training-example-templates")
    assert (response.status_code, response.json()) == (
        200,
        {"templates": upload_templates(), "max_examples": MAX_UPLOADED_EXAMPLES},
    )


def test_every_invalid_example_is_named_in_the_refusal():
    response = _upload(
        {
            "optimizer": "query_enhancement",
            "reviewer": "operator@example.com",
            "examples": [EXAMPLE, {"query": "kiln"}],
        }
    )
    assert (response.status_code, response.json()) == (
        400,
        {
            "detail": {
                "message": "1 of 2 QueryEnhancementExampleSchema examples are invalid",
                "errors": [
                    "examples[1].enhanced_query: Field required",
                    "examples[1].reasoning: Field required",
                ],
            }
        },
    )


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        (
            {"optimizer": "workflow", "reviewer": "op", "examples": [EXAMPLE]},
            "Unknown optimizer 'workflow'; expected one of entity_extraction, "
            "profile, query_enhancement, routing.",
        ),
        (
            {"optimizer": "query_enhancement", "reviewer": "op", "examples": []},
            "An upload needs at least one example.",
        ),
        (
            {"optimizer": "query_enhancement", "reviewer": "  ", "examples": [EXAMPLE]},
            "Name the reviewer.",
        ),
    ],
)
def test_an_upload_without_an_optimizer_examples_or_reviewer_is_refused(body, detail):
    response = _upload(body)
    assert (response.status_code, response.json()) == (400, {"detail": detail})


def test_a_valid_upload_without_a_configured_store_is_an_outage(monkeypatch):
    monkeypatch.setattr(training_examples, "_config_manager", None)
    response = _upload(
        {
            "optimizer": "query_enhancement",
            "reviewer": "operator@example.com",
            "examples": [EXAMPLE],
        }
    )
    assert (response.status_code, response.json()) == (
        503,
        {"detail": "Config manager not initialised"},
    )
