"""Answer agents are handed each grounding hit's title and content.

The dispatcher searches a tenant's profiles through the real SearchAgent
against a test-owned Vespa, flattens the source-collapsed hits it gets back,
and hands them to the summarizer and the detailed report. Every shipped
profile type is fed one source whose schema fields carry a known title and
known text, and the test reads the ``content`` input the answer LM receives
off the wire of an OpenAI-compatible provider it drives: the hit line must
name the source's own title, its content type and every text field it
carries, never "Unknown" with no content.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import dspy
import pytest
from vespa.application import Vespa

from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import (
    BackendProfileConfig,
    SystemConfig,
)
from cogniverse_foundation.dspy import LenientJSONAdapter
from cogniverse_runtime.agent_dispatcher import GROUNDING_SEARCHED, AgentDispatcher
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.vespa_docker import VespaDockerManager

pytestmark = [
    pytest.mark.integration,
    pytest.mark.no_shared_vespa,
    pytest.mark.requires_inference("vllm_colpali"),
    pytest.mark.requires_inference("colbert_pylate"),
    pytest.mark.requires_inference("code_colbert_pylate"),
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SHIPPED_PROFILES = json.loads((_REPO_ROOT / "configs" / "config.json").read_text())[
    "backend"
]["profiles"]

TENANT = "answer_content:t1"

# One source per profile type: the schema fields it is fed, the text-only rank
# profile that finds it without document embeddings, the query that matches
# it, and the hit an answer agent must be handed for it.
SOURCES = {
    "video_frame": {
        "profile": "video_colpali_smol500_mv_frame",
        "ranking": "bm25_only",
        "query": "tugboat barge lighthouse",
        "documents": [
            (
                "harbour_seg_0",
                {
                    "video_id": "harbour",
                    "video_title": "harbour_dawn.mp4",
                    "segment_id": 0,
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "segment_description": (
                        "A tugboat pushes a barge past the lighthouse at dawn."
                    ),
                    "audio_transcript": "Engines idle as the barge clears the pier.",
                },
            ),
            (
                "harbour_seg_1",
                {
                    "video_id": "harbour",
                    "video_title": "harbour_dawn.mp4",
                    "segment_id": 1,
                    "start_time": 1.0,
                    "end_time": 2.0,
                    "segment_description": "Gulls circle a crane beside the tugboat.",
                    "audio_transcript": "",
                },
            ),
        ],
        "title": "harbour_dawn.mp4",
        "content_type": "video",
        "content": (
            "A tugboat pushes a barge past the lighthouse at dawn.\n"
            "Engines idle as the barge clears the pier."
        ),
    },
    "video_chunk": {
        "profile": "video_colqwen_omni_mv_chunk_30s",
        "ranking": "bm25_only",
        "query": "orchard ladder apples",
        "documents": [
            (
                "orchard_chunk_0",
                {
                    "video_id": "orchard",
                    "video_title": "orchard_harvest.mkv",
                    "segment_id": 0,
                    "start_time": 0.0,
                    "end_time": 30.0,
                    "audio_transcript": (
                        "Pickers climb the orchard ladder and fill crates with apples."
                    ),
                },
            ),
        ],
        "title": "orchard_harvest.mkv",
        "content_type": "video",
        "content": "Pickers climb the orchard ladder and fill crates with apples.",
    },
    "document_window": {
        "profile": "document_text_semantic",
        "ranking": "bm25_only",
        "query": "silt tidal basin dredging",
        "documents": [
            (
                "silt_report_w0000",
                {
                    "document_id": "silt_report",
                    "document_title": "silt_report.md",
                    "document_type": "md",
                    "full_text": (
                        "Silt settles across the tidal basin and dredging resumes "
                        "each spring."
                    ),
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "chunk_start": 0,
                    "chunk_end": 69,
                },
            ),
        ],
        "title": "silt_report.md",
        "content_type": "document",
        "content": (
            "Silt settles across the tidal basin and dredging resumes each spring."
        ),
    },
    "audio": {
        "profile": "audio_clap_semantic",
        "ranking": "transcript_search",
        "query": "beekeeper smoker hive",
        "documents": [
            (
                "apiary_chunk_0",
                {
                    "audio_id": "apiary",
                    "audio_title": "apiary_interview.wav",
                    "audio_transcript": (
                        "The beekeeper lights the smoker before opening the hive."
                    ),
                    "chunk_index": 0,
                    "chunk_count": 1,
                },
            ),
        ],
        "title": "apiary_interview.wav",
        "content_type": "audio",
        "content": "The beekeeper lights the smoker before opening the hive.",
    },
    "code": {
        "profile": "code_lateon_mv",
        "ranking": "bm25_only",
        "query": "parse tide table",
        "documents": [
            (
                "tides_chunk_0",
                {
                    "code_id": "tides",
                    "file_path": "src/tides.py",
                    "chunk_name": "parse_tide_table",
                    "chunk_type": "function",
                    "language": "python",
                    "signature": "def parse_tide_table(text)",
                    "source_code": (
                        "def parse_tide_table(text):\n"
                        "    return [line.split() for line in text.splitlines()]"
                    ),
                    "line_start": 1,
                    "line_end": 2,
                    "chunk_index": 0,
                    "chunk_count": 1,
                },
            ),
        ],
        "title": "src/tides.py:parse_tide_table",
        "content_type": "code",
        "content": (
            "def parse_tide_table(text):\n"
            "    return [line.split() for line in text.splitlines()]"
        ),
    },
}

_OUTPUT_FIELD = re.compile(r"^\d+\. `(\w+)` \(([^)]*)\)", re.MULTILINE)
_QUERY_INPUT = re.compile(r"\[\[ ## query ## \]\]\n(.*?)\n")
_CONTENT_INPUT = re.compile(r"\[\[ ## content ## \]\]\n(.*?)\n\n\[\[ ## ", re.DOTALL)


class _AnswerProvider:
    """OpenAI-compatible provider that fills whatever outputs it is asked for
    and records every request's messages."""

    def __init__(self) -> None:
        self.requests: list[list[dict]] = []
        self.lock = threading.Lock()
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with provider.lock:
                    provider.requests.append(body["messages"])
                system = next(
                    message["content"]
                    for message in body["messages"]
                    if message["role"] == "system"
                )
                outputs = system.split("Your output fields are:", 1)[1].split(
                    "All interactions", 1
                )[0]
                user = "\n".join(
                    message["content"]
                    for message in body["messages"]
                    if message["role"] == "user"
                )
                asked = _QUERY_INPUT.search(user)
                fields = {
                    # A query rewrite hands back the query as asked, so the
                    # search matches the fed source.
                    name: (
                        asked.group(1)
                        if asked and "query" in name and "str" in kind
                        else _filled(kind)
                    )
                    for name, kind in _OUTPUT_FIELD.findall(outputs)
                }
                encoded = json.dumps(
                    {
                        "id": "chatcmpl-answer",
                        "object": "chat.completion",
                        "created": 0,
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": json.dumps(fields),
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def content_inputs(self, output_field: str) -> list[str]:
        """The ``content`` input of every request asking for ``output_field``."""
        with self.lock:
            requests = list(self.requests)
        return [
            _CONTENT_INPUT.search(message["content"]).group(1)
            for messages in requests
            if f"`{output_field}`" in messages[0]["content"]
            for message in messages
            if message["role"] == "user"
        ]


def _filled(kind: str) -> object:
    if "float" in kind:
        return 0.9
    if "bool" in kind:
        return True
    if "int" in kind:
        return 1
    if "list" in kind:
        return ["recorded"]
    return "recorded"


@pytest.fixture(scope="module")
def content_vespa():
    from cogniverse_vespa.metadata_schemas import (
        create_config_metadata_schema,
        create_organization_metadata_schema,
        create_tenant_metadata_schema,
    )
    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
    from tests.conftest import _shared_vespa_application_package

    manager = VespaDockerManager()
    info = manager.start_container(f"answer-content-{uuid.uuid4().hex}")
    try:
        manager.wait_for_config_ready(info)
        package = _shared_vespa_application_package(
            [
                create_config_metadata_schema(),
                create_organization_metadata_schema(),
                create_tenant_metadata_schema(),
            ]
        )
        VespaSchemaManager(
            backend_endpoint="http://localhost", backend_port=info["config_port"]
        )._deploy_package(lambda: package)
        manager.wait_for_application_ready(info)
        yield info
    finally:
        manager.stop_container(info)


@pytest.fixture(scope="module")
def answer_provider():
    provider = _AnswerProvider()
    provider.thread.start()
    try:
        yield provider
    finally:
        provider.server.shutdown()
        provider.server.server_close()
        provider.thread.join(5)


@pytest.fixture(scope="module")
def content_config_manager(content_vespa, resolved_inference_endpoints):
    config_manager = ConfigManager(
        store=VespaConfigStore(
            backend_url="http://localhost", backend_port=content_vespa["http_port"]
        )
    )
    config_manager.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=content_vespa["http_port"],
            inference_service_urls={
                service: resolved_inference_endpoints[service].base_url
                for service in ("vllm_colpali", "colbert_pylate", "code_colbert_pylate")
            },
        )
    )
    for source in SOURCES.values():
        name = source["profile"]
        config_manager.add_backend_profile(
            BackendProfileConfig.from_dict(
                name, {**_SHIPPED_PROFILES[name], "default_ranking": source["ranking"]}
            ),
            tenant_id=TENANT,
        )
    return config_manager


@pytest.fixture(scope="module")
def seeded_sources(content_vespa, content_config_manager):
    """Every source fed into its tenant schema, through the tenant's backend."""
    from cogniverse_runtime.ingestion.processors.embedding_generator.backend_factory import (  # noqa: E501
        BackendFactory,
    )

    backend = BackendFactory.create(
        "vespa",
        TENANT,
        {},
        config_manager=content_config_manager,
        schema_loader=FilesystemSchemaLoader(_REPO_ROOT / "configs" / "schemas"),
    )
    schemas = [
        _SHIPPED_PROFILES[source["profile"]]["schema_name"]
        for source in SOURCES.values()
    ]
    backend.schema_registry.deploy_schemas(TENANT, schemas)
    app = Vespa(url="http://localhost", port=content_vespa["http_port"])
    for source, schema in zip(SOURCES.values(), schemas, strict=True):
        tenant_schema = backend.get_tenant_schema_name(TENANT, schema)
        for data_id, fields in source["documents"]:
            response = app.feed_data_point(
                schema=tenant_schema, data_id=data_id, fields=fields
            )
            assert response.status_code == 200, response.json
    time.sleep(3)
    return SOURCES


@pytest.fixture
def answer_dispatcher(
    content_config_manager, seeded_sources, answer_provider, tmp_path, monkeypatch
):
    config = json.loads((_REPO_ROOT / "configs" / "config.json").read_text())
    config.pop("active_video_profile", None)
    config["llm_config"] = {
        "primary": {
            "model": "openai/answer-content",
            "api_base": answer_provider.url,
            "api_key": "not-required",
            "max_tokens": 256,
            "temperature": 0,
            "num_retries": 0,
            "request_timeout": 30,
        }
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(config_file))
    with answer_provider.lock:
        answer_provider.requests.clear()
    lm = dspy.LM(
        "openai/answer-content",
        api_base=answer_provider.url,
        api_key="not-required",
        cache=False,
        num_retries=0,
    )
    with dspy.context(lm=lm, adapter=LenientJSONAdapter()):
        yield AgentDispatcher(
            agent_registry=AgentRegistry(
                tenant_id=TENANT, config_manager=content_config_manager
            ),
            config_manager=content_config_manager,
            schema_loader=FilesystemSchemaLoader(_REPO_ROOT / "configs" / "schemas"),
        )


async def _grounded_hit(dispatcher, source) -> dict:
    grounding = await dispatcher._resolve_answer_search_results(
        source["query"], TENANT, {"profiles": [source["profile"]]}, top_k=5
    )
    assert (grounding.state, grounding.profiles) == (
        GROUNDING_SEARCHED,
        (source["profile"],),
    )
    assert len(grounding.hits) == 1, grounding.hits
    return grounding.hits[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(SOURCES))
async def test_a_grounding_hit_carries_its_sources_title_and_content(
    answer_dispatcher, kind
):
    source = SOURCES[kind]
    hit = await _grounded_hit(answer_dispatcher, source)

    assert (hit["title"], hit["content_type"], hit["description"]) == (
        source["title"],
        source["content_type"],
        source["content"],
    )
    assert hit["text_content"] == source["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(SOURCES))
async def test_the_summarizer_hands_the_lm_each_hits_title_and_content(
    answer_dispatcher, answer_provider, kind
):
    source = SOURCES[kind]
    score = (await _grounded_hit(answer_dispatcher, source))["score"]

    result = await answer_dispatcher._execute_summarization_task(
        source["query"], TENANT, {"profiles": [source["profile"]]}
    )

    assert result["status"] == "success", result
    [content] = answer_provider.content_inputs("summary")
    assert content.startswith(
        f"- {source['title']} ({source['content_type']}, relevance {score:.2f}): "
        f"{source['content']}\n"
    ), content


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(SOURCES))
async def test_the_detailed_report_hands_the_lm_each_hits_title_and_content(
    answer_dispatcher, answer_provider, kind
):
    source = SOURCES[kind]
    score = (await _grounded_hit(answer_dispatcher, source))["score"]

    result = await answer_dispatcher._execute_detailed_report_task(
        source["query"], TENANT, {"profiles": [source["profile"]]}
    )

    assert result["status"] == "success", result
    assert answer_provider.content_inputs("executive_summary") == [
        f"Total Results: 1\n"
        f"- {source['title']} ({source['content_type']}, score: {score:.2f}): "
        f"{source['content']}"
    ]
