"""Self-consistency sampling of the entity teacher, against a real HTTP teacher.

The stub is an OpenAI-compatible chat-completions server that records every
request body, so "three draws" is counted at the wire and a cached second call
is visible as a missing request rather than as agreement.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import dspy
import pytest

from cogniverse_agents.entity_extraction_agent import EntityExtractionModule
from cogniverse_agents.optimizer.entity_self_consistency import (
    AGREEMENT_KEY,
    ENTITIES_KEY,
    ENTITY_TEXT_KEY,
    ENTITY_TYPE_KEY,
    NEEDS_REVIEW_KEY,
    QUERY_KEY,
    SAMPLES_KEY,
    SELF_CONSISTENCY_ENTITY_KEYS,
    SELF_CONSISTENCY_METADATA_KEYS,
    SELF_CONSISTENCY_SAMPLES,
    SELF_CONSISTENCY_TEMPERATURE,
    agreement_entities,
    collect_self_consistency_rows,
    entity_agreement,
    review_row,
    row_confidence,
    row_needs_review,
    sample_entity_extraction,
    unanimous_entities,
)
from cogniverse_foundation.config.llm_factory import (
    create_budgeted_dspy_lm,
    create_sampling_dspy_lm,
)
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

TEACHER_MODEL = "Qwen/Qwen3-14B-AWQ"
QUERY = "a man riding a dirt bike"


class _SamplingTeacher:
    """Chat-completions server that cycles a scripted entity set per request."""

    def __init__(self, scripts_by_query: dict[str, list[list[dict[str, str]]]]) -> None:
        self.chat_requests: list[dict] = []
        self._scripts = scripts_by_query
        self._served: dict[str, int] = {}
        self._lock = threading.Lock()
        self.fail_from_request = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, payload: dict, status: int = 200) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler contract
                self._send(
                    {
                        "object": "list",
                        "data": [{"id": TEACHER_MODEL, "max_model_len": 8192}],
                    }
                )

            def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler contract
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length) or b"{}")
                with outer._lock:
                    outer.chat_requests.append(request)
                    ordinal = len(outer.chat_requests)
                    # Only the final user turn carries the query; the system
                    # turn quotes instruction examples that would match any
                    # script key.
                    turn = str(request.get("messages", [{}])[-1].get("content", ""))
                    query = next((name for name in outer._scripts if name in turn), "")
                    served = outer._served.get(query, 0)
                    outer._served[query] = served + 1
                if (
                    outer.fail_from_request is not None
                    and ordinal >= outer.fail_from_request
                ):
                    self._send({"error": "teacher down"}, status=503)
                    return
                script = outer._scripts[query]
                entities = script[served % len(script)]
                content = json.dumps(
                    {
                        "reasoning": f"draw {served} for {query}",
                        "entities": entities,
                    }
                )
                self._send(
                    {
                        "id": "chatcmpl-stub",
                        "object": "chat.completion",
                        "created": 0,
                        "model": TEACHER_MODEL,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                    }
                )

            def log_message(self, *args):
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "_SamplingTeacher":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    @property
    def api_base(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    @property
    def chat_posts(self) -> list[dict]:
        return list(self.chat_requests)


def _endpoint(api_base: str) -> LLMEndpointConfig:
    return LLMEndpointConfig(
        model=f"openai/{TEACHER_MODEL}",
        api_base=api_base,
        api_key="stub-key",
        temperature=0.1,
        max_tokens=1024,
        context_window=8192,
        request_timeout=30.0,
        num_retries=0,
        seed=17,
    )


@pytest.fixture(autouse=True)
def _dspy_cache_enabled():
    """Leave dspy's cache ON so a cached draw shows up as a missing request."""
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
    yield
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)


class TestAgreementArithmetic:
    def test_agreement_is_the_fraction_of_samples_carrying_the_mention(self):
        samples = [
            [
                {"text": "man", "type": "PERSON"},
                {"text": "dirt bike", "type": "CONCEPT"},
            ],
            [
                {"text": "man", "type": "PERSON"},
                {"text": "dirt bike", "type": "CONCEPT"},
            ],
            [{"text": "man", "type": "PERSON"}, {"text": "bike", "type": "CONCEPT"}],
        ]
        assert entity_agreement(samples) == {
            ("man", "PERSON"): 1.0,
            ("dirt bike", "CONCEPT"): 2 / 3,
            ("bike", "CONCEPT"): 1 / 3,
        }

    def test_same_text_under_two_types_is_disagreement_on_both(self):
        samples = [
            [{"text": "Helsinki", "type": "PLACE"}],
            [{"text": "Helsinki", "type": "ORGANIZATION"}],
            [{"text": "Helsinki", "type": "PLACE"}],
        ]
        assert entity_agreement(samples) == {
            ("helsinki", "PLACE"): 2 / 3,
            ("helsinki", "ORGANIZATION"): 1 / 3,
        }

    def test_agreement_counts_a_repeat_inside_one_sample_once(self):
        samples = [
            [{"text": "man", "type": "PERSON"}, {"text": "Man", "type": "PERSON"}],
            [{"text": "man", "type": "PERSON"}],
        ]
        assert entity_agreement(samples) == {("man", "PERSON"): 1.0}

    def test_agreement_unions_mentions_across_samples(self):
        samples = [
            [{"text": "a", "type": "PERSON"}],
            [{"text": "b", "type": "PLACE"}],
            [{"text": "c", "type": "EVENT"}],
        ]
        assert entity_agreement(samples) == {
            ("a", "PERSON"): 1 / 3,
            ("b", "PLACE"): 1 / 3,
            ("c", "EVENT"): 1 / 3,
        }

    def test_no_samples_is_refused_rather_than_read_as_no_agreement(self):
        with pytest.raises(ValueError, match="needs at least one sample"):
            entity_agreement([])

    def test_agreement_entities_is_the_complete_ordered_record(self):
        samples = [
            [
                {"text": "Man", "type": "PERSON"},
                {"text": "dirt bike", "type": "CONCEPT"},
            ],
            [{"text": "man", "type": "PERSON"}],
            [
                {"text": "man", "type": "PERSON"},
                {"text": "dirt bike", "type": "CONCEPT"},
            ],
        ]
        assert agreement_entities(samples) == [
            {
                "text": "Man",
                "type": "PERSON",
                "agreement": 1.0,
                "needs_review": False,
            },
            {
                "text": "dirt bike",
                "type": "CONCEPT",
                "agreement": 2 / 3,
                "needs_review": True,
            },
        ]

    def test_unanimous_entities_drops_every_flagged_mention(self):
        samples = [
            [{"text": "man", "type": "PERSON"}, {"text": "bike", "type": "CONCEPT"}],
            [{"text": "man", "type": "PERSON"}],
            [{"text": "man", "type": "PERSON"}],
        ]
        assert unanimous_entities(agreement_entities(samples)) == [
            {"text": "man", "type": "PERSON"}
        ]


class TestReviewRowShape:
    SAMPLES = [
        [{"text": "man", "type": "PERSON"}, {"text": "dirt bike", "type": "CONCEPT"}],
        [{"text": "man", "type": "PERSON"}],
        [{"text": "man", "type": "PERSON"}, {"text": "dirt bike", "type": "CONCEPT"}],
    ]

    def test_row_is_exactly_the_unanimous_example_and_the_full_record(self):
        assert review_row(QUERY, self.SAMPLES) == {
            "data": {
                "query": "a man riding a dirt bike",
                "entities": [{"text": "man", "type": "PERSON"}],
                "relationships": [],
            },
            "metadata": {
                "samples": 3,
                "entities": [
                    {
                        "text": "man",
                        "type": "PERSON",
                        "agreement": 1.0,
                        "needs_review": False,
                    },
                    {
                        "text": "dirt bike",
                        "type": "CONCEPT",
                        "agreement": 2 / 3,
                        "needs_review": True,
                    },
                ],
            },
        }

    def test_row_keys_are_the_production_constants(self):
        row = review_row(QUERY, self.SAMPLES)
        assert set(row["metadata"]) == set(SELF_CONSISTENCY_METADATA_KEYS)
        assert set(row["data"]) == {QUERY_KEY, ENTITIES_KEY, "relationships"}
        assert [set(entity) for entity in row["metadata"][ENTITIES_KEY]] == [
            set(SELF_CONSISTENCY_ENTITY_KEYS)
        ] * 2
        assert {
            ENTITY_TEXT_KEY,
            ENTITY_TYPE_KEY,
            AGREEMENT_KEY,
            NEEDS_REVIEW_KEY,
        } == set(SELF_CONSISTENCY_ENTITY_KEYS)
        assert row["metadata"][SAMPLES_KEY] == SELF_CONSISTENCY_SAMPLES

    def test_unanimous_row_needs_no_review_and_scores_one(self):
        unanimous = [[{"text": "man", "type": "PERSON"}]] * 3
        row = review_row(QUERY, unanimous)
        assert row_needs_review(row) is False
        assert row_confidence(row) == 1.0

    def test_flagged_row_confidence_is_the_mean_agreement(self):
        assert row_confidence(review_row(QUERY, self.SAMPLES)) == (1.0 + 2 / 3) / 2
        assert row_needs_review(review_row(QUERY, self.SAMPLES)) is True


class TestSamplingReachesTheTeacherThreeTimes:
    SCRIPT = {
        QUERY: [
            [
                {"text": "man", "type": "PERSON"},
                {"text": "dirt bike", "type": "CONCEPT"},
            ],
            [{"text": "man", "type": "PERSON"}],
            [
                {"text": "man", "type": "PERSON"},
                {"text": "dirt bike", "type": "CONCEPT"},
            ],
        ]
    }

    def test_three_distinct_requests_sample_above_zero_with_no_seed(self):
        with _SamplingTeacher(self.SCRIPT) as teacher:
            lm = create_sampling_dspy_lm(
                _endpoint(teacher.api_base),
                temperature=SELF_CONSISTENCY_TEMPERATURE,
            )
            drawn = sample_entity_extraction(
                EntityExtractionModule, QUERY, lm=lm, samples=SELF_CONSISTENCY_SAMPLES
            )

        posts = teacher.chat_posts
        assert len(posts) == SELF_CONSISTENCY_SAMPLES
        assert [post["temperature"] for post in posts] == [
            SELF_CONSISTENCY_TEMPERATURE
        ] * SELF_CONSISTENCY_SAMPLES
        assert SELF_CONSISTENCY_TEMPERATURE == 0.7
        assert [post.get("seed") for post in posts] == [None] * SELF_CONSISTENCY_SAMPLES
        assert [post.get("extra_body") for post in posts] == [
            None
        ] * SELF_CONSISTENCY_SAMPLES
        assert drawn == [
            [
                {"text": "man", "type": "PERSON"},
                {"text": "dirt bike", "type": "CONCEPT"},
            ],
            [{"text": "man", "type": "PERSON"}],
            [
                {"text": "man", "type": "PERSON"},
                {"text": "dirt bike", "type": "CONCEPT"},
            ],
        ]

    def test_the_budgeted_teacher_lm_serves_repeats_from_cache(self):
        """Control: the LM the compile step uses answers three asks with one call."""
        with _SamplingTeacher(self.SCRIPT) as teacher:
            lm = create_budgeted_dspy_lm(_endpoint(teacher.api_base))
            sample_entity_extraction(
                EntityExtractionModule, QUERY, lm=lm, samples=SELF_CONSISTENCY_SAMPLES
            )
        assert len(teacher.chat_posts) == 1

    def test_sampling_lm_refuses_a_temperature_that_cannot_sample(self):
        with pytest.raises(ValueError, match="temperature above zero"):
            create_sampling_dspy_lm(_endpoint("http://127.0.0.1:1/v1"), temperature=0.0)


class TestServingPathIsUnchanged:
    def test_the_served_module_makes_exactly_one_call_per_query(self):
        with _SamplingTeacher(self.__class__.SCRIPT) as teacher:
            lm = create_budgeted_dspy_lm(_endpoint(teacher.api_base))
            with dspy.context(lm=lm):
                prediction = EntityExtractionModule()(query=QUERY)

        assert len(teacher.chat_posts) == 1
        assert [
            {"text": entity.text, "type": entity.type} for entity in prediction.entities
        ] == [{"text": "man", "type": "PERSON"}]

    SCRIPT = {QUERY: [[{"text": "man", "type": "PERSON"}]]}


class TestFaultAndConcurrency:
    def test_a_teacher_that_dies_mid_draw_records_a_cause_and_no_row(self):
        script = {
            QUERY: [
                [{"text": "man", "type": "PERSON"}],
                [{"text": "man", "type": "PERSON"}],
                [{"text": "man", "type": "PERSON"}],
            ]
        }
        causes: list[str] = []
        with _SamplingTeacher(script) as teacher:
            teacher.fail_from_request = 2
            lm = create_sampling_dspy_lm(
                _endpoint(teacher.api_base),
                temperature=SELF_CONSISTENCY_TEMPERATURE,
            )
            rows = asyncio.run(
                collect_self_consistency_rows(
                    [{"query": QUERY, "example_id": "truth:0"}],
                    EntityExtractionModule,
                    lm=lm,
                    samples=SELF_CONSISTENCY_SAMPLES,
                    record_cause=causes.append,
                )
            )

        assert rows == []
        assert len(causes) == 1
        assert causes[0].startswith(
            "self-consistency sampling failed for query 'a man riding a dirt bike': "
        )

    def test_concurrent_examples_keep_their_own_draws(self):
        queries = [
            "a man riding a dirt bike",
            "interns working at Nokia",
            "a recorded lecture on Matplotlib",
        ]
        script = {
            queries[0]: [[{"text": "man", "type": "PERSON"}]],
            queries[1]: [
                [{"text": "interns", "type": "PERSON"}],
                [{"text": "interns", "type": "PERSON"}],
                [{"text": "Nokia", "type": "ORGANIZATION"}],
            ],
            queries[2]: [[{"text": "recorded lecture", "type": "EVENT"}]],
        }
        causes: list[str] = []
        with _SamplingTeacher(script) as teacher:
            lm = create_sampling_dspy_lm(
                _endpoint(teacher.api_base),
                temperature=SELF_CONSISTENCY_TEMPERATURE,
            )
            rows = asyncio.run(
                collect_self_consistency_rows(
                    [
                        {"query": query, "example_id": f"truth:{index}"}
                        for index, query in enumerate(queries)
                    ],
                    EntityExtractionModule,
                    lm=lm,
                    samples=SELF_CONSISTENCY_SAMPLES,
                    record_cause=causes.append,
                )
            )

        assert causes == []
        assert len(teacher.chat_posts) == 3 * SELF_CONSISTENCY_SAMPLES
        assert [(row["example_id"], row["data"]) for row in rows] == [
            (
                "truth:0",
                {
                    "query": queries[0],
                    "entities": [{"text": "man", "type": "PERSON"}],
                    "relationships": [],
                },
            ),
            (
                "truth:1",
                {
                    "query": queries[1],
                    "entities": [],
                    "relationships": [],
                },
            ),
            (
                "truth:2",
                {
                    "query": queries[2],
                    "entities": [{"text": "recorded lecture", "type": "EVENT"}],
                    "relationships": [],
                },
            ),
        ]
        assert [row["metadata"]["entities"] for row in rows] == [
            [
                {
                    "text": "man",
                    "type": "PERSON",
                    "agreement": 1.0,
                    "needs_review": False,
                }
            ],
            [
                {
                    "text": "interns",
                    "type": "PERSON",
                    "agreement": 2 / 3,
                    "needs_review": True,
                },
                {
                    "text": "Nokia",
                    "type": "ORGANIZATION",
                    "agreement": 1 / 3,
                    "needs_review": True,
                },
            ],
            [
                {
                    "text": "recorded lecture",
                    "type": "EVENT",
                    "agreement": 1.0,
                    "needs_review": False,
                }
            ],
        ]
