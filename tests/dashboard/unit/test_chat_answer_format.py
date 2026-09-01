"""format_gateway_answer renders the gateway's search payload as chat markdown.

The payload shapes below are the ones the gateway actually returns for a chat
query: ``status``/``agent``/``message``/``results``/``results_count``, with no
prose ``response`` or ``result`` field.
"""

from __future__ import annotations

from cogniverse_dashboard.chat import format_gateway_answer


def test_gateway_search_payload_renders_message_and_scored_hits():
    answer = format_gateway_answer(
        {
            "status": "success",
            "agent": "gateway_agent",
            "message": "Found 2 results for 'animal videos compilation'",
            "results_count": 2,
            "results": [
                {
                    "id": "739064eb_seg_22",
                    "document_id": (
                        "id:content:video_colpali_smol500_mv_frame_flywheel_org_"
                        "production::739064eb_seg_22"
                    ),
                    "score": 2.5503595834597945,
                    "metadata": {"audio_transcript": "  a  lion   roars \n loudly "},
                },
                {
                    "document_id": "id:content:video::739064eb_seg_7",
                    "score": 1.25,
                    "metadata": {},
                },
            ],
        }
    )

    assert answer == (
        "Found 2 results for 'animal videos compilation'\n"
        "\n"
        "**1.** `739064eb_seg_22` — score 2.550\n"
        "\n"
        "> a lion roars loudly\n"
        "\n"
        "**2.** `739064eb_seg_7` — score 1.250"
    )


def test_document_id_namespace_is_stripped_to_the_bare_reference():
    answer = format_gateway_answer(
        {
            "message": "Found 1 result",
            "results": [
                {
                    "document_id": "id:content:video_colpali::abc_seg_3",
                    "score": 0.5,
                }
            ],
        }
    )

    assert answer == "Found 1 result\n\n**1.** `abc_seg_3` — score 0.500"


def test_only_the_first_five_hits_are_rendered():
    answer = format_gateway_answer(
        {
            "message": "Found 9 results",
            "results": [{"id": f"seg_{n}", "score": float(n)} for n in range(1, 10)],
        }
    )

    assert answer == (
        "Found 9 results\n"
        "\n"
        "**1.** `seg_1` — score 1.000\n"
        "\n"
        "**2.** `seg_2` — score 2.000\n"
        "\n"
        "**3.** `seg_3` — score 3.000\n"
        "\n"
        "**4.** `seg_4` — score 4.000\n"
        "\n"
        "**5.** `seg_5` — score 5.000"
    )


def test_long_transcript_is_truncated_to_two_hundred_characters():
    answer = format_gateway_answer(
        {
            "message": "Found 1 result",
            "results": [{"id": "seg_1", "metadata": {"audio_transcript": "ab " * 200}}],
        }
    )

    quoted = answer.split("> ", 1)[1]
    assert len(quoted) == 200
    assert quoted == ("ab " * 200).strip()[:200]


def test_message_only_payload_renders_just_the_message():
    assert format_gateway_answer({"message": "Found 0 results"}) == "Found 0 results"


def test_empty_payload_states_that_nothing_came_back():
    assert format_gateway_answer({}) == "The agent returned no message and no results."


def test_raw_payload_internals_never_leak_into_the_reply():
    """A missing prose field must not fall back to stringifying the payload:
    that put document ids, tenant-qualified schema names and raw dict syntax
    into the chat window."""
    answer = format_gateway_answer(
        {
            "status": "success",
            "agent": "gateway_agent",
            "message": "Found 1 result",
            "search_mode": "hybrid",
            "profile": "video_colpali_smol500_mv_frame",
            "downstream_result": {"internal": "detail"},
            "results": [
                {
                    "document_id": "id:content:video_colpali::abc_seg_3",
                    "score": 0.5,
                }
            ],
        }
    )

    for leaked in ("document_id", "downstream_result", "{'", "search_mode", "status"):
        assert leaked not in answer, (
            f"{leaked!r} leaked into the chat reply: {answer!r}"
        )


def test_boolean_score_is_not_formatted_as_a_number():
    """``bool`` is a subclass of ``int``; a True score must not render as
    'score 1.000' and imply a real relevance value."""
    answer = format_gateway_answer(
        {"message": "Found 1 result", "results": [{"id": "seg_1", "score": True}]}
    )

    assert answer == "Found 1 result\n\n**1.** `seg_1`"
