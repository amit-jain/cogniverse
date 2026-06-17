"""Unit tests for the VLM caption bake-off scoring logic.

Exercises the pure score-aggregation and winner-selection functions plus the
judge-JSON parser — no models, no network, no GPU.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from vlm_caption_bakeoff import (  # noqa: E402
    FrameJudgement,
    _parse_judge_json,
    aggregate_scores,
    build_judge_prompt,
    select_winner,
)


def _j(frame, scores):
    return FrameJudgement(frame=frame, scores=scores)


def test_aggregate_means_per_model_per_criterion():
    judgements = [
        _j(
            "f1.jpg",
            {
                "Qwen/Qwen3-VL-8B-Instruct": {
                    "faithfulness": 8,
                    "detail": 6,
                    "hallucination": 10,
                },
                "OpenGVLab/InternVL3_5-8B": {
                    "faithfulness": 6,
                    "detail": 8,
                    "hallucination": 6,
                },
            },
        ),
        _j(
            "f2.jpg",
            {
                "Qwen/Qwen3-VL-8B-Instruct": {
                    "faithfulness": 10,
                    "detail": 8,
                    "hallucination": 10,
                },
                "OpenGVLab/InternVL3_5-8B": {
                    "faithfulness": 8,
                    "detail": 6,
                    "hallucination": 8,
                },
            },
        ),
    ]
    agg = aggregate_scores(judgements)

    assert agg["Qwen/Qwen3-VL-8B-Instruct"] == {
        "faithfulness": 9.0,
        "detail": 7.0,
        "hallucination": 10.0,
        "overall": 8.6667,
    }
    assert agg["OpenGVLab/InternVL3_5-8B"] == {
        "faithfulness": 7.0,
        "detail": 7.0,
        "hallucination": 7.0,
        "overall": 7.0,
    }


def test_winner_is_highest_overall():
    agg = {
        "Qwen/Qwen3-VL-8B-Instruct": {"overall": 8.6667},
        "OpenGVLab/InternVL3_5-8B": {"overall": 7.0},
        "openbmb/MiniCPM-V-4_5": {"overall": 8.0},
    }
    assert select_winner(agg) == "Qwen/Qwen3-VL-8B-Instruct"


def test_winner_tie_break_is_deterministic():
    agg = {
        "model-b": {"overall": 8.0},
        "model-a": {"overall": 8.0},
    }
    # Same overall for two reruns must yield the same winner.
    assert select_winner(agg) == select_winner(agg)
    assert select_winner(agg) in ("model-a", "model-b")


def test_model_averaged_only_over_scored_frames():
    # A model missing on one frame is averaged only over the frame it has.
    judgements = [
        _j("f1.jpg", {"m": {"faithfulness": 4, "detail": 4, "hallucination": 4}}),
        _j("f2.jpg", {}),  # m got no score on f2
    ]
    agg = aggregate_scores(judgements)
    assert agg["m"]["faithfulness"] == 4.0
    assert agg["m"]["overall"] == 4.0


def test_model_with_zero_scores_omitted():
    judgements = [_j("f1.jpg", {})]
    assert aggregate_scores(judgements) == {}


def test_select_winner_empty_returns_none():
    assert select_winner({}) is None


def test_parse_judge_json_plain():
    raw = '{"scores": {"caption_a": {"faithfulness": 9}}, "rationale": "ok"}'
    parsed = _parse_judge_json(raw)
    assert parsed["scores"]["caption_a"]["faithfulness"] == 9
    assert parsed["rationale"] == "ok"


def test_parse_judge_json_code_fenced():
    raw = '```json\n{"scores": {"caption_a": {"detail": 7}}, "rationale": "x"}\n```'
    parsed = _parse_judge_json(raw)
    assert parsed["scores"]["caption_a"]["detail"] == 7


def test_parse_judge_json_embedded_in_prose():
    raw = 'Here is my verdict: {"scores": {}, "rationale": "y"} thanks.'
    parsed = _parse_judge_json(raw)
    assert parsed["rationale"] == "y"


def test_parse_judge_json_invalid_raises():
    with pytest.raises(ValueError):
        _parse_judge_json("not json at all")


def test_build_judge_prompt_lists_all_captions_and_demands_json():
    prompt = build_judge_prompt(
        {"caption_a": "a dog runs", "caption_b": "a cat sleeps"}
    )
    assert "caption_a" in prompt
    assert "a dog runs" in prompt
    assert "caption_b" in prompt
    assert "a cat sleeps" in prompt
    assert "STRICT JSON" in prompt
    assert "hallucination" in prompt
