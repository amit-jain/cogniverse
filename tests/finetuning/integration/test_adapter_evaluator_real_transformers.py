import asyncio
import time

import pytest
import torch
from transformers import BatchEncoding, GPT2Config, GPT2LMHeadModel

from cogniverse_finetuning.evaluation.adapter_evaluator import AdapterEvaluator

pytestmark = pytest.mark.integration


class _InvalidOutputTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.decoded_sequences: list[list[int]] = []

    def __call__(self, text, **kwargs):
        assert text == "Route the launch recording"
        assert kwargs == {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": 512,
        }
        time.sleep(0.05)
        return BatchEncoding(
            {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            }
        )

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        self.decoded_sequences.append(token_ids.tolist())
        return "invalid model response"


@pytest.mark.asyncio
async def test_actual_transformers_generation_is_bounded_and_scored_exactly_off_loop():
    torch.manual_seed(7)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=16,
            n_positions=300,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=15,
            pad_token_id=0,
        )
    )
    tokenizer = _InvalidOutputTokenizer()
    evaluator = object.__new__(AdapterEvaluator)
    evaluator.agent_type = "routing"
    ticks = 0

    async def ticker():
        nonlocal ticks
        for _ in range(3):
            await asyncio.sleep(0.01)
            ticks += 1

    metrics, _ = await asyncio.gather(
        evaluator._evaluate_model(
            model,
            tokenizer,
            [
                {
                    "input": "Route the launch recording",
                    "expected_output": '{"recommended_agent":"search_agent"}',
                }
            ],
        ),
        ticker(),
    )

    assert ticks == 3
    assert len(tokenizer.decoded_sequences) == 3
    assert metrics.accuracy == 0.0
    assert metrics.top_k_accuracy == 0.0
    assert 0.0 < metrics.avg_confidence < 1.0
    assert metrics.confidence_calibration == pytest.approx(1.0 - metrics.avg_confidence)
    assert metrics.error_rate == 1.0
    assert metrics.hallucination_rate == 1.0
    assert metrics.sample_count == 1
    assert metrics.correctness == (False,)
