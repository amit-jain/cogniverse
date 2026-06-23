"""Unit tests for the local training-backend dispatch.

The orchestrator test mocks ``_create_backend`` away, so the backend that wires
the orchestrator to the SFT/DPO/embedding trainers was never instantiated.
These exercise each dispatch path and the TrainingJobResult wrapping, with the
trainers mocked (no real GPU training).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cogniverse_finetuning.training.backend import LocalTrainingBackend


def _fake_trainer(adapter_path: str):
    trainer = MagicMock()
    trainer.train = AsyncMock(
        return_value={"adapter_path": adapter_path, "metrics": {"loss": 0.1}}
    )
    return trainer


@pytest.mark.unit
class TestLocalTrainingBackendDispatch:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method, trainer_path",
        [
            ("train_sft", "cogniverse_finetuning.training.sft_trainer.SFTFinetuner"),
            ("train_dpo", "cogniverse_finetuning.training.dpo_trainer.DPOFinetuner"),
            (
                "train_embedding",
                "cogniverse_finetuning.training.embedding_finetuner.EmbeddingFinetuner",
            ),
        ],
    )
    async def test_dispatch_constructs_trainer_and_wraps_result(
        self, method, trainer_path
    ):
        backend = LocalTrainingBackend(config=MagicMock())
        fake = _fake_trainer("/tmp/adapter")

        with patch(trainer_path, return_value=fake) as mock_cls:
            result = await getattr(backend, method)(
                dataset=[{"x": 1}],
                base_model="HuggingFaceTB/SmolLM-135M",
                output_dir="/tmp/out",
                config={"epochs": 1},
            )

        mock_cls.assert_called_once_with(
            base_model="HuggingFaceTB/SmolLM-135M", output_dir="/tmp/out"
        )
        fake.train.assert_awaited_once()
        assert result.adapter_path == "/tmp/adapter"
        assert result.metrics == {"loss": 0.1}
        assert result.job_id == "local"
