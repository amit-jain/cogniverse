"""Unit tests for the embedding finetuner.

SFT/DPO trainers have train()->_train_local() delegation tests; the embedding
trainer did not. This mirrors that coverage without running real training.
"""

from unittest.mock import AsyncMock, patch

import pytest

from cogniverse_finetuning.training.embedding_finetuner import EmbeddingFinetuner


@pytest.mark.unit
class TestEmbeddingFinetuner:
    def test_init_with_base_model_and_output_dir(self):
        ft = EmbeddingFinetuner(
            base_model="jinaai/jina-embeddings-v3", output_dir="/tmp/output"
        )
        assert ft.base_model == "jinaai/jina-embeddings-v3"
        assert ft.output_dir == "/tmp/output"

    @pytest.mark.asyncio
    async def test_train_delegates_to_train_local(self):
        ft = EmbeddingFinetuner(base_model="m", output_dir="/tmp/output")
        dataset = [{"anchor": "a", "positive": "p", "negative": "n"}]
        config = {"epochs": 1, "triplet_margin": 0.5}

        with patch.object(ft, "_train_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = {"adapter_path": "/tmp/adapter", "metrics": {}}
            result = await ft.train(dataset, config)

            mock_local.assert_called_once_with(dataset, config)
            assert result["adapter_path"] == "/tmp/adapter"
