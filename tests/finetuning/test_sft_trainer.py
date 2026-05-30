"""
Unit tests for SFT trainer.

Tests initialization, delegation, LoRA fallback, and validation split logic.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cogniverse_finetuning.training.sft_trainer import SFTFinetuner


@pytest.mark.unit
class TestTrainerInitialization:
    """Test SFT trainer initialization"""

    def test_init_with_base_model_and_output_dir(self):
        """Test initialization with required params"""
        finetuner = SFTFinetuner(
            base_model="HuggingFaceTB/SmolLM-135M", output_dir="/tmp/output"
        )

        assert finetuner.base_model == "HuggingFaceTB/SmolLM-135M"
        assert finetuner.output_dir == "/tmp/output"

    @pytest.mark.asyncio
    async def test_train_calls_train_local(self):
        """Test that train() delegates to _train_local()"""
        finetuner = SFTFinetuner(
            base_model="HuggingFaceTB/SmolLM-135M", output_dir="/tmp/output"
        )

        dataset = [{"text": "Example instruction"}]
        config = {"epochs": 1, "learning_rate": 2e-4}

        with patch.object(
            finetuner, "_train_local", new_callable=AsyncMock
        ) as mock_train_local:
            mock_train_local.return_value = {
                "adapter_path": "/tmp/adapter",
                "metrics": {},
            }

            result = await finetuner.train(dataset, config)

            mock_train_local.assert_called_once_with(dataset, config)
            assert result["adapter_path"] == "/tmp/adapter"


@pytest.mark.unit
class TestValidationSplit:
    """Test validation split logic for SFT trainer"""

    @pytest.mark.asyncio
    async def test_no_validation_split_for_small_dataset(self):
        """Test that datasets with ≤100 examples don't get a validation split"""
        finetuner = SFTFinetuner(
            base_model="HuggingFaceTB/SmolLM-135M", output_dir="/tmp/output"
        )

        # Small dataset: 50 examples (≤100)
        dataset = [{"text": f"Example {i}"} for i in range(50)]
        config = {"epochs": 1, "batch_size": 2, "learning_rate": 2e-4}

        # Mock all dependencies
        mock_model = MagicMock()
        mock_model.config.eos_token_id = 0
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token = None
        mock_dataset = MagicMock()
        mock_trainer = MagicMock()
        mock_train_result = MagicMock()
        mock_train_result.metrics = {"train_loss": 0.5}
        mock_trainer.train.return_value = mock_train_result

        # Patch at the point where they're imported/used inside _train_local
        with (
            patch("transformers.AutoModelForCausalLM") as mock_model_cls,
            patch("transformers.AutoTokenizer") as mock_tokenizer_cls,
            patch("datasets.Dataset.from_list") as mock_from_list,
            patch("peft.LoraConfig"),
            patch("peft.get_peft_model", return_value=mock_model),
            patch("trl.SFTTrainer", return_value=mock_trainer) as mock_sft_trainer,
            patch("pathlib.Path.mkdir"),
        ):
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            mock_from_list.return_value = mock_dataset

            # No try/except: _train_local must run to completion against these
            # mocks. A swallowed exception here hid the evaluation_strategy
            # TypeError (transformers renamed it) for the whole local-train path.
            await finetuner._train_local(dataset, config)

            # Verify Dataset.from_list was called ONCE (no validation split)
            assert mock_from_list.call_count == 1
            mock_from_list.assert_called_once_with(dataset)

            # SFTTrainer must have been constructed with eval_dataset=None.
            assert mock_sft_trainer.called
            trainer_call_kwargs = mock_sft_trainer.call_args[1]
            assert trainer_call_kwargs.get("eval_dataset") is None

    @pytest.mark.asyncio
    async def test_validation_split_for_large_dataset(self):
        """Test that datasets with >100 examples get a 90/10 train/val split"""
        finetuner = SFTFinetuner(
            base_model="HuggingFaceTB/SmolLM-135M", output_dir="/tmp/output"
        )

        # Large dataset: 150 examples (>100)
        dataset = [{"text": f"Example {i}"} for i in range(150)]
        config = {"epochs": 1, "batch_size": 2, "learning_rate": 2e-4}

        # Mock all dependencies
        mock_model = MagicMock()
        mock_model.config.eos_token_id = 0
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token = None
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        mock_trainer = MagicMock()
        mock_train_result = MagicMock()
        mock_train_result.metrics = {"train_loss": 0.5}
        mock_trainer.train.return_value = mock_train_result
        mock_eval_result = {
            "eval_loss": 0.3,
        }
        mock_trainer.evaluate.return_value = mock_eval_result

        with (
            patch("transformers.AutoModelForCausalLM") as mock_model_cls,
            patch("transformers.AutoTokenizer") as mock_tokenizer_cls,
            patch("datasets.Dataset.from_list") as mock_from_list,
            patch("peft.LoraConfig"),
            patch("peft.get_peft_model", return_value=mock_model),
            patch("trl.SFTTrainer", return_value=mock_trainer) as mock_sft_trainer,
            patch("pathlib.Path.mkdir"),
        ):
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            mock_from_list.side_effect = [
                mock_train_dataset,
                mock_val_dataset,
            ]

            # No try/except: _train_local must run to completion against these
            # mocks. A swallowed exception here hid the evaluation_strategy
            # TypeError (transformers renamed it) for the whole local-train path.
            await finetuner._train_local(dataset, config)

            # Verify Dataset.from_list was called TWICE (train + val split)
            assert mock_from_list.call_count == 2

            # Verify the split: 90% train (135), 10% val (15)
            split_idx = int(150 * 0.9)  # 135
            train_call_args = mock_from_list.call_args_list[0][0][0]
            val_call_args = mock_from_list.call_args_list[1][0][0]

            assert len(train_call_args) == split_idx  # 135
            assert len(val_call_args) == 150 - split_idx  # 15

            # SFTTrainer must have been constructed with a real eval_dataset.
            assert mock_sft_trainer.called
            trainer_call_kwargs = mock_sft_trainer.call_args[1]
            assert trainer_call_kwargs.get("eval_dataset") is not None


@pytest.mark.unit
class TestLoRAFallback:
    """Test LoRA fallback logic for SFT trainer"""

    @pytest.mark.asyncio
    async def test_lora_success_path(self):
        """Test that LoRA is applied successfully when use_lora=True"""
        finetuner = SFTFinetuner(
            base_model="HuggingFaceTB/SmolLM-135M", output_dir="/tmp/output"
        )

        dataset = [{"text": f"Example {i}"} for i in range(50)]
        config = {"use_lora": True, "epochs": 1, "learning_rate": 2e-4}

        # Mock all dependencies
        mock_base_model = MagicMock()
        mock_base_model.config.eos_token_id = 0
        mock_peft_model = MagicMock()
        mock_peft_model.print_trainable_parameters = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token = None
        mock_dataset = MagicMock()
        mock_trainer = MagicMock()
        mock_train_result = MagicMock()
        mock_train_result.metrics = {"train_loss": 0.5}
        mock_trainer.train.return_value = mock_train_result
        mock_lora_config = MagicMock()

        with (
            patch("transformers.AutoModelForCausalLM") as mock_model_cls,
            patch("transformers.AutoTokenizer") as mock_tokenizer_cls,
            patch("datasets.Dataset.from_list") as mock_from_list,
            patch(
                "peft.LoraConfig", return_value=mock_lora_config
            ) as mock_lora_config_cls,
            patch(
                "peft.get_peft_model", return_value=mock_peft_model
            ) as mock_get_peft_model_func,
            patch("trl.SFTTrainer", return_value=mock_trainer),
            patch("pathlib.Path.mkdir"),
        ):
            mock_model_cls.from_pretrained.return_value = mock_base_model
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            mock_from_list.return_value = mock_dataset

            # No try/except: _train_local must run to completion against these
            # mocks. A swallowed exception here hid the evaluation_strategy
            # TypeError (transformers renamed it) for the whole local-train path.
            await finetuner._train_local(dataset, config)

            # Verify LoraConfig was created
            mock_lora_config_cls.assert_called_once()

            # Verify get_peft_model was called with base model and lora config
            mock_get_peft_model_func.assert_called_once_with(
                mock_base_model, mock_lora_config
            )

            # Verify print_trainable_parameters was called (indicates LoRA was applied)
            mock_peft_model.print_trainable_parameters.assert_called_once()

    @pytest.mark.asyncio
    async def test_lora_fallback_on_error(self, caplog):
        """Test that when LoRA fails, it logs a warning and continues with full fine-tuning"""
        finetuner = SFTFinetuner(
            base_model="HuggingFaceTB/SmolLM-135M", output_dir="/tmp/output"
        )

        dataset = [{"text": f"Example {i}"} for i in range(50)]
        config = {"use_lora": True, "epochs": 1, "learning_rate": 2e-4}

        # Mock all dependencies
        mock_model = MagicMock()
        mock_model.config.eos_token_id = 0
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token = None
        mock_dataset = MagicMock()
        mock_trainer = MagicMock()
        mock_train_result = MagicMock()
        mock_train_result.metrics = {"train_loss": 0.5}
        mock_trainer.train.return_value = mock_train_result

        with (
            patch("transformers.AutoModelForCausalLM") as mock_model_cls,
            patch("transformers.AutoTokenizer") as mock_tokenizer_cls,
            patch("datasets.Dataset.from_list") as mock_from_list,
            patch("peft.LoraConfig"),
            patch(
                "peft.get_peft_model", side_effect=Exception("LoRA not supported")
            ) as mock_get_peft_model_func,
            patch("trl.SFTTrainer", return_value=mock_trainer) as mock_sft_trainer,
            patch("pathlib.Path.mkdir"),
            caplog.at_level("WARNING"),
        ):
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            mock_from_list.return_value = mock_dataset

            # No try/except: _train_local must run to completion against these
            # mocks. A swallowed exception here hid the evaluation_strategy
            # TypeError (transformers renamed it) for the whole local-train path.
            await finetuner._train_local(dataset, config)

            # Verify get_peft_model was called (and raised exception)
            mock_get_peft_model_func.assert_called_once()

            # Verify warning was logged
            assert any(
                "Failed to apply LoRA" in record.message for record in caplog.records
            )
            assert any(
                "continuing with full fine-tuning" in record.message
                for record in caplog.records
            )

            # SFTTrainer must have been constructed with the base model (LoRA
            # failed, so training continues without the PEFT wrapper).
            assert mock_sft_trainer.called
            trainer_call_kwargs = mock_sft_trainer.call_args[1]
            assert trainer_call_kwargs.get("model") == mock_model

    @pytest.mark.asyncio
    async def test_lora_disabled_via_config(self):
        """Test that LoRA is skipped when use_lora=False"""
        finetuner = SFTFinetuner(
            base_model="HuggingFaceTB/SmolLM-135M", output_dir="/tmp/output"
        )

        dataset = [{"text": f"Example {i}"} for i in range(50)]
        config = {"use_lora": False, "epochs": 1, "learning_rate": 2e-4}

        # Mock all dependencies
        mock_model = MagicMock()
        mock_model.config.eos_token_id = 0
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token = None
        mock_dataset = MagicMock()
        mock_trainer = MagicMock()
        mock_train_result = MagicMock()
        mock_train_result.metrics = {"train_loss": 0.5}
        mock_trainer.train.return_value = mock_train_result

        with (
            patch("transformers.AutoModelForCausalLM") as mock_model_cls,
            patch("transformers.AutoTokenizer") as mock_tokenizer_cls,
            patch("datasets.Dataset.from_list") as mock_from_list,
            patch("peft.LoraConfig") as mock_lora_config_cls,
            patch("peft.get_peft_model") as mock_get_peft_model_func,
            patch("trl.SFTTrainer", return_value=mock_trainer) as mock_sft_trainer,
            patch("pathlib.Path.mkdir"),
        ):
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            mock_from_list.return_value = mock_dataset

            # No try/except: _train_local must run to completion against these
            # mocks. A swallowed exception here hid the evaluation_strategy
            # TypeError (transformers renamed it) for the whole local-train path.
            await finetuner._train_local(dataset, config)

            # Verify LoraConfig was NOT called
            mock_lora_config_cls.assert_not_called()

            # Verify get_peft_model was NOT called
            mock_get_peft_model_func.assert_not_called()

            # SFTTrainer must have been constructed with the base model.
            assert mock_sft_trainer.called
            trainer_call_kwargs = mock_sft_trainer.call_args[1]
            assert trainer_call_kwargs.get("model") == mock_model
