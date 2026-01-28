"""
Supervised Fine-Tuning (SFT) with LoRA for LLM agents.

Uses TRL SFTTrainer with PEFT/LoRA for parameter-efficient fine-tuning.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """Configuration for supervised fine-tuning."""

    # Model
    base_model: str  # "HuggingFaceTB/SmolLM-135M", "Qwen/Qwen2.5-3B"

    # LoRA config
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = None  # ["q_proj", "v_proj", "k_proj", "o_proj"]

    # Training hyperparameters
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_seq_length: int = 512
    fp16: bool = True  # Mixed precision training

    # Dataset format
    dataset_text_field: str = "text"  # Field for TRL SFTTrainer
    format: str = "alpaca_text"  # "alpaca_text", "chatml", "sharegpt"

    # Output
    output_dir: str = "outputs/sft_adapters"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100

    def __post_init__(self):
        if self.target_modules is None:
            # Default: target all linear layers in attention
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


@dataclass
class SFTResult:
    """Results from SFT training."""

    adapter_path: str
    metrics: Dict
    base_model: str
    lora_config: Dict


class SFTFinetuner:
    """
    Supervised fine-tuning for LLM agents.

    Uses TRL SFTTrainer with LoRA adapters for efficient training
    on instruction-response pairs.
    """

    def __init__(self, base_model: str, output_dir: str):
        """
        Initialize with base model and output directory.

        Args:
            base_model: Base model name (e.g., "HuggingFaceTB/SmolLM-135M")
            output_dir: Output directory for adapter
        """
        self.base_model = base_model
        self.output_dir = output_dir

    async def train(
        self,
        dataset: List[Dict],
        config: Dict,
    ) -> Dict:
        """
        Train model with supervised fine-tuning.

        Args:
            dataset: Training dataset as List[Dict] with "text" field (Alpaca format)
            config: Training configuration dict with keys:
                - use_lora: bool (default True)
                - lora_r: int (default 8)
                - lora_alpha: int (default 16)
                - epochs: int (default 3)
                - batch_size: int (default 4)
                - learning_rate: float (default 2e-4)
                - dataset_text_field: str (default "text")

        Returns:
            Dict with adapter_path and metrics
        """
        return await self._train_local(dataset, config)

    async def _train_local(
        self,
        dataset: List[Dict],
        config: Dict,
    ) -> Dict:
        """Train locally with TRL SFTTrainer."""
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer

        logger.info(f"Training {self.base_model} with {len(dataset)} examples...")

        # 1. Create dataset (already formatted with "text" field)
        # Add validation split for larger datasets (>100 examples)
        if len(dataset) > 100:
            # Use 90/10 split for train/val
            split_idx = int(len(dataset) * 0.9)
            train_data = dataset[:split_idx]
            val_data = dataset[split_idx:]

            train_dataset = Dataset.from_list(train_data)
            val_dataset = Dataset.from_list(val_data)

            logger.info(
                f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} validation examples"
            )
        else:
            train_dataset = Dataset.from_list(dataset)
            val_dataset = None
            logger.info(
                f"Training on {len(train_dataset)} examples (no validation split)"
            )

        # 2. Load model and tokenizer
        fp16 = config.get("fp16", True)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if fp16 else torch.float32,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        # 3. Apply LoRA if enabled
        use_lora = config.get("use_lora", True)
        if use_lora:
            try:
                lora_config = LoraConfig(
                    r=config.get("lora_r", 8),
                    lora_alpha=config.get("lora_alpha", 16),
                    target_modules=config.get(
                        "target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]
                    ),
                    lora_dropout=config.get("lora_dropout", 0.1),
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()
                logger.info("LoRA applied successfully")
            except Exception as e:
                logger.warning(
                    f"Failed to apply LoRA (continuing with full fine-tuning): {e}. "
                    "This may happen if the model architecture doesn't have the expected linear layers. "
                    "Training will proceed with full model fine-tuning instead."
                )
                # Continue with full fine-tuning (no LoRA)

        # 4. Training arguments
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=config.get("epochs", 3),
            per_device_train_batch_size=config.get("batch_size", 4),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            learning_rate=config.get("learning_rate", 2e-4),
            warmup_steps=config.get("warmup_steps", 100),
            fp16=fp16,
            logging_steps=config.get("logging_steps", 100),
            save_steps=config.get("save_steps", 500),
            eval_steps=config.get("eval_steps", 500) if val_dataset else None,
            evaluation_strategy="steps" if val_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False if val_dataset else None,
            report_to="none",
        )

        # 5. Early stopping callback (if validation enabled)
        callbacks = []
        if val_dataset:
            from transformers import EarlyStoppingCallback

            # Stop if eval_loss doesn't improve for 3 evaluations
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.0,
            )
            callbacks.append(early_stopping)
            logger.info("Early stopping enabled (patience=3)")

        # 6. Create SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            dataset_text_field=config.get("dataset_text_field", "text"),
            max_seq_length=config.get("max_seq_length", 512),
            callbacks=callbacks,
        )

        # 7. Train
        logger.info("Starting training...")
        train_result = trainer.train()

        # 8. Save adapter
        trainer.save_model(str(output_path))
        tokenizer.save_pretrained(str(output_path))

        logger.info(f"Training complete. Adapter saved to {output_path}")

        # 9. Collect metrics and return
        metrics = {
            "train_loss": train_result.metrics.get("train_loss"),
            "train_samples": train_result.metrics.get("train_samples"),
            "total_examples": len(dataset),
            "train_examples": (
                len(train_dataset)
                if isinstance(train_dataset, Dataset)
                else len(dataset)
            ),
            "epochs": config.get("epochs", 3),
            "batch_size": config.get("batch_size", 4),
            "learning_rate": config.get("learning_rate", 2e-4),
            "max_seq_length": config.get("max_seq_length", 512),
        }

        # Add validation metrics if validation was used
        if val_dataset:
            eval_result = trainer.evaluate()
            metrics["eval_loss"] = eval_result.get("eval_loss")
            metrics["eval_samples"] = eval_result.get("eval_samples")
            metrics["val_examples"] = len(val_dataset)
            metrics["used_validation_split"] = True
            logger.info(f"Validation loss: {metrics['eval_loss']:.4f}")
        else:
            metrics["used_validation_split"] = False

        return {
            "adapter_path": str(output_path),
            "metrics": metrics,
        }
