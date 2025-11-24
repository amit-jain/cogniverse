"""
Direct Preference Optimization (DPO) with LoRA for LLM agents.

Uses TRL DPOTrainer with PEFT/LoRA for preference-based fine-tuning.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """Configuration for DPO training."""

    # Model
    base_model: str  # "HuggingFaceTB/SmolLM-135M", "Qwen/Qwen2.5-3B"

    # LoRA config
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = None  # ["q_proj", "v_proj", "k_proj", "o_proj"]

    # DPO hyperparameters
    beta: float = 0.1  # KL penalty coefficient (lower = more exploration)
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5  # Lower LR for DPO
    warmup_steps: int = 100
    max_seq_length: int = 512
    max_prompt_length: int = 256
    fp16: bool = True

    # Output
    output_dir: str = "outputs/dpo_adapters"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


@dataclass
class DPOResult:
    """Results from DPO training."""

    adapter_path: str
    metrics: Dict
    base_model: str
    lora_config: Dict


class DPOFinetuner:
    """
    Direct Preference Optimization for LLM agents.

    Uses TRL DPOTrainer with LoRA adapters to learn from
    preference pairs (chosen vs rejected responses).

    More sample-efficient than SFT when preference data available.
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
        Train model with DPO.

        Args:
            dataset: Training dataset as List[Dict] with prompt/chosen/rejected fields
            config: Training configuration dict with keys:
                - use_lora: bool (default True)
                - lora_r: int (default 8)
                - lora_alpha: int (default 16)
                - epochs: int (default 3)
                - batch_size: int (default 4)
                - learning_rate: float (default 5e-5)
                - beta: float (default 0.1)

        Returns:
            Dict with adapter_path and metrics
        """
        return await self._train_local(dataset, config)

    async def _train_local(
        self,
        dataset: List[Dict],
        config: Dict,
    ) -> Dict:
        """Train locally with TRL DPOTrainer."""
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import DPOTrainer

        logger.info(
            f"Training {self.base_model} with {len(dataset)} preference pairs..."
        )

        # 1. Create dataset (already formatted with prompt/chosen/rejected fields)
        # Add validation split for larger datasets (>100 pairs)
        if len(dataset) > 100:
            # Use 90/10 split for train/val
            split_idx = int(len(dataset) * 0.9)
            train_data = dataset[:split_idx]
            val_data = dataset[split_idx:]

            train_dataset = Dataset.from_list(train_data)
            val_dataset = Dataset.from_list(val_data)

            logger.info(
                f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} validation pairs"
            )
        else:
            train_dataset = Dataset.from_list(dataset)
            val_dataset = None
            logger.info(f"Training on {len(train_dataset)} preference pairs (no validation split)")

        # 2. Load model and tokenizer
        fp16 = config.get("fp16", True)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if fp16 else torch.float32,
            device_map="auto",
        )

        # DPOTrainer requires a reference model (frozen copy)
        model_ref = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if fp16 else torch.float32,
            device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
            model_ref.config.pad_token_id = model_ref.config.eos_token_id

        # 3. Apply LoRA if enabled (only to trainable model, not reference)
        use_lora = config.get("use_lora", True)
        if use_lora:
            try:
                lora_config = LoraConfig(
                    r=config.get("lora_r", 8),
                    lora_alpha=config.get("lora_alpha", 16),
                    target_modules=config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
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
            learning_rate=config.get("learning_rate", 5e-5),
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
            remove_unused_columns=False,  # DPOTrainer needs all columns
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

        # 6. Create DPOTrainer
        beta = config.get("beta", 0.1)
        trainer = DPOTrainer(
            model=model,
            ref_model=model_ref,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            beta=beta,
            max_length=config.get("max_seq_length", 512),
            max_prompt_length=config.get("max_prompt_length", 256),
            callbacks=callbacks,
        )

        # 7. Train
        logger.info(f"Starting DPO training (beta={beta})...")
        train_result = trainer.train()

        # 8. Save adapter
        trainer.save_model(str(output_path))
        tokenizer.save_pretrained(str(output_path))

        logger.info(f"Training complete. Adapter saved to {output_path}")

        # 9. Collect metrics and return
        metrics = {
            "train_loss": train_result.metrics.get("train_loss"),
            "train_samples": train_result.metrics.get("train_samples"),
            "total_pairs": len(dataset),
            "train_pairs": len(train_dataset) if isinstance(train_dataset, Dataset) else len(dataset),
            "epochs": config.get("epochs", 3),
            "batch_size": config.get("batch_size", 4),
            "learning_rate": config.get("learning_rate", 5e-5),
            "beta": beta,
            "max_seq_length": config.get("max_seq_length", 512),
        }

        # Add validation metrics if validation was used
        if val_dataset:
            eval_result = trainer.evaluate()
            metrics["eval_loss"] = eval_result.get("eval_loss")
            metrics["eval_samples"] = eval_result.get("eval_samples")
            metrics["val_pairs"] = len(val_dataset)
            # DPO-specific metrics
            metrics["eval_reward_accuracy"] = eval_result.get("rewards/accuracies")
            metrics["eval_reward_margin"] = eval_result.get("rewards/margins")
            metrics["used_validation_split"] = True
            logger.info(
                f"Validation loss: {metrics['eval_loss']:.4f}, "
                f"Reward accuracy: {metrics['eval_reward_accuracy']:.4f}"
            )
        else:
            metrics["used_validation_split"] = False

        return {
            "adapter_path": str(output_path),
            "metrics": metrics,
        }
