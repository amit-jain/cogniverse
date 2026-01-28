"""
Modal app for GPU training.

Deploy this file to Modal:
    modal deploy libs/finetuning/cogniverse_finetuning/training/modal_app.py

Then call from Python:
    from cogniverse_finetuning.training import modal_app
    result = modal_app.train_sft_remote.remote(...)
"""

import logging

import modal

logger = logging.getLogger(__name__)

# Create Modal app
app = modal.App("cogniverse-finetuning")

# Create container image with all dependencies
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "transformers>=4.50.0",
    "trl>=0.12.0",
    "peft>=0.13.0",
    "accelerate>=1.2.0",
    "torch>=2.5.0",
    "datasets>=3.2.0",
    "bitsandbytes>=0.45.0",
    "sentence-transformers>=3.3.0",
    "boto3>=1.28.0",
    "huggingface-hub>=0.28.0",
)

# Create volume for caching models
volume = modal.Volume.from_name("finetuning-models", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",  # Can be overridden: T4, A10G, A100-40GB, A100-80GB, H100
    cpu=4,
    memory=16384,
    timeout=3600,  # 1 hour
    volumes={"/cache": volume},
)
def train_sft_remote(
    dataset: list[dict],  # Pass dataset directly, no S3 needed!
    base_model: str,
    config: dict,
) -> dict:
    """
    Train SFT model on Modal GPU.

    Args:
        dataset: List of dicts (formatted training data)
        base_model: Base model name
        config: Training configuration dict

    Returns:
        Dict with adapter (as bytes) and metrics
    """
    import tarfile
    from io import BytesIO

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    logger.info(f"Starting SFT training: model={base_model}, examples={len(dataset)}")

    # 1. Load dataset (passed directly, no download needed!)
    dataset = Dataset.from_list(dataset)
    logger.info(f"Loaded {len(dataset)} examples")

    # 2. Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if config.get("fp16", True) else torch.float32,
        device_map="auto",
        cache_dir="/cache",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir="/cache")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    logger.info(f"Loaded model: {base_model}")

    # 3. Apply LoRA
    if config.get("use_lora", True):
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

    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir="/tmp/adapter",
        num_train_epochs=config.get("epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 2e-4),
        warmup_steps=config.get("warmup_steps", 100),
        fp16=config.get("fp16", True),
        logging_steps=config.get("logging_steps", 100),
        save_steps=config.get("save_steps", 500),
        save_total_limit=3,
        report_to="none",
    )

    # 5. Train
    logger.info("Starting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field=config.get("dataset_text_field", "text"),
        max_seq_length=config.get("max_seq_length", 512),
    )

    train_result = trainer.train()
    trainer.save_model("/tmp/adapter")
    tokenizer.save_pretrained("/tmp/adapter")

    logger.info(f"Training complete. Loss: {train_result.metrics.get('train_loss')}")

    # 6. Create tar.gz of adapter in memory
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add("/tmp/adapter", arcname="adapter")

    adapter_bytes = buffer.getvalue()
    logger.info(f"Adapter size: {len(adapter_bytes) / 1024 / 1024:.2f} MB")

    # 7. Return adapter as bytes (Modal handles transfer automatically)
    return {
        "adapter_bytes": adapter_bytes,  # Modal serializes this automatically
        "metrics": {
            "train_loss": train_result.metrics.get("train_loss"),
            "total_examples": len(dataset),
            "epochs": config.get("epochs", 3),
        },
    }


@app.function(
    image=image,
    gpu="A10G",
    cpu=4,
    memory=16384,
    timeout=3600,
    volumes={"/cache": volume},
)
def train_dpo_remote(
    dataset: list[dict],  # Pass dataset directly, no S3 needed!
    base_model: str,
    config: dict,
) -> dict:
    """
    Train DPO model on Modal GPU.

    Args:
        dataset: List of dicts with prompt/chosen/rejected fields
        base_model: Base model name
        config: Training configuration dict

    Returns:
        Dict with adapter (as bytes) and metrics
    """
    import tarfile
    from io import BytesIO

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import DPOTrainer

    logger.info(f"Starting DPO training: model={base_model}, pairs={len(dataset)}")

    # 1. Load dataset (passed directly, no download needed!)
    dataset = Dataset.from_list(dataset)
    logger.info(f"Loaded {len(dataset)} preference pairs")

    # Load models (trainable + reference)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if config.get("fp16", True) else torch.float32,
        device_map="auto",
        cache_dir="/cache",
    )
    model_ref = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if config.get("fp16", True) else torch.float32,
        device_map="auto",
        cache_dir="/cache",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir="/cache")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        model_ref.config.pad_token_id = model_ref.config.eos_token_id

    # Apply LoRA (only to trainable model)
    if config.get("use_lora", True):
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

    # Training arguments
    training_args = TrainingArguments(
        output_dir="/tmp/adapter",
        num_train_epochs=config.get("epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 5e-5),
        warmup_steps=config.get("warmup_steps", 100),
        fp16=config.get("fp16", True),
        logging_steps=config.get("logging_steps", 100),
        save_steps=config.get("save_steps", 500),
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
    )

    # DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        beta=config.get("beta", 0.1),
        max_length=config.get("max_seq_length", 512),
        max_prompt_length=config.get("max_prompt_length", 256),
    )

    train_result = trainer.train()
    trainer.save_model("/tmp/adapter")
    tokenizer.save_pretrained("/tmp/adapter")

    logger.info(f"Training complete. Loss: {train_result.metrics.get('train_loss')}")

    # 6. Create tar.gz of adapter in memory
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add("/tmp/adapter", arcname="adapter")

    adapter_bytes = buffer.getvalue()
    logger.info(f"Adapter size: {len(adapter_bytes) / 1024 / 1024:.2f} MB")

    # 7. Return adapter as bytes (Modal handles transfer automatically)
    return {
        "adapter_bytes": adapter_bytes,  # Modal serializes this automatically
        "metrics": {
            "train_loss": train_result.metrics.get("train_loss"),
            "total_pairs": len(dataset),
            "epochs": config.get("epochs", 3),
            "beta": config.get("beta", 0.1),
        },
    }


@app.function(
    image=image,
    gpu="A10G",
    cpu=4,
    memory=16384,
    timeout=3600,
    volumes={"/cache": volume},
)
def train_embedding_remote(
    dataset: list[dict],  # Pass dataset directly, no S3 needed!
    base_model: str,
    config: dict,
) -> dict:
    """
    Train embedding model on Modal GPU.

    Args:
        dataset: List of dicts with anchor/positive/negative fields
        base_model: Base embedding model name
        config: Training configuration dict

    Returns:
        Dict with adapter (as bytes) and metrics
    """
    import tarfile
    from io import BytesIO

    from sentence_transformers import (
        InputExample,
        SentenceTransformer,
        losses,
    )
    from torch.utils.data import DataLoader

    logger.info(
        f"Starting embedding training: model={base_model}, triplets={len(dataset)}"
    )

    # 1. Convert to InputExample format (passed directly, no download!)
    train_examples = [
        InputExample(texts=[item["anchor"], item["positive"], item["negative"]])
        for item in dataset
    ]
    logger.info(f"Loaded {len(train_examples)} triplets")

    # Load model
    model = SentenceTransformer(base_model, cache_folder="/cache")

    # Create dataloader
    train_dataloader = DataLoader(
        train_examples,
        batch_size=config.get("batch_size", 16),
        shuffle=True,
    )

    # Triplet loss
    distance_metric_map = {
        "cosine": losses.TripletDistanceMetric.COSINE,
        "euclidean": losses.TripletDistanceMetric.EUCLIDEAN,
    }
    distance_metric = distance_metric_map.get(
        config.get("distance_metric", "cosine"),
        losses.TripletDistanceMetric.COSINE,
    )

    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=distance_metric,
        triplet_margin=config.get("triplet_margin", 0.5),
    )

    # 2. Train
    logger.info("Starting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config.get("epochs", 3),
        warmup_steps=config.get("warmup_steps", 100),
        output_path="/tmp/adapter",
        save_best_model=True,
        show_progress_bar=True,
    )

    logger.info("Training complete")

    # 3. Create tar.gz of adapter in memory
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add("/tmp/adapter", arcname="adapter")

    adapter_bytes = buffer.getvalue()
    logger.info(f"Adapter size: {len(adapter_bytes) / 1024 / 1024:.2f} MB")

    # 4. Return adapter as bytes (Modal handles transfer automatically)
    return {
        "adapter_bytes": adapter_bytes,  # Modal serializes this automatically
        "metrics": {
            "total_triplets": len(train_examples),
            "epochs": config.get("epochs", 3),
            "triplet_margin": config.get("triplet_margin", 0.5),
        },
    }
