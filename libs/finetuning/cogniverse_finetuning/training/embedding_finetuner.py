"""
Embedding model fine-tuning with LoRA and contrastive learning.

Uses sentence-transformers with triplet loss for training
embedding models (text, vision, multi-modal).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingTrainingConfig:
    """Configuration for embedding model training."""

    # Model
    base_model: str  # "jinaai/jina-embeddings-v3", "sentence-transformers/all-MiniLM-L6-v2"

    # LoRA config
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = None  # ["q_proj", "v_proj"] for attention layers

    # Training hyperparameters
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    evaluation_steps: int = 500
    save_steps: int = 1000

    # Triplet loss config
    triplet_margin: float = 0.5  # Separation between positive and negative
    distance_metric: Literal["cosine", "euclidean", "manhattan"] = "cosine"

    # Output
    output_dir: str = "outputs/embedding_adapters"

    def __post_init__(self):
        if self.target_modules is None:
            # Default: target attention layers
            self.target_modules = ["q_proj", "v_proj"]


@dataclass
class EmbeddingTrainingResult:
    """Results from embedding model training."""

    adapter_path: str
    metrics: Dict
    base_model: str
    lora_config: Dict


class EmbeddingFinetuner:
    """
    Fine-tune embedding models with contrastive learning.

    Supports:
    - Text embeddings (jina-embeddings-v3, e5-mistral)
    - Vision embeddings (ColPali, VideoPrism via sentence-transformers wrapper)
    - Multi-modal embeddings (CLIP, SigLIP)

    Training:
    - Triplet loss: max(0, margin + sim(anchor, neg) - sim(anchor, pos))
    - LoRA adapters for parameter-efficient fine-tuning
    - Optional Modal GPU training
    """

    def __init__(self, base_model: str, output_dir: str):
        """
        Initialize with base model and output directory.

        Args:
            base_model: Base model name (e.g., "jinaai/jina-embeddings-v3")
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
        Train embedding model with triplet loss.

        Args:
            dataset: Training dataset as List[Dict] with anchor/positive/negative fields
            config: Training configuration dict with keys:
                - use_lora: bool (default True)
                - lora_r: int (default 8)
                - lora_alpha: int (default 16)
                - epochs: int (default 3)
                - batch_size: int (default 16)
                - learning_rate: float (default 2e-5)
                - triplet_margin: float (default 0.5)
                - distance_metric: str (default "cosine")

        Returns:
            Dict with adapter_path and metrics
        """
        return await self._train_local(dataset, config)

    async def _train_local(
        self,
        dataset: List[Dict],
        config: Dict,
    ) -> Dict:
        """Train embedding model locally."""
        from sentence_transformers import (
            InputExample,
            SentenceTransformer,
        )
        from torch.utils.data import DataLoader

        logger.info(
            f"Training {self.base_model} with {len(dataset)} triplets..."
        )

        # 1. Load base model
        model = SentenceTransformer(self.base_model)

        # 2. Apply LoRA if enabled
        use_lora = config.get("use_lora", True)
        if use_lora:
            model = self._apply_lora(
                model,
                lora_r=config.get("lora_r", 8),
                lora_alpha=config.get("lora_alpha", 16),
                lora_dropout=config.get("lora_dropout", 0.1),
                target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
            )
            logger.info(
                f"Applied LoRA: r={config.get('lora_r', 8)}, "
                f"alpha={config.get('lora_alpha', 16)}, "
                f"modules={config.get('target_modules', ['q_proj', 'v_proj'])}"
            )

        # 3. Prepare training data (convert List[Dict] to InputExample)
        train_examples = [
            InputExample(texts=[item["anchor"], item["positive"], item["negative"]])
            for item in dataset
        ]
        train_dataloader = DataLoader(
            train_examples,
            batch_size=config.get("batch_size", 16),
            shuffle=True,
        )

        # 4. Create triplet loss
        train_loss = self._create_triplet_loss(
            model,
            triplet_margin=config.get("triplet_margin", 0.5),
            distance_metric=config.get("distance_metric", "cosine"),
        )

        # 5. Train
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=config.get("epochs", 3),
            warmup_steps=config.get("warmup_steps", 100),
            output_path=str(output_path),
            save_best_model=True,
            show_progress_bar=True,
        )

        logger.info(f"Training complete. Adapter saved to {output_path}")

        # 6. Collect metrics and return
        metrics = {
            "total_triplets": len(dataset),
            "epochs": config.get("epochs", 3),
            "batch_size": config.get("batch_size", 16),
            "learning_rate": config.get("learning_rate", 2e-5),
            "triplet_margin": config.get("triplet_margin", 0.5),
        }

        return {
            "adapter_path": str(output_path),
            "metrics": metrics,
        }

    def _apply_lora(self, model, lora_r: int, lora_alpha: int, lora_dropout: float, target_modules: List[str]):
        """Apply LoRA adapters to model."""
        try:
            from peft import LoraConfig, get_peft_model

            # Create LoRA config
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",  # Embedding models
            )

            # Apply to model
            model = get_peft_model(model, lora_config)

            return model

        except Exception as e:
            logger.warning(
                f"Failed to apply LoRA (continuing without): {e}. "
                "This may be because sentence-transformers doesn't directly "
                "support PEFT. Consider using transformers AutoModel instead."
            )
            return model

    def _create_triplet_loss(self, model, triplet_margin: float, distance_metric: str):
        """Create triplet loss function."""
        from sentence_transformers import losses

        # Map distance metric
        distance_metric_map = {
            "cosine": losses.TripletDistanceMetric.COSINE,
            "euclidean": losses.TripletDistanceMetric.EUCLIDEAN,
            "manhattan": losses.TripletDistanceMetric.MANHATTAN,
        }

        metric = distance_metric_map.get(
            distance_metric,
            losses.TripletDistanceMetric.COSINE,
        )

        return losses.TripletLoss(
            model=model,
            distance_metric=metric,
            triplet_margin=triplet_margin,
        )
