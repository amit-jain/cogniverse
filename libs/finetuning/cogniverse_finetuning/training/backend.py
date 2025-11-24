"""
Training backend abstraction.

Provides a unified interface for running training jobs on different backends:
- Local: CPU/GPU training on local machine
- Remote: Cloud GPU training (Modal, AWS SageMaker, Azure ML, etc.)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingJobConfig:
    """Configuration for training job."""

    # GPU configuration (for remote backends)
    gpu: str = "A10G"  # Options: T4, A10G, A100-40GB, A100-80GB, H100
    gpu_count: int = 1
    cpu: int = 4
    memory: int = 16384  # MB

    # Timeouts
    timeout: int = 3600  # 1 hour default


@dataclass
class TrainingJobResult:
    """Result from training job."""

    job_id: str
    adapter_path: str
    metrics: Dict
    logs_url: Optional[str] = None


class TrainingBackend(ABC):
    """
    Abstract interface for training backends.

    Implementations:
    - LocalTrainingBackend: Uses local trainers (SFTFinetuner, DPOFinetuner, EmbeddingFinetuner)
    - RemoteTrainingBackend: Delegates to cloud GPU service (Modal, SageMaker, etc.)
    """

    @abstractmethod
    async def train_sft(
        self,
        dataset: List[Dict],
        base_model: str,
        output_dir: str,
        config: Dict,
    ) -> TrainingJobResult:
        """Run SFT training."""
        pass

    @abstractmethod
    async def train_dpo(
        self,
        dataset: List[Dict],
        base_model: str,
        output_dir: str,
        config: Dict,
    ) -> TrainingJobResult:
        """Run DPO training."""
        pass

    @abstractmethod
    async def train_embedding(
        self,
        dataset: List[Dict],
        base_model: str,
        output_dir: str,
        config: Dict,
    ) -> TrainingJobResult:
        """Run embedding training."""
        pass


class LocalTrainingBackend(TrainingBackend):
    """
    Local training backend.

    Uses local GPU/CPU with SFTFinetuner, DPOFinetuner, EmbeddingFinetuner.
    """

    def __init__(self, config: TrainingJobConfig):
        """
        Initialize local backend.

        Args:
            config: Training job configuration
        """
        self.config = config

    async def train_sft(
        self,
        dataset: List[Dict],
        base_model: str,
        output_dir: str,
        config: Dict,
    ) -> TrainingJobResult:
        """Run SFT training locally."""
        from cogniverse_finetuning.training.sft_trainer import SFTFinetuner

        logger.info(f"Starting local SFT training: {len(dataset)} examples")

        trainer = SFTFinetuner(base_model=base_model, output_dir=output_dir)
        result = await trainer.train(dataset=dataset, config=config)

        return TrainingJobResult(
            job_id="local",
            adapter_path=result["adapter_path"],
            metrics=result["metrics"],
            logs_url=None,
        )

    async def train_dpo(
        self,
        dataset: List[Dict],
        base_model: str,
        output_dir: str,
        config: Dict,
    ) -> TrainingJobResult:
        """Run DPO training locally."""
        from cogniverse_finetuning.training.dpo_trainer import DPOFinetuner

        logger.info(f"Starting local DPO training: {len(dataset)} pairs")

        trainer = DPOFinetuner(base_model=base_model, output_dir=output_dir)
        result = await trainer.train(dataset=dataset, config=config)

        return TrainingJobResult(
            job_id="local",
            adapter_path=result["adapter_path"],
            metrics=result["metrics"],
            logs_url=None,
        )

    async def train_embedding(
        self,
        dataset: List[Dict],
        base_model: str,
        output_dir: str,
        config: Dict,
    ) -> TrainingJobResult:
        """Run embedding training locally."""
        from cogniverse_finetuning.training.embedding_finetuner import (
            EmbeddingFinetuner,
        )

        logger.info(f"Starting local embedding training: {len(dataset)} triplets")

        trainer = EmbeddingFinetuner(base_model=base_model, output_dir=output_dir)
        result = await trainer.train(dataset=dataset, config=config)

        return TrainingJobResult(
            job_id="local",
            adapter_path=result["adapter_path"],
            metrics=result["metrics"],
            logs_url=None,
        )


class RemoteTrainingBackend(TrainingBackend):
    """
    Remote training backend.

    Delegates to cloud GPU service. Currently supports Modal,
    but can be extended for AWS SageMaker, Azure ML, etc.
    """

    def __init__(
        self,
        config: TrainingJobConfig,
        provider: Literal["modal"] = "modal",
    ):
        """
        Initialize remote backend.

        Args:
            config: Training job configuration
            provider: Cloud provider (modal, sagemaker, azure_ml, etc.)
        """
        self.config = config
        self.provider = provider

        if provider == "modal":
            from cogniverse_finetuning.training.modal_runner import (
                ModalJobConfig,
                ModalTrainingRunner,
            )

            modal_config = ModalJobConfig(
                gpu=config.gpu,
                gpu_count=config.gpu_count,
                cpu=config.cpu,
                memory=config.memory,
                timeout=config.timeout,
            )
            self.runner = ModalTrainingRunner(modal_config)
        else:
            raise ValueError(f"Unsupported remote provider: {provider}")

    async def train_sft(
        self,
        dataset: List[Dict],
        base_model: str,
        output_dir: str,
        config: Dict,
    ) -> TrainingJobResult:
        """Run SFT training on remote GPU."""
        logger.info(
            f"Starting remote SFT training ({self.provider}): {len(dataset)} examples"
        )

        result = await self.runner.run_sft(
            dataset=dataset,
            base_model=base_model,
            output_dir=output_dir,
            sft_config=config,
        )

        return TrainingJobResult(
            job_id=result.job_id,
            adapter_path=result.adapter_path,
            metrics=result.metrics,
            logs_url=result.logs_url,
        )

    async def train_dpo(
        self,
        dataset: List[Dict],
        base_model: str,
        output_dir: str,
        config: Dict,
    ) -> TrainingJobResult:
        """Run DPO training on remote GPU."""
        logger.info(
            f"Starting remote DPO training ({self.provider}): {len(dataset)} pairs"
        )

        result = await self.runner.run_dpo(
            dataset=dataset,
            base_model=base_model,
            output_dir=output_dir,
            dpo_config=config,
        )

        return TrainingJobResult(
            job_id=result.job_id,
            adapter_path=result.adapter_path,
            metrics=result.metrics,
            logs_url=result.logs_url,
        )

    async def train_embedding(
        self,
        dataset: List[Dict],
        base_model: str,
        output_dir: str,
        config: Dict,
    ) -> TrainingJobResult:
        """Run embedding training on remote GPU."""
        logger.info(
            f"Starting remote embedding training ({self.provider}): {len(dataset)} triplets"
        )

        result = await self.runner.run_embedding(
            dataset=dataset,
            base_model=base_model,
            output_dir=output_dir,
            embedding_config=config,
        )

        return TrainingJobResult(
            job_id=result.job_id,
            adapter_path=result.adapter_path,
            metrics=result.metrics,
            logs_url=result.logs_url,
        )
