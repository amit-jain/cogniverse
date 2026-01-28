"""
Modal GPU training integration.

Runs fine-tuning jobs on Modal cloud GPUs with automatic
dataset upload/download and environment management.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModalJobConfig:
    """Configuration for Modal training job."""

    # GPU configuration
    gpu: str = "A10G"  # Options: T4, A10G, A100-40GB, A100-80GB, H100
    gpu_count: int = 1
    cpu: int = 4
    memory: int = 16384  # MB

    # Timeouts
    timeout: int = 3600  # 1 hour default


@dataclass
class ModalJobResult:
    """Result from Modal training job."""

    job_id: str
    adapter_path: str
    metrics: Dict
    logs_url: Optional[str] = None


class ModalTrainingRunner:
    """
    Run fine-tuning jobs on Modal cloud GPUs.

    Workflow:
    1. Upload dataset to S3/HF Hub
    2. Create Modal function with GPU requirements
    3. Launch training job
    4. Wait for completion
    5. Download adapter
    6. Return results
    """

    def __init__(self, config: ModalJobConfig):
        """
        Initialize Modal runner.

        Args:
            config: ModalJobConfig instance
        """
        self.config = config

    async def run_sft(
        self,
        dataset: List[Dict],
        base_model: str,
        output_dir: str,
        sft_config: Dict,
    ) -> ModalJobResult:
        """
        Run SFT training on Modal GPU.

        Args:
            dataset: Training dataset (formatted for SFT) - passed directly to Modal
            base_model: Base model name
            output_dir: Output directory for adapter
            sft_config: SFT configuration dict

        Returns:
            ModalJobResult with local adapter path

        Flow:
        1. Pass dataset directly to Modal function (no S3!)
        2. Modal trains and returns adapter as bytes
        3. Extract adapter bytes to local directory
        """
        logger.info(f"Launching SFT training on Modal ({self.config.gpu})...")
        logger.info(f"Dataset: {len(dataset)} examples")

        # 1. Get Modal function
        modal_fn = self._create_sft_function()

        # 2. Call Modal directly with dataset (no upload needed!)
        logger.info("Calling Modal function...")
        result = modal_fn.remote(
            dataset=dataset,  # Passed directly!
            base_model=base_model,
            config=sft_config,
        )

        logger.info("Modal training complete")

        # 3. Extract adapter bytes to local directory
        import tarfile
        from io import BytesIO

        adapter_bytes = result["adapter_bytes"]
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract tar.gz
        buffer = BytesIO(adapter_bytes)
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            tar.extractall(output_path)

        adapter_path = str(output_path / "adapter")
        logger.info(f"Adapter extracted to {adapter_path}")

        return ModalJobResult(
            job_id="N/A",  # Direct call, no job ID
            adapter_path=adapter_path,
            metrics=result["metrics"],
            logs_url=None,
        )

    async def run_dpo(
        self,
        dataset: List[Dict],
        base_model: str,
        output_dir: str,
        dpo_config: Dict,
    ) -> ModalJobResult:
        """
        Run DPO training on Modal GPU.

        Args:
            dataset: Preference pairs dataset (formatted for DPO) - passed directly to Modal
            base_model: Base model name
            output_dir: Output directory for adapter
            dpo_config: DPO configuration dict

        Returns:
            ModalJobResult with local adapter path

        Flow:
        1. Pass dataset directly to Modal function (no S3!)
        2. Modal trains and returns adapter as bytes
        3. Extract adapter bytes to local directory
        """
        logger.info(f"Launching DPO training on Modal ({self.config.gpu})...")
        logger.info(f"Dataset: {len(dataset)} preference pairs")

        # 1. Get Modal function
        modal_fn = self._create_dpo_function()

        # 2. Call Modal directly with dataset (no upload needed!)
        logger.info("Calling Modal function...")
        result = modal_fn.remote(
            dataset=dataset,  # Passed directly!
            base_model=base_model,
            config=dpo_config,
        )

        logger.info("Modal training complete")

        # 3. Extract adapter bytes to local directory
        import tarfile
        from io import BytesIO

        adapter_bytes = result["adapter_bytes"]
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract tar.gz
        buffer = BytesIO(adapter_bytes)
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            tar.extractall(output_path)

        adapter_path = str(output_path / "adapter")
        logger.info(f"Adapter extracted to {adapter_path}")

        return ModalJobResult(
            job_id="N/A",  # Direct call, no job ID
            adapter_path=adapter_path,
            metrics=result["metrics"],
            logs_url=None,
        )

    async def run_embedding(
        self,
        dataset: List[Dict],
        base_model: str,
        output_dir: str,
        embedding_config: Dict,
    ) -> ModalJobResult:
        """
        Run embedding training on Modal GPU.

        Args:
            dataset: Triplets dataset (anchor/positive/negative) - passed directly to Modal
            base_model: Base embedding model name
            output_dir: Output directory for adapter
            embedding_config: Embedding configuration dict

        Returns:
            ModalJobResult with local adapter path

        Flow:
        1. Pass dataset directly to Modal function (no S3!)
        2. Modal trains and returns adapter as bytes
        3. Extract adapter bytes to local directory
        """
        logger.info(f"Launching embedding training on Modal ({self.config.gpu})...")
        logger.info(f"Dataset: {len(dataset)} triplets")

        # 1. Get Modal function
        modal_fn = self._create_embedding_function()

        # 2. Call Modal directly with dataset (no upload needed!)
        logger.info("Calling Modal function...")
        result = modal_fn.remote(
            dataset=dataset,  # Passed directly!
            base_model=base_model,
            config=embedding_config,
        )

        logger.info("Modal training complete")

        # 3. Extract adapter bytes to local directory
        import tarfile
        from io import BytesIO

        adapter_bytes = result["adapter_bytes"]
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract tar.gz
        buffer = BytesIO(adapter_bytes)
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            tar.extractall(output_path)

        adapter_path = str(output_path / "adapter")
        logger.info(f"Adapter extracted to {adapter_path}")

        return ModalJobResult(
            job_id="N/A",  # Direct call, no job ID
            adapter_path=adapter_path,
            metrics=result["metrics"],
            logs_url=None,
        )

    def _create_sft_function(self):
        """Get deployed SFT training function from Modal."""
        try:
            from cogniverse_finetuning.training import modal_app

            return modal_app.train_sft_remote
        except ImportError:
            raise ImportError(
                "Modal app not found. Deploy it first with:\n"
                "  modal deploy libs/finetuning/cogniverse_finetuning/training/modal_app.py"
            )

    def _create_dpo_function(self):
        """Get deployed DPO training function from Modal."""
        try:
            from cogniverse_finetuning.training import modal_app

            return modal_app.train_dpo_remote
        except ImportError:
            raise ImportError(
                "Modal app not found. Deploy it first with:\n"
                "  modal deploy libs/finetuning/cogniverse_finetuning/training/modal_app.py"
            )

    def _create_embedding_function(self):
        """Get deployed embedding training function from Modal."""
        try:
            from cogniverse_finetuning.training import modal_app

            return modal_app.train_embedding_remote
        except ImportError:
            raise ImportError(
                "Modal app not found. Deploy it first with:\n"
                "  modal deploy libs/finetuning/cogniverse_finetuning/training/modal_app.py"
            )
