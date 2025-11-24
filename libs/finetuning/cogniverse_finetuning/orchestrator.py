"""
End-to-end fine-tuning orchestration.

Combines all components into a high-level API:
- Auto-selection (SFT vs DPO vs embedding)
- Dataset extraction from telemetry
- Synthetic generation with mandatory approval
- Training with LoRA (local or remote backend)
- Adapter registration
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Literal, Optional

from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

from cogniverse_finetuning.dataset.embedding_extractor import TripletExtractor
from cogniverse_finetuning.dataset.formatters import InstructionFormatter
from cogniverse_finetuning.dataset.method_selector import TrainingMethodSelector
from cogniverse_finetuning.dataset.preference_extractor import PreferencePairExtractor
from cogniverse_finetuning.dataset.trace_converter import TraceToInstructionConverter
from cogniverse_finetuning.training.backend import (
    LocalTrainingBackend,
    RemoteTrainingBackend,
    TrainingBackend,
    TrainingJobConfig,
)

logger = logging.getLogger(__name__)


def validate_sft_dataset(dataset: List[Dict]) -> None:
    """
    Validate SFT dataset has required fields.

    Args:
        dataset: List of formatted SFT examples

    Raises:
        ValueError: If dataset is empty or missing required fields
    """
    if len(dataset) == 0:
        raise ValueError(
            "Cannot train with empty dataset. No training examples available after formatting."
        )

    required_fields = ["text"]
    for idx, item in enumerate(dataset):
        missing = [f for f in required_fields if f not in item]
        if missing:
            raise ValueError(
                f"Invalid SFT dataset at index {idx}: missing required fields {missing}. "
                f"Expected fields: {required_fields}, got: {list(item.keys())}"
            )


def validate_dpo_dataset(dataset: List[Dict]) -> None:
    """
    Validate DPO dataset has required fields.

    Args:
        dataset: List of formatted DPO preference pairs

    Raises:
        ValueError: If dataset is empty or missing required fields
    """
    if len(dataset) == 0:
        raise ValueError(
            "Cannot train with empty dataset. No preference pairs available after formatting."
        )

    required_fields = ["prompt", "chosen", "rejected"]
    for idx, item in enumerate(dataset):
        missing = [f for f in required_fields if f not in item]
        if missing:
            raise ValueError(
                f"Invalid DPO dataset at index {idx}: missing required fields {missing}. "
                f"Expected fields: {required_fields}, got: {list(item.keys())}"
            )


def validate_embedding_dataset(dataset: List[Dict]) -> None:
    """
    Validate embedding dataset has required fields.

    Args:
        dataset: List of formatted triplets

    Raises:
        ValueError: If dataset is empty or missing required fields
    """
    if len(dataset) == 0:
        raise ValueError(
            "Cannot train with empty dataset. No triplets available after formatting."
        )

    required_fields = ["anchor", "positive", "negative"]
    for idx, item in enumerate(dataset):
        missing = [f for f in required_fields if f not in item]
        if missing:
            raise ValueError(
                f"Invalid embedding dataset at index {idx}: missing required fields {missing}. "
                f"Expected fields: {required_fields}, got: {list(item.keys())}"
            )


@dataclass
class OrchestrationConfig:
    """Configuration for fine-tuning orchestration."""

    # Tenant and project
    tenant_id: str
    project: str

    # Model type
    model_type: Literal["llm", "embedding"]

    # Agent type (for LLM)
    agent_type: Optional[Literal["routing", "profile_selection", "entity_extraction"]] = None

    # Modality (for embedding)
    modality: Optional[Literal["video", "image", "text"]] = None

    # Base model
    base_model: str = "HuggingFaceTB/SmolLM-135M"

    # Auto-selection thresholds
    min_sft_examples: int = 50
    min_dpo_pairs: int = 20
    min_triplets: int = 100  # For embeddings

    # Training config
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    use_lora: bool = True
    backend: Literal["local", "remote"] = "local"
    backend_provider: str = "modal"  # Used when backend="remote" (modal, sagemaker, azure_ml, etc.)

    # Remote backend config (GPU, timeouts, etc.)
    gpu: str = "A10G"
    gpu_count: int = 1
    cpu: int = 4
    memory: int = 16384
    timeout: int = 3600

    # Synthetic generation
    generate_synthetic: bool = True

    # Output
    output_dir: str = "outputs/adapters"


@dataclass
class OrchestrationResult:
    """Result from orchestration."""

    model_type: Literal["llm", "embedding"]
    training_method: Literal["sft", "dpo", "embedding"]
    adapter_path: str
    metrics: Dict
    base_model: str
    lora_config: Dict
    used_synthetic: bool
    synthetic_approval_count: Optional[int] = None


class FinetuningOrchestrator:
    """
    End-to-end fine-tuning orchestration.

    Workflow:
    1. Analyze available data from telemetry
    2. Auto-select training method (SFT/DPO/embedding)
    3. Generate synthetic if needed (with mandatory approval)
    4. Extract and format dataset
    5. Train with LoRA
    6. Return adapter and metrics
    """

    def __init__(
        self,
        telemetry_provider: TelemetryProvider,
        synthetic_service: Optional[any] = None,
        approval_orchestrator: Optional[any] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            telemetry_provider: TelemetryProvider for querying spans/annotations
            synthetic_service: Optional SyntheticDataService for data generation
            approval_orchestrator: Optional ApprovalOrchestrator for human approval
        """
        self.provider = telemetry_provider
        self.synthetic_service = synthetic_service
        self.approval_orchestrator = approval_orchestrator

    def _create_backend(self, config: OrchestrationConfig) -> TrainingBackend:
        """Create training backend based on config."""
        training_config = TrainingJobConfig(
            gpu=config.gpu,
            gpu_count=config.gpu_count,
            cpu=config.cpu,
            memory=config.memory,
            timeout=config.timeout,
        )

        if config.backend == "local":
            return LocalTrainingBackend(training_config)
        elif config.backend == "remote":
            return RemoteTrainingBackend(training_config, provider=config.backend_provider)
        else:
            raise ValueError(f"Unknown backend: {config.backend}")

    async def run(self, config: OrchestrationConfig) -> OrchestrationResult:
        """
        Run end-to-end fine-tuning pipeline.

        Args:
            config: OrchestrationConfig

        Returns:
            OrchestrationResult with adapter path and metrics

        Example:
            >>> config = OrchestrationConfig(
            ...     tenant_id="tenant1",
            ...     project="cogniverse-tenant1",
            ...     model_type="llm",
            ...     agent_type="routing",
            ...     base_model="HuggingFaceTB/SmolLM-135M",
            ...     generate_synthetic=True
            ... )
            >>> result = await orchestrator.run(config)
            >>> print(f"Adapter saved to: {result.adapter_path}")
        """
        logger.info(
            f"Starting fine-tuning orchestration: type={config.model_type}, "
            f"agent={config.agent_type}, modality={config.modality}"
        )

        if config.model_type == "llm":
            return await self._run_llm_finetuning(config)
        elif config.model_type == "embedding":
            return await self._run_embedding_finetuning(config)
        else:
            raise ValueError(f"Unknown model_type: {config.model_type}")

    async def _run_llm_finetuning(
        self, config: OrchestrationConfig
    ) -> OrchestrationResult:
        """Run LLM fine-tuning (SFT or DPO)."""
        if not config.agent_type:
            raise ValueError("agent_type required for LLM fine-tuning")

        # 1. Analyze data and select method
        logger.info("Step 1: Analyzing available data...")
        selector = TrainingMethodSelector(
            synthetic_service=self.synthetic_service,
            approval_orchestrator=self.approval_orchestrator,
        )

        analysis, approved_batch = await selector.analyze_and_prepare(
            provider=self.provider,
            project=config.project,
            agent_type=config.agent_type,
            min_sft_examples=config.min_sft_examples,
            min_dpo_pairs=config.min_dpo_pairs,
            generate_synthetic=config.generate_synthetic,
        )

        logger.info(
            f"Data analysis: method={analysis.recommended_method}, "
            f"approved={analysis.approved_count}, "
            f"pairs={analysis.preference_pairs}, "
            f"synthetic={analysis.needs_synthetic}"
        )

        # 2. Extract and format dataset based on method
        logger.info("Step 2: Extracting and formatting dataset...")
        backend = self._create_backend(config)

        if analysis.recommended_method == "dpo":
            # DPO: Extract preference pairs
            extractor = PreferencePairExtractor(self.provider)
            dataset_obj = await extractor.extract(config.project, config.agent_type)
            pairs = dataset_obj.pairs

            logger.info(f"Extracted {len(pairs)} preference pairs for DPO")

            # Format for training
            formatted_dataset = InstructionFormatter.format_dpo(pairs)

            # Validate dataset
            validate_dpo_dataset(formatted_dataset)
            logger.info(f"Dataset validation passed: {len(formatted_dataset)} pairs")

            # Train with backend (add timestamp to avoid conflicts)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{config.output_dir}/dpo_{config.agent_type}_{timestamp}"
            training_config = {
                "use_lora": config.use_lora,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
            }

            result = await backend.train_dpo(
                dataset=formatted_dataset,
                base_model=config.base_model,
                output_dir=output_dir,
                config=training_config,
            )

            return OrchestrationResult(
                model_type="llm",
                training_method="dpo",
                adapter_path=result.adapter_path,
                metrics=result.metrics,
                base_model=config.base_model,
                lora_config={"use_lora": config.use_lora},
                used_synthetic=analysis.needs_synthetic,
                synthetic_approval_count=approved_batch.approved_count
                if approved_batch
                else None,
            )

        elif analysis.recommended_method == "sft":
            # SFT: Extract instruction examples
            converter = TraceToInstructionConverter(self.provider)
            dataset_obj = await converter.convert(config.project, config.agent_type)
            examples = dataset_obj.examples

            logger.info(f"Extracted {len(examples)} examples for SFT")

            # Format for training (Alpaca format with "text" field for TRL)
            formatted_dataset = InstructionFormatter.format_alpaca_text(examples)

            # Validate dataset
            validate_sft_dataset(formatted_dataset)
            logger.info(f"Dataset validation passed: {len(formatted_dataset)} examples")

            # Train with backend (add timestamp to avoid conflicts)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{config.output_dir}/sft_{config.agent_type}_{timestamp}"
            training_config = {
                "use_lora": config.use_lora,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "dataset_text_field": "text",  # Required for TRL SFTTrainer
            }

            result = await backend.train_sft(
                dataset=formatted_dataset,
                base_model=config.base_model,
                output_dir=output_dir,
                config=training_config,
            )

            return OrchestrationResult(
                model_type="llm",
                training_method="sft",
                adapter_path=result.adapter_path,
                metrics=result.metrics,
                base_model=config.base_model,
                lora_config={"use_lora": config.use_lora},
                used_synthetic=analysis.needs_synthetic,
                synthetic_approval_count=approved_batch.approved_count
                if approved_batch
                else None,
            )

        else:
            raise ValueError(
                f"Insufficient data and synthetic generation failed. "
                f"Analysis: {analysis}"
            )

    async def _run_embedding_finetuning(
        self, config: OrchestrationConfig
    ) -> OrchestrationResult:
        """Run embedding fine-tuning (contrastive learning)."""
        if not config.modality:
            raise ValueError("modality required for embedding fine-tuning")

        # 1. Extract triplets from search logs
        logger.info("Step 1: Extracting triplets from search logs...")
        extractor = TripletExtractor(self.provider)
        triplets = await extractor.extract(
            project=config.project,
            modality=config.modality,
            strategy="top_k",
            min_triplets=config.min_triplets,
        )

        logger.info(f"Extracted {len(triplets)} triplets")

        if len(triplets) < config.min_triplets:
            logger.warning(
                f"Insufficient triplets: {len(triplets)} < {config.min_triplets}. "
                "Proceeding with available data, but results may be suboptimal."
            )

        # 2. Format triplets for training (convert to dicts with anchor/positive/negative)
        formatted_dataset = [
            {
                "anchor": t.anchor,
                "positive": t.positive,
                "negative": t.negative,
            }
            for t in triplets
        ]

        # Validate dataset
        validate_embedding_dataset(formatted_dataset)
        logger.info(f"Dataset validation passed: {len(formatted_dataset)} triplets")

        # 3. Train with backend
        logger.info("Step 3: Training embedding model...")
        backend = self._create_backend(config)

        # Add timestamp to avoid conflicts
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{config.output_dir}/embedding_{config.modality}_{timestamp}"
        training_config = {
            "use_lora": config.use_lora,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
        }

        result = await backend.train_embedding(
            dataset=formatted_dataset,
            base_model=config.base_model,
            output_dir=output_dir,
            config=training_config,
        )

        return OrchestrationResult(
            model_type="embedding",
            training_method="embedding",
            adapter_path=result.adapter_path,
            metrics=result.metrics,
            base_model=config.base_model,
            lora_config={"use_lora": config.use_lora},
            used_synthetic=False,  # Embedding doesn't use synthetic yet
        )


# High-level convenience function
async def finetune(
    telemetry_provider: TelemetryProvider,
    tenant_id: str,
    project: str,
    model_type: Literal["llm", "embedding"],
    agent_type: Optional[str] = None,
    modality: Optional[str] = None,
    base_model: str = "HuggingFaceTB/SmolLM-135M",
    backend: Literal["local", "remote"] = "local",
    backend_provider: str = "modal",  # Used when backend="remote"
    # Training hyperparameters
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    # Remote backend configuration (GPU, timeouts, etc.)
    gpu: str = "A10G",  # T4, A10G, A100-40GB, A100-80GB, H100
    gpu_count: int = 1,
    cpu: int = 4,
    memory: int = 16384,
    timeout: int = 3600,  # seconds
    # Synthetic and approval
    synthetic_service: Optional[any] = None,
    approval_orchestrator: Optional[any] = None,
    # Output
    output_dir: str = "outputs/adapters",
) -> OrchestrationResult:
    """
    High-level fine-tuning function.

    Extracts dataset from telemetry and trains using specified backend (local or remote).

    Flow:
    1. Extract dataset locally from TelemetryProvider
    2. Auto-select training method (SFT vs DPO vs embedding)
    3. Generate synthetic data if needed (with approval)
    4. Train using backend (local or remote GPU)
    5. Return adapter and metrics

    Args:
        telemetry_provider: TelemetryProvider instance
        tenant_id: Tenant ID
        project: Project name
        model_type: "llm" or "embedding"
        agent_type: Agent type for LLM (routing, profile_selection, entity_extraction)
        modality: Modality for embedding (video, image, text)
        base_model: Base model name
        backend: "local" (local GPU/CPU) or "remote" (cloud GPU)
        backend_provider: Remote provider when backend="remote" (modal, sagemaker, azure_ml, etc.)
        gpu: GPU type for remote backend (T4, A10G, A100-40GB, etc.)
        synthetic_service: Optional SyntheticDataService
        approval_orchestrator: Optional ApprovalOrchestrator

    Returns:
        OrchestrationResult

    Example (Local Training):
        >>> provider = manager.get_provider("tenant1", "cogniverse-tenant1")
        >>> result = await finetune(
        ...     telemetry_provider=provider,
        ...     tenant_id="tenant1",
        ...     project="cogniverse-tenant1",
        ...     model_type="llm",
        ...     agent_type="routing",
        ...     backend="local"  # Train locally
        ... )

    Example (Remote GPU Training):
        >>> result = await finetune(
        ...     telemetry_provider=provider,
        ...     tenant_id="tenant1",
        ...     project="cogniverse-tenant1",
        ...     model_type="llm",
        ...     agent_type="routing",
        ...     backend="remote",  # Train on remote GPU
        ...     backend_provider="modal",  # Use Modal
        ...     gpu="A100-40GB",  # Choose GPU type
        ...     epochs=5,
        ...     batch_size=8,
        ...     learning_rate=1e-4
        ... )
    """
    # Create orchestration config with all parameters
    config = OrchestrationConfig(
        tenant_id=tenant_id,
        project=project,
        model_type=model_type,
        agent_type=agent_type,
        modality=modality,
        base_model=base_model,
        backend=backend,
        backend_provider=backend_provider,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gpu=gpu,
        gpu_count=gpu_count,
        cpu=cpu,
        memory=memory,
        timeout=timeout,
        output_dir=output_dir,
    )

    # Create orchestrator
    orchestrator = FinetuningOrchestrator(
        telemetry_provider=telemetry_provider,
        synthetic_service=synthetic_service,
        approval_orchestrator=approval_orchestrator,
    )

    return await orchestrator.run(config)
