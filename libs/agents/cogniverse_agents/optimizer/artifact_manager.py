"""
Artifact Manager — save/load DSPy optimization artifacts via telemetry stores.

Provides tenant-isolated, versioned artifact persistence using DatasetStore
(for prompts and demonstrations) and ExperimentStore (for optimization metrics).
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Manage DSPy optimization artifacts through telemetry stores.

    Uses DatasetStore for prompts and demonstrations, ExperimentStore for metrics.
    Dataset naming: ``dspy-{kind}-{tenant_id}-{agent_type}`` where *kind* is
    ``prompts`` or ``demos``.
    """

    def __init__(self, telemetry_provider: TelemetryProvider, tenant_id: str) -> None:
        if not tenant_id:
            raise ValueError("tenant_id is required")
        self._provider = telemetry_provider
        self._tenant_id = tenant_id

    def _prompt_dataset_name(self, agent_type: str) -> str:
        return f"dspy-prompts-{self._tenant_id}-{agent_type}"

    def _demo_dataset_name(self, agent_type: str) -> str:
        return f"dspy-demos-{self._tenant_id}-{agent_type}"

    def _experiment_name(self) -> str:
        return f"dspy-optimization-{self._tenant_id}"

    async def save_prompts(self, agent_type: str, prompts: Dict[str, str]) -> str:
        """Persist optimized prompts as a dataset.

        Args:
            agent_type: Agent identifier (e.g. ``agent_routing``, ``query_analysis``).
            prompts: Mapping of prompt key to prompt text.

        Returns:
            Dataset identifier assigned by the store.
        """
        rows = [{"name": k, "value": v} for k, v in prompts.items()]
        df = pd.DataFrame(rows)
        dataset_name = self._prompt_dataset_name(agent_type)
        dataset_id = await self._provider.datasets.create_dataset(
            name=dataset_name,
            data=df,
            metadata={
                "artifact_type": "dspy_prompts",
                "agent_type": agent_type,
                "tenant_id": self._tenant_id,
                "created_at": datetime.now().isoformat(),
            },
        )
        logger.info(
            "Saved %d prompts for %s/%s → dataset %s",
            len(prompts),
            self._tenant_id,
            agent_type,
            dataset_id,
        )
        return dataset_id

    async def load_prompts(self, agent_type: str) -> Optional[Dict[str, str]]:
        """Load optimized prompts from dataset.

        Returns:
            ``{key: prompt_text}`` or ``None`` if no dataset exists.

        Raises:
            Exception: Propagates store errors (connection, deserialization).
                ``KeyError`` raised by the store when the dataset does not exist
                is treated as "no artifacts" and returns ``None`` instead.
        """
        dataset_name = self._prompt_dataset_name(agent_type)
        try:
            df = await self._provider.datasets.get_dataset(name=dataset_name)
        except KeyError:
            logger.debug(
                "No prompt dataset found for %s/%s",
                self._tenant_id,
                agent_type,
            )
            return None

        if df is None or df.empty:
            return None

        prompts = dict(zip(df["name"], df["value"]))
        logger.info(
            "Loaded %d prompts for %s/%s",
            len(prompts),
            self._tenant_id,
            agent_type,
        )
        return prompts

    async def save_demonstrations(
        self, agent_type: str, demos: List[Dict[str, Any]]
    ) -> str:
        """Persist few-shot demonstrations as a dataset.

        Each demo is a dict with at least ``input`` and ``output`` keys,
        plus optional ``metadata``.

        Returns:
            Dataset identifier assigned by the store.
        """
        df = pd.DataFrame(demos)
        dataset_name = self._demo_dataset_name(agent_type)
        dataset_id = await self._provider.datasets.create_dataset(
            name=dataset_name,
            data=df,
            metadata={
                "artifact_type": "dspy_demos",
                "agent_type": agent_type,
                "tenant_id": self._tenant_id,
                "created_at": datetime.now().isoformat(),
            },
        )
        logger.info(
            "Saved %d demonstrations for %s/%s → dataset %s",
            len(demos),
            self._tenant_id,
            agent_type,
            dataset_id,
        )
        return dataset_id

    async def load_demonstrations(
        self, agent_type: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Load few-shot demonstrations from dataset.

        Returns:
            List of demo dicts or ``None`` if no dataset exists.
        """
        dataset_name = self._demo_dataset_name(agent_type)
        df = await self._provider.datasets.get_dataset(name=dataset_name)

        if df is None or df.empty:
            return None

        demos = df.to_dict(orient="records")
        logger.info(
            "Loaded %d demonstrations for %s/%s",
            len(demos),
            self._tenant_id,
            agent_type,
        )
        return demos

    async def log_optimization_run(
        self, agent_type: str, metrics: Dict[str, Any]
    ) -> str:
        """Log an optimization run to the experiment store.

        Returns:
            Run identifier.
        """
        experiment_name = self._experiment_name()
        experiment_id = await self._provider.experiments.create_experiment(
            name=experiment_name,
            metadata={"tenant_id": self._tenant_id},
        )
        run_id = await self._provider.experiments.log_run(
            experiment_id=experiment_id,
            inputs={"agent_type": agent_type, "tenant_id": self._tenant_id},
            outputs=metrics,
            metadata={"timestamp": datetime.now().isoformat()},
        )
        logger.info(
            "Logged optimization run for %s/%s → experiment %s, run %s",
            self._tenant_id,
            agent_type,
            experiment_id,
            run_id,
        )
        return run_id
