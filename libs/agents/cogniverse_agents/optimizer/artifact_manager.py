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
                "input_keys": ["name"],
                "output_keys": ["value"],
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
                ``ValueError``/``KeyError`` raised by the store when the dataset
                does not exist is treated as "no artifacts" and returns ``None``.
        """
        dataset_name = self._prompt_dataset_name(agent_type)
        try:
            df = await self._provider.datasets.get_dataset(name=dataset_name)
        except (KeyError, ValueError):
            logger.debug(
                "No prompt dataset found for %s/%s",
                self._tenant_id,
                agent_type,
            )
            return None

        if df is None or df.empty:
            return None

        prompts = self._extract_prompts_from_dataframe(df)
        logger.info(
            "Loaded %d prompts for %s/%s",
            len(prompts),
            self._tenant_id,
            agent_type,
        )
        return prompts

    @staticmethod
    def _extract_prompts_from_dataframe(df: pd.DataFrame) -> Dict[str, str]:
        """Extract name→value prompt dict from a Phoenix dataset DataFrame.

        Phoenix may return columns in different layouts depending on how
        ``input_keys``/``output_keys`` were specified at upload time:

        1. Flat columns ``name``, ``value`` (when keys were specified).
        2. A single ``input`` column containing dicts with ``name`` key,
           and an ``output`` column containing dicts with ``value`` key.
        3. A single ``input`` column containing dicts with both keys
           (when no keys were specified at all).
        """
        if "name" in df.columns and "value" in df.columns:
            return dict(zip(df["name"], df["value"]))

        prompts: Dict[str, str] = {}
        for _, row in df.iterrows():
            inp = row.get("input", {})
            out = row.get("output", {})
            if isinstance(inp, dict) and isinstance(out, dict):
                name = inp.get("name", "")
                value = out.get("value", "")
                if name:
                    prompts[name] = value
            elif isinstance(inp, dict):
                name = inp.get("name", "")
                value = inp.get("value", "")
                if name:
                    prompts[name] = value
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
                "input_keys": ["input"],
                "output_keys": ["output"],
                "metadata_keys": ["metadata"] if "metadata" in df.columns else [],
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
        try:
            df = await self._provider.datasets.get_dataset(name=dataset_name)
        except (KeyError, ValueError):
            logger.debug(
                "No demo dataset found for %s/%s",
                self._tenant_id,
                agent_type,
            )
            return None

        if df is None or df.empty:
            return None

        demos = self._extract_demos_from_dataframe(df)
        logger.info(
            "Loaded %d demonstrations for %s/%s",
            len(demos),
            self._tenant_id,
            agent_type,
        )
        return demos

    @staticmethod
    def _extract_demos_from_dataframe(
        df: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """Extract demo dicts from a Phoenix dataset DataFrame.

        Phoenix may return flat columns (``input``, ``output``, ``metadata``)
        when ``input_keys``/``output_keys``/``metadata_keys`` were set, or nested
        dicts in ``input``/``output`` columns otherwise.  This method normalises
        both layouts to ``[{"input": ..., "output": ..., "metadata": ...}, ...]``.
        """
        demos: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            inp = row.get("input", "")
            out = row.get("output", "")
            meta = row.get("metadata", "")

            # Phoenix wraps each role's columns into a dict when there's
            # exactly one column per role — unwrap the single value.
            if isinstance(inp, dict) and len(inp) == 1 and "input" in inp:
                inp = inp["input"]
            if isinstance(out, dict) and len(out) == 1 and "output" in out:
                out = out["output"]
            if isinstance(meta, dict) and len(meta) == 1 and "metadata" in meta:
                meta = meta["metadata"]

            demos.append({"input": inp, "output": out, "metadata": meta})
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
