"""
Artifact Manager — save/load DSPy optimization artifacts via telemetry stores.

Provides tenant-isolated, versioned artifact persistence using DatasetStore
for prompts, demonstrations, and experiment metrics.

Optimization metrics are stored as a typed ``ExperimentMetrics``
row in a dedicated dataset (``dspy-experiments-{tenant}-{agent}``). The
previous ``save_blob`` workaround papered over PhoenixProvider's
``save_experiment`` no-op stub and made metrics impossible to query
historically — every run overwrote the previous one. The dedicated dataset
appends per run so callers can fetch the latest, the full history, or
filter by score/timestamp/baseline.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from cogniverse_agents.optimizer.signature_variants import (
    DEFAULT_VARIANT_ID,
    variant_qualified_agent_key,
)
from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

logger = logging.getLogger(__name__)

_CACHE_MISS = object()


@dataclass(frozen=True)
class ExperimentMetrics:
    """Typed record for a single optimization run.

    Stored as one row in the per-tenant per-agent experiments dataset. Fields
    are intentionally a small fixed set so the dataset stays queryable; ad-hoc
    metrics go under ``extra_metrics`` (serialised to JSON in storage).
    """

    tenant_id: str
    agent_type: str
    run_id: str  # caller-supplied ULID/UUID; used to dedupe + correlate
    timestamp: str  # ISO-8601 UTC
    optimizer: str  # e.g. "BootstrapFewShot", "MIPROv2"
    baseline_score: Optional[float] = None
    candidate_score: Optional[float] = None
    improvement: Optional[float] = None  # candidate - baseline
    promoted: bool = False  # True iff the candidate passed the gate
    train_examples: Optional[int] = None
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        """Flatten to a row suitable for DataFrame storage.

        ``extra_metrics`` is JSON-serialised to keep the dataset schema flat;
        consumers should ``json.loads`` it back when reading.
        """
        d = asdict(self)
        d["extra_metrics"] = json.dumps(d["extra_metrics"], default=str)
        return d

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "ExperimentMetrics":
        # Phoenix's ``to_dataframe()`` round-trip lands columns in one
        # of four shapes depending on how create_dataset was called:
        #   * flat with original names (when metadata_keys covers them all)
        #   * dict under ``input`` / ``output`` / ``metadata`` (Phoenix's
        #     default categorisation when no keys are specified — every
        #     column except input_keys/output_keys lands under ``input``)
        #   * nested dict under ``metadata`` (older Phoenix versions)
        #   * dotted-flat like ``metadata.tenant_id``
        # Normalise all four back to the flat layout the dataclass expects.
        if "tenant_id" not in row:
            promoted = {}
            for nested_key in ("input", "metadata", "output"):
                nested = row.get(nested_key)
                if isinstance(nested, dict):
                    promoted.update(nested)
            if promoted:
                row = {
                    **promoted,
                    **{
                        k: v
                        for k, v in row.items()
                        if k not in ("input", "metadata", "output")
                    },
                }
            elif any(k.startswith("metadata.") for k in row):
                row = {
                    **{
                        k.split(".", 1)[1]: v
                        for k, v in row.items()
                        if k.startswith("metadata.")
                    },
                    **{k: v for k, v in row.items() if not k.startswith("metadata.")},
                }
            if "tenant_id" not in row:
                raise KeyError(
                    f"ExperimentMetrics.from_row could not locate 'tenant_id' "
                    f"in any of: top-level row, nested input/output/metadata "
                    f"dicts, or 'metadata.<key>' dotted columns. "
                    f"Row columns: {sorted(row.keys())}"
                )

        extras = row.get("extra_metrics") or {}
        if isinstance(extras, str):
            try:
                extras = json.loads(extras)
            except json.JSONDecodeError:
                extras = {}
        return cls(
            tenant_id=row["tenant_id"],
            agent_type=row["agent_type"],
            run_id=row["run_id"],
            timestamp=row["timestamp"],
            optimizer=row["optimizer"],
            baseline_score=_optional_float(row.get("baseline_score")),
            candidate_score=_optional_float(row.get("candidate_score")),
            improvement=_optional_float(row.get("improvement")),
            promoted=_parse_bool(row.get("promoted", False)),
            train_examples=_optional_int(row.get("train_examples")),
            extra_metrics=extras,
        )


def _optional_float(v: Any) -> Optional[float]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_bool(v: Any) -> bool:
    """Round-trip-safe bool parse.

    Phoenix's dataset round-trip can stringify booleans (``True`` → ``"True"``
    and ``False`` → ``"False"``); a naked ``bool("False")`` is truthy. Treat
    common string spellings explicitly.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes"}
    if v is None:
        return False
    try:
        return bool(int(v))
    except (TypeError, ValueError):
        return bool(v)


def _optional_int(v: Any) -> Optional[int]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


class ArtifactManager:
    """Manage DSPy optimization artifacts through telemetry stores.

    Uses DatasetStore for prompts, demonstrations, and experiment-metric rows.
    Dataset naming: ``dspy-{kind}-{tenant_id}-{agent_type}`` where *kind* is
    ``prompts`` or ``demos``.
    """

    def __init__(self, telemetry_provider: TelemetryProvider, tenant_id: str) -> None:
        if not tenant_id:
            raise ValueError("tenant_id is required")
        # Canonicalize so a bare org id (``acme``) and the canonical
        # form (``acme:acme``) reach the SAME dataset name. Dispatchers
        # apply ``require_tenant_id`` at request boundaries which
        # canonicalizes; without matching that here, an artifact saved
        # under ``acme`` cannot be loaded by ``_load_artifact`` after
        # the dispatcher has rewritten the tenant to ``acme:acme``.
        from cogniverse_core.common.tenant_utils import canonical_tenant_id

        self._provider = telemetry_provider
        self._tenant_id = canonical_tenant_id(tenant_id)
        # Short-TTL per-request cache for the hot load_for_request path
        # (artefact state + resolved prompts). State changes only on
        # promote/retire, which invalidate via _save_artefact_state.
        self._request_cache: Dict[str, tuple[float, Any]] = {}

    _REQUEST_CACHE_TTL_SECONDS = 5.0

    def _request_cache_get(self, key: str) -> Any:
        entry = self._request_cache.get(key)
        if entry is None:
            return _CACHE_MISS
        expiry, value = entry
        if time.monotonic() >= expiry:
            self._request_cache.pop(key, None)
            return _CACHE_MISS
        return value

    def _request_cache_put(self, key: str, value: Any) -> None:
        self._request_cache[key] = (
            time.monotonic() + self._REQUEST_CACHE_TTL_SECONDS,
            value,
        )

    def _invalidate_request_cache(self, agent_type: str) -> None:
        for key in [k for k in self._request_cache if agent_type in k]:
            self._request_cache.pop(key, None)

    @staticmethod
    def qualified_agent_key(
        agent_type: str, variant_id: str = DEFAULT_VARIANT_ID
    ) -> str:
        """Public helper: build the per-(agent, variant) dataset key.

        The dataset-name builders treat their ``agent_type`` argument as
        already-qualified, so callers that want per-variant artefacts
        compute this key first and pass it through. The default variant
        maps to the bare agent_type (its canonical, unsuffixed key).
        """
        return variant_qualified_agent_key(agent_type, variant_id)

    def _prompt_dataset_name(self, agent_type: str) -> str:
        return f"dspy-prompts-{self._tenant_id}-{agent_type}"

    def _demo_dataset_name(self, agent_type: str) -> str:
        return f"dspy-demos-{self._tenant_id}-{agent_type}"

    def _experiment_name(self) -> str:
        return f"dspy-optimization-{self._tenant_id}"

    def _experiments_dataset_name(self, agent_type: str) -> str:
        """Dataset that stores typed ExperimentMetrics rows for an agent."""
        return f"dspy-experiments-{self._tenant_id}-{agent_type}"

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

    def _blob_dataset_name(self, kind: str, key: str) -> str:
        return f"dspy-{kind}-{self._tenant_id}-{key}"

    async def save_blob(self, kind: str, key: str, content: str) -> str:
        """Persist an arbitrary string blob as a single-row dataset.

        Intended for serialized models (DSPy JSON, XGBoost JSON),
        checkpoints, embedding caches, and other opaque string payloads.

        Args:
            kind: Category (e.g. ``model``, ``checkpoint``, ``embeddings``).
            key: Identifier within the category.
            content: The string payload to store.

        Returns:
            Dataset identifier assigned by the store.
        """
        df = pd.DataFrame([{"content": content}])
        dataset_name = self._blob_dataset_name(kind, key)
        metadata = {
            "artifact_type": f"blob_{kind}",
            "key": key,
            "tenant_id": self._tenant_id,
            "created_at": datetime.now().isoformat(),
            "input_keys": ["content"],
            "output_keys": [],
        }
        # Blobs are last-write-wins. Delete any existing dataset so the store
        # holds exactly one row — create_dataset on an existing name appends a
        # new version, growing the dataset (and load's full-history download)
        # unboundedly across saves. The pre-read makes the delete recoverable:
        # a create failure after the committed delete would otherwise destroy
        # the previous blob and read back as "never optimized".
        previous = await self.load_blob(kind, key)
        await self._provider.datasets.delete_dataset(dataset_name)
        try:
            dataset_id = await self._provider.datasets.create_dataset(
                name=dataset_name,
                data=df,
                metadata=metadata,
            )
        except Exception:
            if previous is not None:
                try:
                    await self._provider.datasets.create_dataset(
                        name=dataset_name,
                        data=pd.DataFrame([{"content": previous}]),
                        metadata=metadata,
                    )
                    logger.warning(
                        "Blob %s/%s overwrite failed for %s; previous content restored",
                        kind,
                        key,
                        self._tenant_id,
                    )
                except Exception:
                    logger.exception(
                        "Blob %s/%s overwrite failed for %s and the restore "
                        "also failed; previous content lost",
                        kind,
                        key,
                        self._tenant_id,
                    )
            raise
        logger.info(
            "Saved blob %s/%s for %s → dataset %s",
            kind,
            key,
            self._tenant_id,
            dataset_id,
        )
        return dataset_id

    async def load_blob(self, kind: str, key: str) -> Optional[str]:
        """Load a string blob from dataset.

        Returns:
            The stored string or ``None`` if no dataset exists.
        """
        dataset_name = self._blob_dataset_name(kind, key)
        try:
            df = await self._provider.datasets.get_dataset(name=dataset_name)
        except (KeyError, ValueError):
            logger.debug(
                "No blob dataset found for %s/%s/%s",
                self._tenant_id,
                kind,
                key,
            )
            return None

        if df is None or df.empty:
            return None

        # Extract content from the dataset (last row = latest version)
        if "content" in df.columns:
            content = df["content"].iloc[-1]
        elif "input" in df.columns:
            inp = df["input"].iloc[-1]
            content = inp.get("content", inp) if isinstance(inp, dict) else inp
        else:
            logger.warning(
                "Blob dataset %s has unexpected columns: %s",
                dataset_name,
                list(df.columns),
            )
            return None

        logger.info(
            "Loaded blob %s/%s for %s (%d chars)",
            kind,
            key,
            self._tenant_id,
            len(content) if content else 0,
        )
        return content

    def _versioned_dataset_name(self, kind: str, agent_type: str, version: int) -> str:
        return f"dspy-{kind}-{self._tenant_id}-{agent_type}-v{version}"

    async def _get_next_version(self, kind: str, agent_type: str) -> int:
        """Probe versioned dataset names sequentially to find the next version.

        DatasetStore has no list_datasets(), so we probe v1, v2, ... until
        get_dataset raises KeyError/ValueError (not found).
        """
        v = 1
        while True:
            name = self._versioned_dataset_name(kind, agent_type, v)
            try:
                df = await self._provider.datasets.get_dataset(name=name)
                if df is None or df.empty:
                    break
                v += 1
            except (KeyError, ValueError):
                break
        return v

    async def save_prompts_versioned(
        self, agent_type: str, prompts: Dict[str, str]
    ) -> tuple[str, int]:
        """Save prompts with auto-incrementing version.

        Returns:
            Tuple of (dataset_id, version_number).
        """
        version = await self._get_next_version("prompts", agent_type)
        rows = [{"name": k, "value": v} for k, v in prompts.items()]
        df = pd.DataFrame(rows)
        dataset_name = self._versioned_dataset_name("prompts", agent_type, version)
        dataset_id = await self._provider.datasets.create_dataset(
            name=dataset_name,
            data=df,
            metadata={
                "artifact_type": "dspy_prompts",
                "agent_type": agent_type,
                "tenant_id": self._tenant_id,
                "version": version,
                "created_at": datetime.now().isoformat(),
                "input_keys": ["name"],
                "output_keys": ["value"],
            },
        )
        logger.info(
            "Saved prompts v%d for %s/%s → dataset %s",
            version,
            self._tenant_id,
            agent_type,
            dataset_id,
        )
        return dataset_id, version

    async def save_demonstrations_versioned(
        self, agent_type: str, demos: List[Dict[str, Any]]
    ) -> tuple[str, int]:
        """Save demonstrations with auto-incrementing version.

        Returns:
            Tuple of (dataset_id, version_number).
        """
        version = await self._get_next_version("demos", agent_type)
        df = pd.DataFrame(demos)
        dataset_name = self._versioned_dataset_name("demos", agent_type, version)
        dataset_id = await self._provider.datasets.create_dataset(
            name=dataset_name,
            data=df,
            metadata={
                "artifact_type": "dspy_demos",
                "agent_type": agent_type,
                "tenant_id": self._tenant_id,
                "version": version,
                "created_at": datetime.now().isoformat(),
                "input_keys": ["input"],
                "output_keys": ["output"],
                "metadata_keys": ["metadata"] if "metadata" in df.columns else [],
            },
        )
        logger.info(
            "Saved demonstrations v%d for %s/%s → dataset %s",
            version,
            self._tenant_id,
            agent_type,
            dataset_id,
        )
        return dataset_id, version

    async def list_versions(self, kind: str, agent_type: str) -> List[Dict[str, Any]]:
        """List all versions of a dataset kind for an agent type.

        Probes v1, v2, ... sequentially until no dataset is found.

        Args:
            kind: ``prompts`` or ``demos``.
            agent_type: Agent identifier.

        Returns:
            List of dicts with ``version`` and ``name`` keys,
            sorted by version ascending.
        """
        versions = []
        v = 1
        while True:
            name = self._versioned_dataset_name(kind, agent_type, v)
            try:
                df = await self._provider.datasets.get_dataset(name=name)
                if df is None or df.empty:
                    break
                versions.append({"version": v, "name": name})
                v += 1
            except (KeyError, ValueError):
                break
        return versions

    async def get_version_lineage(
        self, kind: str, agent_type: str
    ) -> List[Dict[str, Any]]:
        """Get version lineage with metadata for each version.

        Returns:
            List of version info dicts including metadata from each dataset.
        """
        versions = await self.list_versions(kind, agent_type)
        lineage = []
        for v_info in versions:
            entry = {**v_info}
            try:
                df = await self._provider.datasets.get_dataset(name=v_info["name"])
                entry["row_count"] = len(df) if df is not None else 0
            except (KeyError, ValueError):
                entry["row_count"] = 0
            lineage.append(entry)
        return lineage

    # --- canary state machine ---------------------------------------------

    def _state_blob_key(self, agent_type: str) -> str:
        return f"artefact_state_{agent_type}"

    async def get_artefact_state(self, agent_type: str) -> Dict[str, Any]:
        """Return the current state machine snapshot for an agent.

        Schema:
            {
              "active":   {"version": int, "promoted_at": ISO} | None,
              "canary":   {"version": int, "promoted_at": ISO,
                           "traffic_pct": int} | None,
              "retired":  [{"version": int, "retired_at": ISO,
                            "reason": str}, ...]
            }

        Backed by ``save_blob`` for atomicity (single JSON document).
        """
        raw = await self.load_blob("config", self._state_blob_key(agent_type))
        if not raw:
            return {"active": None, "canary": None, "retired": []}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "Artefact state blob for %s/%s was not JSON; resetting",
                self._tenant_id,
                agent_type,
            )
            return {"active": None, "canary": None, "retired": []}

    async def _save_artefact_state(
        self, agent_type: str, state: Dict[str, Any]
    ) -> None:
        await self.save_blob(
            "config",
            self._state_blob_key(agent_type),
            json.dumps(state, default=str),
        )
        self._invalidate_request_cache(agent_type)

    async def promote_to_canary(
        self,
        agent_type: str,
        version: int,
        *,
        traffic_pct: int = 10,
    ) -> Dict[str, Any]:
        """Mark a versioned artefact as the canary at ``traffic_pct`` % traffic.

        Replaces any existing canary (the previous canary moves to retired
        with a synthetic ``retired_at``). The active artefact is untouched.
        """
        if not 1 <= traffic_pct <= 100:
            raise ValueError(f"traffic_pct must be in [1, 100]; got {traffic_pct}")
        state = await self.get_artefact_state(agent_type)
        if state.get("canary"):
            state.setdefault("retired", []).append(
                {
                    "version": state["canary"]["version"],
                    "retired_at": datetime.now(timezone.utc).isoformat(),
                    "reason": "superseded_by_new_canary",
                }
            )
        state["canary"] = {
            "version": version,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "traffic_pct": traffic_pct,
        }
        await self._save_artefact_state(agent_type, state)
        return state

    async def promote_canary_to_active(self, agent_type: str) -> Dict[str, Any]:
        """Promote the current canary to active. Previous active retired."""
        state = await self.get_artefact_state(agent_type)
        if not state.get("canary"):
            raise ValueError(
                f"no canary set for {self._tenant_id}/{agent_type}; "
                "call promote_to_canary first"
            )

        if state.get("active"):
            state.setdefault("retired", []).append(
                {
                    "version": state["active"]["version"],
                    "retired_at": datetime.now(timezone.utc).isoformat(),
                    "reason": "superseded_by_canary_promotion",
                }
            )

        canary_version = state["canary"]["version"]
        state["active"] = {
            "version": canary_version,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }
        state["canary"] = None
        # Active dataset content needs to land at the un-versioned name
        # agents read at __init__. Copy from the versioned snapshot.
        await self._restore_active_from_version(agent_type, canary_version)
        await self._save_artefact_state(agent_type, state)
        return state

    async def retire_canary(
        self, agent_type: str, *, reason: str = "manual_retire"
    ) -> Dict[str, Any]:
        """Drop the current canary back to retired (active untouched)."""
        state = await self.get_artefact_state(agent_type)
        if not state.get("canary"):
            return state
        state.setdefault("retired", []).append(
            {
                "version": state["canary"]["version"],
                "retired_at": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
            }
        )
        state["canary"] = None
        await self._save_artefact_state(agent_type, state)
        return state

    async def _restore_active_from_version(self, agent_type: str, version: int) -> None:
        """Copy a versioned artefact pair (prompts + demos) into the active slot."""
        prompts_name = self._versioned_dataset_name("prompts", agent_type, version)
        demos_name = self._versioned_dataset_name("demos", agent_type, version)
        try:
            df = await self._provider.datasets.get_dataset(name=prompts_name)
            prompts = self._extract_prompts_from_dataframe(df)
            await self.save_prompts(agent_type, prompts)
        except (KeyError, ValueError):
            logger.warning(
                "No versioned prompts at v%d for %s/%s; active prompts left as-is",
                version,
                self._tenant_id,
                agent_type,
            )
        try:
            df = await self._provider.datasets.get_dataset(name=demos_name)
            demos = df.to_dict(orient="records")
            await self.save_demonstrations(agent_type, demos)
        except (KeyError, ValueError):
            logger.debug(
                "No versioned demos at v%d for %s/%s",
                version,
                self._tenant_id,
                agent_type,
            )

    @staticmethod
    def _route_to_canary(request_seed: str, traffic_pct: int) -> bool:
        """Stable routing decision: hash(request_seed) → canary iff in band."""
        import hashlib as _hashlib

        digest = _hashlib.sha1(request_seed.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 100
        return bucket < int(traffic_pct)

    async def load_for_request(
        self,
        agent_type: str,
        *,
        request_seed: str,
        variant_id: str = DEFAULT_VARIANT_ID,
    ) -> Dict[str, Any]:
        """Choose canary vs active for a given request and return its prompts.

        Returns ``{"prompts": ..., "served_from": "active|canary|default",
        "version": int | None, "variant_id": str}``. Stable per request_seed.

        When ``variant_id`` is non-default, the dataset names
        consulted are the variant-qualified ones
        (``dspy-prompts-{tenant}-{agent}::variant={vid}-vN``). Two variants
        of the same agent therefore have entirely separate canary state +
        prompts datasets, so an operator can canary a tenant-specific
        signature variant without disturbing the default.
        """
        # Apply variant qualification once; downstream lookups use this key.
        agent_key = self.qualified_agent_key(agent_type, variant_id)
        state = await self._request_state(agent_key)
        canary = state.get("canary")
        active = state.get("active")

        if canary and self._route_to_canary(
            request_seed, int(canary.get("traffic_pct", 10))
        ):
            try:
                prompts = await self._request_prompts(
                    self._versioned_dataset_name(
                        "prompts", agent_key, int(canary["version"])
                    )
                )
                return {
                    "prompts": prompts,
                    "served_from": "canary",
                    "version": int(canary["version"]),
                    "variant_id": variant_id,
                }
            except (KeyError, ValueError):
                logger.warning(
                    "canary v%d dataset missing for %s/%s (variant=%s); falling back",
                    canary["version"],
                    self._tenant_id,
                    agent_type,
                    variant_id,
                )

        if active is not None:
            try:
                prompts = await self._request_prompts(
                    self._versioned_dataset_name(
                        "prompts", agent_key, int(active["version"])
                    )
                )
                return {
                    "prompts": prompts,
                    "served_from": "active",
                    "version": int(active["version"]),
                    "variant_id": variant_id,
                }
            except (KeyError, ValueError):
                pass

        return {
            "prompts": await self._request_default_prompts(agent_key),
            "served_from": "default",
            "version": None,
            "variant_id": variant_id,
        }

    async def _request_state(self, agent_key: str) -> Dict[str, Any]:
        """Cached artefact-state read for the hot request path."""
        cached = self._request_cache_get(f"state::{agent_key}")
        if cached is not _CACHE_MISS:
            return cached
        state = await self.get_artefact_state(agent_key)
        self._request_cache_put(f"state::{agent_key}", state)
        return state

    async def _request_prompts(self, dataset_name: str) -> Dict[str, str]:
        """Cached versioned-prompts read; raises on a missing dataset so the
        caller's fallback logic still runs."""
        cached = self._request_cache_get(f"prompts::{dataset_name}")
        if cached is not _CACHE_MISS:
            return cached
        df = await self._provider.datasets.get_dataset(name=dataset_name)
        prompts = self._extract_prompts_from_dataframe(df)
        self._request_cache_put(f"prompts::{dataset_name}", prompts)
        return prompts

    async def _request_default_prompts(
        self, agent_key: str
    ) -> Optional[Dict[str, str]]:
        """Cached default (un-versioned) prompts read."""
        cached = self._request_cache_get(f"defaultprompts::{agent_key}")
        if cached is not _CACHE_MISS:
            return cached
        prompts = await self.load_prompts(agent_key)
        self._request_cache_put(f"defaultprompts::{agent_key}", prompts)
        return prompts

    async def snapshot_active(self, agent_type: str) -> Optional[Dict[str, Any]]:
        """Snapshot the current active prompts + demos as a versioned pair.

        Used by ``promote_if_better`` *before* it overwrites the active
        artefacts so a rollback CLI has something to restore. Returns a
        dict ``{prompts_version, demos_version}`` or None when there are no
        active artefacts to snapshot.
        """
        active_prompts = await self.load_prompts(agent_type)
        active_demos = await self.load_demonstrations(agent_type)
        if not active_prompts and not active_demos:
            return None

        snapshot: Dict[str, Any] = {}
        if active_prompts:
            _, v_p = await self.save_prompts_versioned(agent_type, active_prompts)
            snapshot["prompts_version"] = v_p
        if active_demos:
            _, v_d = await self.save_demonstrations_versioned(agent_type, active_demos)
            snapshot["demos_version"] = v_d
        return snapshot

    async def rollback_to_version(
        self,
        agent_type: str,
        prompts_version: Optional[int] = None,
        demos_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Restore active artefacts from a previously-snapshotted version.

        Reads the ``dspy-prompts-{tenant}-{agent}-v{N}`` (and demos) datasets
        and re-promotes their content as the active ones. The currently-active
        artefacts are themselves snapshotted first so the rollback is itself
        reversible. Returns a summary dict.
        """
        # Snapshot current state so the operator can undo the rollback.
        backup_versions = await self.snapshot_active(agent_type) or {}

        result: Dict[str, Any] = {
            "agent_type": agent_type,
            "backup_versions": backup_versions,
            "restored": {},
        }

        if prompts_version is not None:
            name = self._versioned_dataset_name("prompts", agent_type, prompts_version)
            try:
                df = await self._provider.datasets.get_dataset(name=name)
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"prompts version {prompts_version} not found for "
                    f"{self._tenant_id}/{agent_type}"
                ) from exc
            prompts = self._extract_prompts_from_dataframe(df)
            await self.save_prompts(agent_type, prompts)
            result["restored"]["prompts_version"] = prompts_version

        if demos_version is not None:
            name = self._versioned_dataset_name("demos", agent_type, demos_version)
            try:
                df = await self._provider.datasets.get_dataset(name=name)
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"demos version {demos_version} not found for "
                    f"{self._tenant_id}/{agent_type}"
                ) from exc
            demos = df.to_dict(orient="records")
            await self.save_demonstrations(agent_type, demos)
            result["restored"]["demos_version"] = demos_version

        if not result["restored"]:
            raise ValueError(
                "rollback_to_version called with both prompts_version and "
                "demos_version as None — nothing to restore"
            )
        logger.info(
            "Rolled back %s/%s to %s (backup snapshots: %s)",
            self._tenant_id,
            agent_type,
            result["restored"],
            backup_versions,
        )
        return result

    async def promote_if_better(
        self,
        agent_type: str,
        candidate_prompts: Dict[str, str],
        candidate_demos: Optional[List[Dict[str, Any]]],
        baseline_score: float,
        candidate_score: float,
        *,
        tolerance: float = 0.0,
        optimizer: str = "unknown",
        train_examples: Optional[int] = None,
        run_id: Optional[str] = None,
        extra_metrics: Optional[Dict[str, Any]] = None,
        snapshot_before_promote: bool = True,
    ) -> ExperimentMetrics:
        """Regression-reject gate — promote a candidate only when it wins.

        Compares candidate against baseline on a held-out score. The candidate
        is *promoted* (prompts + demos saved, becoming the active artefact)
        only when ``candidate_score >= baseline_score - tolerance``. Otherwise
        it is *rejected*: prompts/demos are NOT saved, and the experiment is
        recorded with ``promoted=False`` plus a ``rejection_reason`` so
        operators can audit why a recompile did not flip active.

        Either outcome lands as a typed ``ExperimentMetrics`` row in the
        per-agent experiments dataset, so the loop is observable end-to-end:
        rejected runs stay in the ledger with their scores, not silently
        discarded.

        Args:
            agent_type: Agent identifier the artefacts belong to.
            candidate_prompts: New prompts to consider for promotion.
            candidate_demos: Optional new demos.
            baseline_score: Score of the currently-active artefacts.
            candidate_score: Score of the candidate on the same eval set.
            tolerance: Allowed regression band; defaults to 0 (strict win).
                Pass a small positive value (e.g. 0.005) to tolerate noise.
            optimizer: Name of the optimizer that produced the candidate.
            train_examples: Number of examples the optimizer compiled against.
            run_id: Caller-supplied id; auto-generated when omitted.
            extra_metrics: Additional metrics to log under ``extra_metrics``.

        Returns:
            The ``ExperimentMetrics`` record persisted for this run. Inspect
            ``record.promoted`` to branch on the outcome.
        """
        if tolerance < 0:
            raise ValueError(f"tolerance must be >= 0; got {tolerance}")

        improvement = candidate_score - baseline_score
        threshold = baseline_score - tolerance
        promoted = candidate_score >= threshold

        rid = run_id or _generate_run_id()
        extras: Dict[str, Any] = dict(extra_metrics or {})
        extras["tolerance"] = tolerance

        if promoted:
            # Snapshot the about-to-be-overwritten active artefacts as
            # versioned datasets so a future rollback can restore them. Best
            # effort: snapshot failures are logged and recorded in extras
            # but never block the promotion itself.
            if snapshot_before_promote:
                try:
                    snap = await self.snapshot_active(agent_type)
                    if snap:
                        extras["pre_promote_snapshot"] = snap
                except Exception as exc:
                    logger.warning(
                        "Pre-promote snapshot failed for %s/%s: %s — promotion "
                        "will proceed but rollback may not be available",
                        self._tenant_id,
                        agent_type,
                        exc,
                    )
            await self.save_prompts(agent_type, candidate_prompts)
            if candidate_demos is not None:
                await self.save_demonstrations(agent_type, candidate_demos)
            logger.info(
                "Promoted candidate for %s/%s: candidate=%.4f baseline=%.4f "
                "(improvement=%.4f, tolerance=%.4f)",
                self._tenant_id,
                agent_type,
                candidate_score,
                baseline_score,
                improvement,
                tolerance,
            )
        else:
            extras["rejection_reason"] = (
                f"candidate_score={candidate_score:.4f} < "
                f"baseline_score - tolerance ({threshold:.4f}); "
                f"regression of {-improvement:.4f}"
            )
            logger.warning(
                "Rejected candidate for %s/%s: candidate=%.4f baseline=%.4f "
                "(regression=%.4f, tolerance=%.4f) — active artefacts "
                "unchanged, experiment logged for audit",
                self._tenant_id,
                agent_type,
                candidate_score,
                baseline_score,
                -improvement,
                tolerance,
            )

        record = ExperimentMetrics(
            tenant_id=self._tenant_id,
            agent_type=agent_type,
            run_id=rid,
            timestamp=datetime.now(timezone.utc).isoformat(),
            optimizer=optimizer,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            improvement=improvement,
            promoted=promoted,
            train_examples=train_examples,
            extra_metrics=extras,
        )
        await self.save_experiment(record)
        return record

    async def save_experiment(self, metrics: ExperimentMetrics) -> str:
        """Persist a typed ``ExperimentMetrics`` record.

        Each call appends a row to ``dspy-experiments-{tenant}-{agent}``.
        PhoenixProvider's ``append_to_dataset`` adds examples to the same
        dataset (a new Phoenix version), so ``get_dataset`` always returns
        the full history.

        The previous workaround (``save_blob`` overwriting per-agent) lost
        history every run; this implementation makes the experiment ledger
        a real, queryable artefact comparable to prompts/demos.
        """
        if metrics.tenant_id != self._tenant_id:
            raise ValueError(
                f"ExperimentMetrics.tenant_id={metrics.tenant_id!r} does not "
                f"match ArtifactManager.tenant_id={self._tenant_id!r}"
            )
        dataset_name = self._experiments_dataset_name(metrics.agent_type)
        row_df = pd.DataFrame([metrics.to_row()])

        # First run: create the dataset; subsequent runs: append. Every
        # metric column is sent as metadata_keys so Phoenix's
        # to_dataframe() round-trip preserves them under their original
        # names — without metadata_keys, Phoenix collapses unclassified
        # columns and the from_row read raises KeyError on tenant_id.
        all_columns = list(row_df.columns)
        try:
            await self._provider.datasets.append_to_dataset(
                name=dataset_name,
                data=row_df,
                metadata={"metadata_keys": all_columns},
            )
            return dataset_name
        except (KeyError, ValueError):
            dataset_id = await self._provider.datasets.create_dataset(
                name=dataset_name,
                data=row_df,
                metadata={
                    "artifact_type": "dspy_experiments",
                    "agent_type": metrics.agent_type,
                    "tenant_id": self._tenant_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "metadata_keys": all_columns,
                },
            )
            logger.info(
                "Created experiments dataset for %s/%s → %s",
                self._tenant_id,
                metrics.agent_type,
                dataset_id,
            )
            return dataset_id

    async def load_experiments(self, agent_type: str) -> List[ExperimentMetrics]:
        """Return the full experiment history for an agent in chronological order.

        ``get_dataset`` on the base name returns the latest Phoenix version,
        which carries every appended row. Empty list when no experiments
        have been logged.
        """
        dataset_name = self._experiments_dataset_name(agent_type)
        latest_df = await self._latest_versioned_dataset(dataset_name)
        if latest_df is None or latest_df.empty:
            return []

        records = [
            ExperimentMetrics.from_row(row.to_dict()) for _, row in latest_df.iterrows()
        ]
        # Defensive sort: timestamps are ISO-8601, so lexical sort == chronological.
        records.sort(key=lambda m: m.timestamp)
        return records

    async def load_latest_experiment(
        self, agent_type: str
    ) -> Optional[ExperimentMetrics]:
        """Return the most recent experiment record, or None if none exist."""
        history = await self.load_experiments(agent_type)
        return history[-1] if history else None

    async def _latest_versioned_dataset(self, base_name: str) -> Optional[pd.DataFrame]:
        """Load the dataset's full history.

        The base dataset carries every appended row (Phoenix versions the
        same dataset on append). The ``{base_name}_v{ts}`` enumeration below
        remains only to read side datasets written by the pre-native-append
        implementation.
        """
        # The base name is authoritative — appends land here.
        try:
            df = await self._provider.datasets.get_dataset(name=base_name)
            return df
        except (KeyError, ValueError):
            pass

        # Otherwise enumerate ``base_name_vYYYYMMDD_HHMMSS`` datasets via
        # the provider's listing API if available, fall back to a
        # bounded scan of recent timestamps.
        listing = getattr(self._provider.datasets, "list_datasets", None)
        if listing is not None:
            try:
                names = await listing()
            except Exception:
                names = []
            candidates = [n for n in names if n.startswith(f"{base_name}_v")]
            if candidates:
                latest = sorted(candidates)[-1]
                return await self._provider.datasets.get_dataset(name=latest)
        return None


def _generate_run_id() -> str:
    """Cheap monotonic-ish run id without pulling in extra deps."""
    import uuid

    return uuid.uuid4().hex


def load_optimized_module(agent: Any, blob_key: str) -> None:
    """Load a compiled DSPy module blob into ``agent.dspy_module``.

    Shared ``_load_artifact`` body for the dispatcher-served DSPy agents.
    Records ``agent.artifact_load_status`` ∈ {``no_telemetry``,
    ``no_artifact``, ``loaded``, ``error``} so a telemetry outage (silent
    reversion to the base module) is distinguishable from "tenant never
    optimized". Failures log at WARNING and never raise — the agent keeps
    serving on defaults.
    """
    agent.artifact_load_status = "no_telemetry"
    if not getattr(agent, "telemetry_manager", None):
        return
    try:
        from cogniverse_core.common.utils.async_bridge import run_coro_blocking

        tenant_id = getattr(agent, "_artifact_tenant_id", None)
        if not tenant_id:
            raise RuntimeError(
                f"{type(agent).__name__}._load_artifact called before the "
                f"dispatcher injected _artifact_tenant_id"
            )
        provider = agent.telemetry_manager.get_provider(tenant_id=tenant_id)
        am = ArtifactManager(provider, tenant_id)

        async def _load() -> Optional[str]:
            return await am.load_blob("model", blob_key)

        blob = run_coro_blocking(_load())
        if not blob:
            agent.artifact_load_status = "no_artifact"
            logger.info(
                "%s: no persisted %s artifact for tenant %s; using defaults",
                type(agent).__name__,
                blob_key,
                tenant_id,
            )
            return
        agent.dspy_module.load_state(json.loads(blob))
        agent.artifact_load_status = "loaded"
        logger.info(
            "%s loaded optimized DSPy module from artifact", type(agent).__name__
        )
    except Exception as e:
        agent.artifact_load_status = "error"
        logger.warning(
            "%s artifact load failed; using defaults: %s", type(agent).__name__, e
        )
