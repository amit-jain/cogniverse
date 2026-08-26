"""CLI entry point for optimization — called by Argo CronWorkflows.

Per-agent batch optimization modes. Each mode reads production spans from
Phoenix, builds DSPy training examples, compiles optimized modules, and
saves artifacts via ArtifactManager. Agents load artifacts at startup.

Usage:
    python -m cogniverse_runtime.optimization_cli --mode simba --tenant-id acme:production
    python -m cogniverse_runtime.optimization_cli --mode workflow --tenant-id acme:production
    python -m cogniverse_runtime.optimization_cli --mode gateway-thresholds --tenant-id acme:production
    python -m cogniverse_runtime.optimization_cli --mode online-routing-eval --tenant-id acme:production
    python -m cogniverse_runtime.optimization_cli --mode profile --tenant-id acme:production
    python -m cogniverse_runtime.optimization_cli --mode entity-extraction --tenant-id acme:production
    python -m cogniverse_runtime.optimization_cli --mode cleanup --log-retention-days 7
    python -m cogniverse_runtime.optimization_cli --mode triggered \
        --tenant-id acme:production --agents search,summary \
        --trigger-dataset optimization-trigger-acme-production-20260403_040000
"""

import argparse
import asyncio
import contextlib
import io
import json
import logging
import os
import sys
import uuid
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from cogniverse_agents.optimizer.example_selection import (
    TRAINING_SELECTION_DEFAULTS as _TRAINING_SELECTION_DEFAULTS,
)
from cogniverse_agents.optimizer.example_selection import (
    SelectionReport,
    TrainingSelectionKnobs,
    confirmation_stats,
    decay_weight,
    embed_texts,
    select_training_records,
)
from cogniverse_core.durable import (
    PipelineCheckpoint,
    PipelineCheckpointStatus,
    PipelineCheckpointStorage,
)
from cogniverse_foundation.telemetry.config import (
    SPAN_NAME_ENTITY_EXTRACTION,
    SPAN_NAME_PROFILE_SELECTION,
    SPAN_NAME_QUERY_ENHANCEMENT,
)
from cogniverse_foundation.telemetry.span_contract import (
    read_span_attributes,
    read_span_id,
    read_span_io,
)
from cogniverse_sdk.interfaces.schema_loader import SchemaLoader

logger = logging.getLogger(__name__)
SHIPPED_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "config.json"
PROFILE_SELECTION_LABEL_SOURCE_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "testset"
    / "evaluation"
    / "sample_videos_retrieval_queries.json"
)
_TRAINING_SELECTION_FIELDS = (
    ("trainset_cap", int),
    ("mmr_lambda", float),
    ("low_confirmation_threshold", int),
    ("downweight_age_days", int),
    ("downweight_factor", float),
    ("confirmation_score_threshold", float),
)


@contextlib.contextmanager
def _redirect_stdout_to_stderr():
    """Keep stdout reserved for the final JSON document.

    Library setup code can still emit human diagnostics, but they must land on
    stderr so callers can json.loads(stdout) reliably.
    """
    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, io.UnsupportedOperation):
        with contextlib.redirect_stdout(sys.stderr):
            yield
        return

    saved_stdout_fd = os.dup(stdout_fd)
    try:
        sys.stdout.flush()
        os.dup2(stderr_fd, stdout_fd)
        yield
    finally:
        try:
            sys.stdout.flush()
        finally:
            os.dup2(saved_stdout_fd, stdout_fd)
            os.close(saved_stdout_fd)


class _TriggeredOptCheckpointer:
    """Per-agent checkpoint + resume for ``run_triggered_optimization``.

    A killed Argo pod re-runs the whole ``--mode triggered`` invocation from
    scratch. When durable execution is enabled for the tenant, each successful
    agent compile is checkpointed (as a telemetry span keyed by a workflow id
    derived from tenant + trigger dataset), so a re-run skips the agents that
    already compiled instead of redoing their expensive DSPy ``compile()``.
    """

    def __init__(self, storage, workflow_id, tenant_id, agents, trigger_dataset):
        self._storage = storage
        self._workflow_id = workflow_id
        self._tenant_id = tenant_id
        self._agents = list(agents)
        self._trigger_dataset = trigger_dataset
        self._units: Dict[str, Dict[str, Any]] = {}
        self._resume_count = 0

    @classmethod
    async def maybe_resume(
        cls,
        *,
        config_manager,
        telemetry_manager,
        tenant_id: str,
        agents: List[str],
        trigger_dataset: str,
        phoenix_endpoint: str,
    ) -> Optional["_TriggeredOptCheckpointer"]:
        """Build a checkpointer (loading prior progress) if enabled, else None."""
        if not config_manager.get_durable_execution_config(tenant_id).enabled:
            return None

        provider_config = (
            getattr(telemetry_manager.config, "provider_config", None) or {}
        )
        grpc_endpoint = provider_config.get("grpc_endpoint") or getattr(
            telemetry_manager.config, "otlp_endpoint", None
        )
        http_endpoint = provider_config.get("http_endpoint") or phoenix_endpoint
        storage = PipelineCheckpointStorage(
            grpc_endpoint=grpc_endpoint,
            http_endpoint=http_endpoint,
            tenant_id=tenant_id,
            telemetry_manager=telemetry_manager,
        )
        workflow_id = f"opt_triggered:{tenant_id}:{trigger_dataset}"
        inst = cls(storage, workflow_id, tenant_id, agents, trigger_dataset)

        latest = await storage.get_latest_checkpoint(workflow_id)
        if (
            latest is not None
            and latest.status != PipelineCheckpointStatus.COMPLETED.value
        ):
            inst._units = dict(latest.completed_units)
            inst._resume_count = latest.resume_count + 1
            logger.info(
                "Resuming optimization %s from checkpoint (%d agents already done)",
                workflow_id,
                len(inst.done_agents()),
            )
        return inst

    def done_agents(self) -> set:
        return {
            agent
            for agent, unit in self._units.items()
            if isinstance(unit, dict) and unit.get("status") == "completed"
        }

    def stored_result(self, agent_name: str) -> Dict[str, Any]:
        return self._units[agent_name]["result"]

    async def record(self, agent_name: str, result: Dict[str, Any]) -> None:
        """Persist a checkpoint recording ``agent_name`` as completed."""
        self._units[agent_name] = {"status": "completed", "result": result}
        await self._save(PipelineCheckpointStatus.ACTIVE)

    async def finalize(self, had_failure: bool) -> None:
        """Mark the workflow completed when the run finished cleanly.

        On a failure the last ACTIVE checkpoint is left in place so the next
        run resumes and retries the un-compiled agents.
        """
        if not had_failure:
            await self._save(PipelineCheckpointStatus.COMPLETED)

    async def _save(self, status: PipelineCheckpointStatus) -> None:
        checkpoint = PipelineCheckpoint(
            checkpoint_id=f"ckpt_{uuid.uuid4().hex[:12]}",
            workflow_id=self._workflow_id,
            tenant_id=self._tenant_id,
            status=status.value,
            phases=self._agents,
            phase_index=len(self.done_agents()),
            completed_units=self._units,
            metadata={
                "trigger_dataset": self._trigger_dataset,
                "agents": self._agents,
            },
            created_at=datetime.now(timezone.utc),
            resume_count=self._resume_count,
        )
        await self._storage.save_checkpoint(checkpoint)


def _span_example_id(row: Any, *, span_name: str, position: int) -> str:
    """The ledger id of a served call: ``span:<context.span_id>``.

    A record without a span id cannot be attributed by the ledger, so a row
    that yields a training example must carry one; that is an error, never
    a ``span:None`` id.
    """
    span_id = read_span_id(row)
    if not span_id:
        raise ValueError(
            f"{span_name} span row {position} has no context.span_id; "
            "the optimizer cannot record which example it consumed"
        )
    return f"span:{span_id}"


def _query_enhancement_pairs(spans_df) -> List[Dict[str, Any]]:
    """One record per query_enhancement span: the served call's inputs and outputs.

    ``input.value`` / ``input.source_text`` / ``input.grounding_context`` are the
    prompt inputs; ``output.value`` carries every produced field. Rows without a
    query or an enhanced query are skipped. ``trainable`` marks a record worth
    serving as a demo — a non-identity enhancement with at least one expansion
    term; every record is still an evaluation probe (its inputs are real).
    """
    pairs: List[Dict[str, Any]] = []
    for position, (_, row) in enumerate(spans_df.iterrows()):
        span_io = read_span_io(row)
        original = span_io["input"] or ""
        output = span_io["output"] if isinstance(span_io["output"], dict) else {}
        enhanced = output.get("enhanced_query", "") or ""
        if not original or not enhanced:
            continue
        example_id = _span_example_id(
            row, span_name=SPAN_NAME_QUERY_ENHANCEMENT, position=position
        )
        attrs = read_span_attributes(row)
        expansion_terms = [str(t) for t in output.get("expansion_terms", []) or []]
        pairs.append(
            {
                "query": original,
                "source_text": str(attrs.get("input.source_text") or ""),
                "grounding_context": str(attrs.get("input.grounding_context") or ""),
                "enhanced_query": enhanced,
                "expansion_terms": expansion_terms,
                "synonyms": [str(s) for s in output.get("synonyms", []) or []],
                "context": [str(c) for c in output.get("context_additions", []) or []],
                "confidence": float(output.get("confidence", 0.0) or 0.0),
                "example_id": example_id,
                "trainable": (
                    enhanced.strip().lower() != original.strip().lower()
                    and bool(expansion_terms)
                ),
            }
        )
    return pairs


def _query_enhancement_quality(prediction, example) -> float | None:
    """1.0 for a usable enhancement of ``example``'s inputs, else 0.0 or None.

    Usable: a non-empty enhanced query that differs from the query, at least
    one expansion term, and when a scoreable source context is present at
    least one grounded expansion term. If the example has neither source text
    nor grounding context, it is unscoreable and returns ``None``. When a
    grounding context was supplied, at least one of its entity names must still
    be present in the enhanced query or the expansion terms. Labels are not
    consulted: the score is a property of the module's own output for real
    inputs.
    """
    from cogniverse_agents.query_enhancement_agent import grounding_entity_names
    from cogniverse_synthetic.grounding import source_term_keys, term_is_grounded

    query = str(getattr(example, "query", "") or "").strip().lower()
    enhanced = str(getattr(prediction, "enhanced_query", "") or "").strip()
    source_text = str(getattr(example, "source_text", "") or "").strip()
    grounding_context = str(getattr(example, "grounding_context", "") or "").strip()
    raw_terms = getattr(prediction, "expansion_terms", "") or ""
    if isinstance(raw_terms, str):
        terms = [t.strip() for t in raw_terms.split(",") if t.strip()]
    elif isinstance(raw_terms, (list, tuple, set)):
        terms = [str(t).strip() for t in raw_terms if str(t).strip()]
    else:
        terms = [t.strip() for t in str(raw_terms).split(",") if t.strip()]

    if not source_text and not grounding_context:
        return None
    if not enhanced or enhanced.lower() == query or not terms:
        return 0.0

    if source_text:
        source_keys = source_term_keys(source_text)
        if not any(term_is_grounded(term, source_keys) for term in terms):
            return 0.0

    names = grounding_entity_names(grounding_context)
    if names:
        haystack = " ".join([enhanced, *terms]).lower()
        if not any(name.lower() in haystack for name in names):
            return 0.0
    return 1.0


def teacher_lm_or_raise(llm_config, *, probe=None):
    """Build the DSPy teacher LM, refusing when nothing serves the endpoint.

    BootstrapFewShot asks the teacher for every demonstration, so a declared
    but unserved teacher turns each request into a failure the caller degrades
    to a fallback: the job walks the whole trainset, collects no demos and
    hits its timeout with no indication of why. Probe first and refuse,
    naming the endpoint and the model mismatch when there is one.
    """
    from cogniverse_foundation.config.llm_factory import create_dspy_lm

    endpoint = require_reachable_teacher(llm_config, probe=probe)
    return create_dspy_lm(endpoint)


def require_reachable_teacher(llm_config, *, probe=None):
    """Return the teacher endpoint, refusing when nothing serves it."""
    from cogniverse_runtime.inference_health_check import probe_service_model

    endpoint = llm_config.resolve_teacher()
    api_base = (getattr(endpoint, "api_base", None) or "").rstrip("/")
    if not api_base:
        raise RuntimeError(
            "LLM teacher endpoint declares no api_base; DSPy bootstrap "
            "requires a reachable teacher service"
        )
    service_root = api_base[: -len("/v1")] if api_base.endswith("/v1") else api_base

    served = (probe or probe_service_model)(service_root)
    if served is None:
        raise RuntimeError(
            f"LLM teacher endpoint {api_base} is unreachable; DSPy bootstrap "
            f"requires it to generate demonstrations. Enable "
            f"inference.vllm_llm_teacher and wait for the pod to report ready."
        )

    configured = getattr(endpoint, "model", "") or ""
    if served not in configured:
        raise RuntimeError(
            f"LLM teacher endpoint {api_base} serves {served!r} but the "
            f"configured teacher model is {configured!r}; requests for an "
            f"unserved model id are rejected"
        )
    return endpoint


def _query_enhancement_metric(example, prediction, trace=None) -> bool:
    """BootstrapFewShot metric: keep a teacher trace only when it is usable."""
    del trace
    return _query_enhancement_quality(prediction, example) == 1.0


def _profile_selection_value(record, key: str):
    if isinstance(record, dict):
        return record.get(key)
    return getattr(record, key, None)


class ProfileLabelDerivationResult(dict):
    """Derived profile labels plus the exclusions the producer surfaced."""

    def __init__(
        self,
        labels: Dict[str, str],
        records: list[Dict[str, Any]],
        exclusions: list[Dict[str, Any]],
    ):
        super().__init__(labels)
        self.records = tuple(records)
        self.exclusions = tuple(exclusions)
        self.excluded_count = len(exclusions)
        self.excluded_queries = tuple(
            str(exclusion.get("query", "")).strip() for exclusion in exclusions
        )


def _profile_selection_content_key(value: Any) -> str:
    """Basename of ``value`` without its file extension."""
    name = Path(str(value).strip()).name
    suffix = Path(name).suffix
    extension = suffix[1:]
    if extension.isalnum() and any(ch.isalpha() for ch in extension):
        return name[: -len(suffix)]
    return name


def _profile_selection_result_titles(
    rows: Any, profile: str, title_field: str
) -> tuple[list[str], list[str]]:
    """Content keys of the titled ``SearchResult.to_dict()`` rows, plus the
    document ids of the rows that carry no title."""
    if isinstance(rows, (str, bytes, Mapping)):
        raise TypeError(
            f"Profile selection retrieval for profile {profile!r} must return "
            f"a sequence of mappings, got {type(rows).__name__}"
        )
    keys: list[str] = []
    untitled: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(
                f"Profile selection retrieval row {index} for profile {profile!r} "
                f"must be a mapping, got {type(row).__name__}"
            )
        metadata = row.get("metadata")
        title = metadata.get(title_field) if isinstance(metadata, Mapping) else None
        key = _profile_selection_content_key(title) if isinstance(title, str) else ""
        if key:
            keys.append(key)
        else:
            untitled.append(str(row.get("document_id", "")) or f"row:{index}")
    return keys, untitled


def _profile_selection_title_fields(
    config_manager: Any,
    tenant_id: str,
    candidate_profiles: Iterable[str],
    schema_loader: SchemaLoader,
) -> dict[str, str]:
    """Title field of each candidate profile, read from its schema's
    ``document_mapping``."""
    from cogniverse_sdk.document import DocumentFieldMapping

    title_fields: dict[str, str] = {}
    for profile in candidate_profiles:
        profile_config = config_manager.get_backend_profile(
            profile, tenant_id=tenant_id
        )
        if profile_config is None:
            raise ValueError(
                f"Profile selection derivation: profile {profile!r} is not "
                f"configured for tenant {tenant_id!r}"
            )
        schema_name = profile_config.schema_name
        if not schema_name:
            raise ValueError(
                f"Profile selection derivation: profile {profile!r} declares no "
                "schema_name"
            )
        schema = schema_loader.load_schema(schema_name)
        mapping_config = schema.get("document_mapping")
        if mapping_config is None:
            raise ValueError(
                f"Profile selection derivation: schema {schema_name!r} for profile "
                f"{profile!r} declares no document_mapping"
            )
        title_field = DocumentFieldMapping.from_dict(mapping_config).title
        if not title_field:
            raise ValueError(
                f"Profile selection derivation: schema {schema_name!r} for profile "
                f"{profile!r} declares no document_mapping.title"
            )
        title_fields[profile] = title_field
    return title_fields


def _profile_selection_query_key(
    query: str, position: int, query_counts: Counter[str]
) -> str:
    if query and query_counts[query] == 1:
        return query
    if query:
        return f"{query}#{position}"
    return f"row:{position}"


def _profile_selection_expected_videos(record: dict[str, Any]) -> list[str]:
    for key in ("expected_items", "expected_videos", "ground_truth"):
        value = record.get(key)
        if not value:
            continue
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        item = str(value).strip()
        return [item] if item else []
    return []


def _profile_selection_recovery_score(
    expected_videos: Sequence[str], retrieved_ids: Sequence[str]
) -> float:
    expected = {str(item).strip() for item in expected_videos if str(item).strip()}
    if not expected:
        return 0.0
    retrieved = {str(item).strip() for item in retrieved_ids if str(item).strip()}
    return len(expected & retrieved) / len(expected)


def _load_profile_selection_label_source(
    queries_path: Path = PROFILE_SELECTION_LABEL_SOURCE_PATH,
) -> list[dict[str, Any]]:
    try:
        with queries_path.open(encoding="utf-8") as source_file:
            loaded = json.load(source_file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Profile selection label source missing: {queries_path}"
        ) from exc
    except OSError as exc:
        raise OSError(
            f"Profile selection label source unreadable: {queries_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Profile selection label source unreadable JSON: {queries_path}"
        ) from exc

    if not isinstance(loaded, list):
        raise ValueError(
            f"Profile selection label source must be a JSON list: {queries_path}"
        )
    if not loaded:
        raise ValueError(f"Profile selection label source is empty: {queries_path}")
    return loaded


def derive_profile_labels(
    queries: Iterable[dict[str, Any]],
    candidate_profiles: Iterable[str],
    retrieve: Callable[[str, str], Sequence[Mapping[str, Any]]],
    *,
    title_fields: Mapping[str, str],
) -> ProfileLabelDerivationResult:
    query_rows = list(queries)
    if not query_rows:
        raise ValueError("Profile selection label source is empty")

    profiles = [
        str(profile).strip() for profile in candidate_profiles if str(profile).strip()
    ]
    if not profiles:
        raise ValueError("Profile selection derivation requires candidate profiles")
    for profile in profiles:
        title_field = title_fields.get(profile)
        if not isinstance(title_field, str) or not title_field.strip():
            raise ValueError(
                f"Profile selection derivation has no title field for profile "
                f"{profile!r}"
            )

    query_counts = Counter(
        str(row.get("query", "")).strip() for row in query_rows if isinstance(row, dict)
    )

    labels: Dict[str, str] = {}
    records: list[Dict[str, Any]] = []
    exclusions: list[Dict[str, Any]] = []

    for position, query_record in enumerate(query_rows):
        if not isinstance(query_record, dict):
            raise TypeError(
                f"Profile selection label source row {position} must be a dict"
            )

        query = str(query_record.get("query", "") or "").strip()
        query_key = _profile_selection_query_key(query, position, query_counts)
        expected_videos = _profile_selection_expected_videos(query_record)

        if not query:
            exclusions.append(
                {
                    "query": query_key,
                    "reason": "missing_query",
                    "position": position,
                    "expected_videos": expected_videos,
                }
            )
            continue

        if not expected_videos:
            exclusions.append(
                {
                    "query": query,
                    "reason": "missing_expected_videos",
                    "position": position,
                }
            )
            continue

        expected_keys = [
            _profile_selection_content_key(video) for video in expected_videos
        ]
        scored_profiles: list[dict[str, Any]] = []
        untitled_results: list[dict[str, Any]] = []
        last_error: Exception | None = None
        for profile in profiles:
            try:
                rows = retrieve(query, profile)
            except Exception as exc:
                last_error = exc
                continue

            retrieved, untitled = _profile_selection_result_titles(
                rows, profile, title_fields[profile]
            )
            if untitled:
                untitled_results.append(
                    {
                        "profile": profile,
                        "title_field": title_fields[profile],
                        "document_ids": untitled,
                    }
                )
            scored_profiles.append(
                {
                    "profile": profile,
                    "retrieved": retrieved,
                    "score": _profile_selection_recovery_score(
                        expected_keys, retrieved
                    ),
                }
            )

        if not scored_profiles:
            raise RuntimeError(
                f"Profile selection retrieval failed for query {query!r}"
            ) from last_error

        if untitled_results:
            exclusions.append(
                {
                    "query": query,
                    "reason": "result_missing_title",
                    "expected_videos": expected_videos,
                    "position": position,
                    "untitled_results": untitled_results,
                }
            )
            continue

        best_score = max(profile_result["score"] for profile_result in scored_profiles)
        best_profiles = [
            profile_result["profile"]
            for profile_result in scored_profiles
            if profile_result["score"] == best_score
        ]

        if best_score < 1.0:
            exclusion: dict[str, Any] = {
                "query": query,
                "reason": "no_profile_recovered_expected_videos",
                "expected_videos": expected_videos,
                "position": position,
                "best_score": best_score,
                "candidate_profiles": [result["profile"] for result in scored_profiles],
            }
            exclusions.append(exclusion)
            continue

        if len(best_profiles) > 1:
            exclusions.append(
                {
                    "query": query,
                    "reason": "ambiguous_profile_tie",
                    "expected_videos": expected_videos,
                    "tied_profiles": best_profiles,
                    "best_score": best_score,
                    "position": position,
                }
            )
            continue

        selected_profile = best_profiles[0]
        labels[query_key] = selected_profile
        record = {
            "query": query,
            "expected_videos": expected_videos,
            "available_profiles": list(profiles),
            "selected_profile": selected_profile,
            "confidence": best_score,
            "reasoning": f"{selected_profile} recovered {', '.join(expected_videos)}",
            "example_id": f"span:profile-label:{position}",
        }
        for key in ("ground_truth", "query_type", "source"):
            if key in query_record and query_record[key] not in (None, ""):
                record[key] = query_record[key]
        records.append(record)

    return ProfileLabelDerivationResult(labels, records, exclusions)


def _load_profile_selection_labels(
    *,
    queries_path: Path = PROFILE_SELECTION_LABEL_SOURCE_PATH,
    candidate_profiles: Iterable[str],
    retrieve: Callable[[str, str], Sequence[Mapping[str, Any]]],
    title_fields: Mapping[str, str],
) -> ProfileLabelDerivationResult:
    queries = _load_profile_selection_label_source(queries_path)
    return derive_profile_labels(
        queries, candidate_profiles, retrieve, title_fields=title_fields
    )


def _profile_selection_label_source(
    *,
    config: Any,
    config_manager: Any,
    tenant_id: str,
    candidate_profiles: Sequence[str],
    schema_loader: SchemaLoader,
) -> ProfileLabelDerivationResult:
    """Derive the tenant's profile labels by running the shipped label source
    through its SearchService per candidate profile."""
    title_fields = _profile_selection_title_fields(
        config_manager, tenant_id, candidate_profiles, schema_loader
    )
    search_service: Any | None = None

    def retrieve(query: str, profile: str) -> list[dict[str, Any]]:
        nonlocal search_service
        if search_service is None:
            from cogniverse_agents.search.service import SearchService

            search_service = SearchService(
                config=config,
                config_manager=config_manager,
                schema_loader=schema_loader,
            )

        results = search_service.search(
            query=query,
            profile=profile,
            tenant_id=tenant_id,
            top_k=10,
        )
        return [result.to_dict() for result in results]

    return _load_profile_selection_labels(
        candidate_profiles=candidate_profiles,
        retrieve=retrieve,
        title_fields=title_fields,
    )


def _profile_selection_pool(available_profiles) -> list[str]:
    if isinstance(available_profiles, str):
        return [
            profile.strip()
            for profile in available_profiles.split(",")
            if profile.strip()
        ]
    if isinstance(available_profiles, (list, tuple, set)):
        return [
            str(profile).strip()
            for profile in available_profiles
            if str(profile).strip()
        ]
    raw = str(available_profiles or "").strip()
    return [raw] if raw else []


def _profile_selection_quality(prediction, example) -> float:
    """1.0 only for the exact recorded profile inside the recorded pool."""
    selected = str(
        _profile_selection_value(prediction, "selected_profile") or ""
    ).strip()
    recorded = str(_profile_selection_value(example, "selected_profile") or "").strip()
    available = _profile_selection_pool(
        _profile_selection_value(example, "available_profiles")
    )
    if not selected or not recorded or not available:
        return 0.0
    return 1.0 if selected == recorded and recorded in available else 0.0


def _profile_selection_metric(example, prediction, trace=None) -> bool:
    """BootstrapFewShot metric: keep only exact profile matches."""
    del trace
    return _profile_selection_quality(prediction, example) == 1.0


def _profile_selection_scores(module, holdout) -> float:
    """Mean ``_profile_selection_quality`` over the held-out profile spans."""
    scores = []
    for example in holdout:
        prediction = module(
            **{k: getattr(example, k) for k in _PROFILE_SELECTION_INPUTS}
        )
        scores.append(_profile_selection_quality(prediction, example))
    return sum(scores) / len(scores) if scores else 0.0


def _select_simba_artifact(
    baseline_score: float,
    current_score: Optional[float],
    candidate_score: Optional[float],
    min_improvement: float,
) -> str:
    """Which module the tenant should serve after this run.

    ``promote``: persist the compiled candidate — it scores at least
    ``min_improvement`` above whatever is served today (the base module, or
    the persisted artifact when that beats base). ``keep``: the persisted
    artifact stays. ``rollback``: the persisted artifact scores below the
    base module, so the base state is persisted in its place. ``reject``:
    nothing is served and the candidate did not earn it.
    """
    served = (
        baseline_score if current_score is None else max(baseline_score, current_score)
    )
    if candidate_score is not None and candidate_score >= served + min_improvement:
        return "promote"
    if current_score is None:
        return "reject"
    if current_score < baseline_score:
        return "rollback"
    return "keep"


def _entity_extraction_pairs(spans_df) -> List[Dict[str, Any]]:
    """(query -> entities) training pairs from entity_extraction spans.

    Reads the canonical span slots: input.value holds the query, output.value
    holds ``{"entities": [...], ...}``.
    """
    pairs: List[Dict[str, Any]] = []
    for position, (_, row) in enumerate(spans_df.iterrows()):
        span_io = read_span_io(row)
        query = span_io["input"] or ""
        output = span_io["output"] if isinstance(span_io["output"], dict) else {}
        entities = output.get("entities", [])
        if not query or not entities:
            continue
        pairs.append(
            {
                "query": query,
                "entities": entities,
                "example_id": _span_example_id(
                    row, span_name=SPAN_NAME_ENTITY_EXTRACTION, position=position
                ),
            }
        )
    return pairs


def _entity_extraction_is_scoreable(record: dict) -> bool:
    """True when the served entity span recorded at least one entity."""
    return bool(record.get("entities"))


def _entity_extraction_texts(raw: Any) -> list[str]:
    """Extract entity texts from JSON records or pipe-delimited output lines."""
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        else:
            raw = parsed

        if parsed is None:
            texts = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                head = line.split("|", 1)[0].strip()
                if head:
                    texts.append(head)
            return texts

    if isinstance(raw, dict):
        raw = [raw]

    if isinstance(raw, (list, tuple, set)):
        texts = []
        for entity in raw:
            if isinstance(entity, dict):
                value = entity.get("text", "")
            elif isinstance(entity, str):
                value = entity.split("|", 1)[0]
            else:
                value = getattr(entity, "text", "")
            text = str(value or "").strip()
            if text:
                texts.append(text)
        return texts

    text = str(raw or "").strip()
    return [text] if text else []


def _entity_extraction_quality(prediction, example) -> float:
    """Token-set F1 between predicted entity texts and the recorded texts."""
    predicted_texts = _entity_extraction_texts(getattr(prediction, "entities", ""))
    recorded_texts = _entity_extraction_texts(getattr(example, "entities", ""))
    query = str(getattr(example, "query", "") or "").strip()
    if not recorded_texts:
        raise ValueError(
            f"entity extraction example for query {query!r} carries no recorded "
            "entities"
        )
    return _token_f1(" ".join(predicted_texts), " ".join(recorded_texts))


ENTITY_BOOTSTRAP_METRIC_THRESHOLD = 1.0


def _entity_bootstrap_threshold(
    baseline_score: float,
    current_score: Optional[float],
    *,
    bar: float = ENTITY_BOOTSTRAP_METRIC_THRESHOLD,
) -> float:
    """Token-set F1 a teacher trace must reach to become a bootstrapped demo:
    ``bar``, never below what the served module already scores on the holdout.
    """
    served = (
        baseline_score if current_score is None else max(baseline_score, current_score)
    )
    return max(bar, served)


def _entity_pipe_lines(entities) -> str:
    """Recorded entity dicts as the signature's ``text|type|confidence`` lines."""
    lines = []
    for entity in entities:
        text = str(entity.get("text") or "").strip()
        if not text:
            continue
        confidence = entity.get("confidence")
        confidence = 1.0 if confidence is None else round(float(confidence), 2)
        lines.append(f"{text}|{entity.get('type') or 'CONCEPT'}|{confidence}")
    return "\n".join(lines)


def _entity_extraction_example(record: Dict[str, Any]):
    """A served or approved entity record as the module's training example."""
    import dspy

    entities = record["entities"]
    return dspy.Example(
        query=record["query"],
        entities=entities
        if isinstance(entities, str)
        else _entity_pipe_lines(entities),
        entity_types=str(record.get("entity_types") or ""),
    ).with_inputs("query")


class BootstrapMetricRecorder:
    """BootstrapFewShot metric returning the quality score of every teacher
    trace and recording it, so one compile yields the acceptance histogram.
    BootstrapFewShot compares the score against its ``metric_threshold``.
    """

    def __init__(self, quality, *, threshold: float):
        self._quality = quality
        self.threshold = threshold
        self.attempts: list[tuple[str, float]] = []

    def __call__(self, example, prediction, trace=None) -> float:
        del trace
        score = float(self._quality(prediction, example))
        query = str(getattr(example, "query", "") or "")
        self.attempts.append((query, score))
        logger.info(
            "bootstrap attempt %d query=%r metric=%.3f accepted=%s",
            len(self.attempts),
            query,
            score,
            score >= self.threshold,
        )
        return score


def _bootstrap_report(
    recorder: BootstrapMetricRecorder, teleprompter, compiled, trainset_size: int
) -> Dict[str, Any]:
    """What the bootstrap walk cost and what the compiled module carries."""
    scores = [score for _, score in recorder.attempts]
    demos = [
        demo for _, predictor in compiled.named_predictors() for demo in predictor.demos
    ]
    bootstrapped = sum(1 for demo in demos if demo.get("augmented", False))
    return {
        "trainset": trainset_size,
        "max_bootstrapped_demos": teleprompter.max_bootstrapped_demos,
        "max_labeled_demos": teleprompter.max_labeled_demos,
        "max_rounds": teleprompter.max_rounds,
        "metric_threshold": teleprompter.metric_threshold,
        "attempts": len(scores),
        "errors": teleprompter.error_count,
        "examples_walked": len({query for query, _ in recorder.attempts}),
        "accepted": sum(1 for score in scores if score >= recorder.threshold),
        "bootstrapped_demos": bootstrapped,
        "labeled_demos": len(demos) - bootstrapped,
        "metric_values": sorted(scores),
    }


def _entity_extraction_scores(module, holdout) -> float:
    """Mean ``_entity_extraction_quality`` over the held-out entity spans."""
    scores = []
    for example in holdout:
        prediction = module(query=getattr(example, "query", ""))
        scores.append(_entity_extraction_quality(prediction, example))
    return sum(scores) / len(scores) if scores else 0.0


def _span_available_profiles(row: Any) -> Optional[str]:
    """Read the candidate pool recorded on a profile_selection span."""
    getter = getattr(row, "get", None)
    if getter is None:
        attrs = row if isinstance(row, dict) else dict(row)
        raw = attrs.get("attributes.available_profiles")
        if raw is None:
            raw = attrs.get("available_profiles")
    else:
        raw = getter("attributes.available_profiles")
        if raw is None:
            raw = getter("available_profiles")

    if isinstance(raw, str):
        return raw if raw.strip() else None
    if isinstance(raw, list):
        profiles = [str(profile).strip() for profile in raw if str(profile).strip()]
        return ", ".join(profiles) if profiles else None
    return None


def _profile_selection_pairs(
    spans_df, *, config_manager, tenant_id
) -> List[Dict[str, Any]]:
    """(query -> selected_profile) training pairs from profile_selection spans.

    Reads the canonical span slots: input.value holds the query, output.value
    holds ``{"selected_profile", "modality", "complexity", "intent",
    "confidence"}``. Only high-confidence (>= 0.5) selections are kept.
    ``available_profiles`` comes from the span attribute when present; legacy
    spans derive it from the tenant's live usable profile set.
    """
    from cogniverse_agents.profile_selection_agent import tenant_usable_profile_names

    pairs: List[Dict[str, Any]] = []
    for position, (_, row) in enumerate(spans_df.iterrows()):
        span_io = read_span_io(row)
        query = span_io["input"] or ""
        output = span_io["output"] if isinstance(span_io["output"], dict) else {}
        selected = output.get("selected_profile", "")
        confidence = float(output.get("confidence", 0.0) or 0.0)
        if not query or not selected or confidence < 0.5:
            continue
        example_id = _span_example_id(
            row, span_name=SPAN_NAME_PROFILE_SELECTION, position=position
        )
        available_profiles = _span_available_profiles(row)
        if available_profiles is None:
            available_profiles = ", ".join(
                tenant_usable_profile_names(config_manager, tenant_id)
            )
        pairs.append(
            {
                "query": query,
                "available_profiles": available_profiles,
                "selected_profile": selected,
                "modality": output.get("modality", "video"),
                "complexity": output.get("complexity", "simple"),
                "intent": output.get("intent", ""),
                "confidence": confidence,
                "example_id": example_id,
            }
        )
    return pairs


async def run_triggered_optimization(
    tenant_id: str,
    agents: list[str],
    trigger_dataset: str,
    config_manager=None,
    phoenix_endpoint: str = None,
    telemetry_otlp_endpoint: str | None = None,
) -> dict:
    """Run optimization triggered by quality monitor.

    Loads scored examples from Phoenix trigger dataset, then compiles
    DSPy modules for each flagged agent using those examples as training data.

    Args:
        tenant_id: Tenant to optimize for.
        agents: List of agent names to optimize.
        trigger_dataset: Phoenix dataset name with scored trace examples.
        config_manager: Optional ConfigManager (for testing). If None,
            creates default from config.json.
        phoenix_endpoint: Optional Phoenix HTTP URL (for testing). If None,
            reads from SystemConfig.telemetry_url.
    """
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    if config_manager is None:
        from cogniverse_foundation.config.utils import create_default_config_manager

        config_manager = create_default_config_manager()

    telemetry_manager = get_telemetry_manager(otlp_endpoint=telemetry_otlp_endpoint)
    telemetry_provider = telemetry_manager.get_provider(tenant_id=tenant_id)

    # Load trigger dataset from Phoenix
    from phoenix.client import Client as PhoenixSyncClient

    system_config = config_manager.get_system_config()
    if phoenix_endpoint is None:
        phoenix_endpoint = system_config.telemetry_url
    sync_client = PhoenixSyncClient(base_url=phoenix_endpoint)

    try:
        dataset = sync_client.datasets.get_dataset(dataset=trigger_dataset)
        trigger_df = dataset.to_dataframe()

        # Phoenix wraps columns under input/output dicts — flatten
        if "input" in trigger_df.columns and "agent" not in trigger_df.columns:
            import pandas as _pd

            flat = []
            for _, row in trigger_df.iterrows():
                inp = row.get("input", {}) or {}
                out = row.get("output", {}) or {}
                flat.append({**inp, **out})
            trigger_df = _pd.DataFrame(flat)
    except Exception as e:
        logger.error(f"Failed to load trigger dataset '{trigger_dataset}': {e}")
        return {"status": "failed", "error": str(e)}

    results = {}
    from cogniverse_foundation.config.utils import get_config

    config_utils = get_config(tenant_id=tenant_id, config_manager=config_manager)
    llm_config = config_utils.get_llm_config()
    llm_endpoint = llm_config.resolve("optimization")

    checkpointer = await _TriggeredOptCheckpointer.maybe_resume(
        config_manager=config_manager,
        telemetry_manager=telemetry_manager,
        tenant_id=tenant_id,
        agents=agents,
        trigger_dataset=trigger_dataset,
        phoenix_endpoint=phoenix_endpoint,
    )

    for agent_name in agents:
        if checkpointer is not None and agent_name in checkpointer.done_agents():
            results[agent_name] = checkpointer.stored_result(agent_name)
            logger.info(
                "Resuming: agent '%s' already optimized, skipping recompile",
                agent_name,
            )
            continue

        agent_df = trigger_df[trigger_df["agent"] == agent_name]
        if agent_df.empty:
            logger.info(f"No training data for agent '{agent_name}', skipping")
            results[agent_name] = {"status": "skipped", "reason": "no_data"}
            continue

        low_scoring = agent_df[agent_df["category"] == "low_scoring"]
        high_scoring = agent_df[agent_df["category"] == "high_scoring"]

        logger.info(
            f"Optimizing {agent_name}: "
            f"{len(low_scoring)} negative, {len(high_scoring)} positive examples"
        )

        try:
            result = await _optimize_agent(
                agent_name=agent_name,
                low_scoring_df=low_scoring,
                high_scoring_df=high_scoring,
                llm_endpoint=llm_endpoint,
                config_manager=config_manager,
                telemetry_provider=telemetry_provider,
                tenant_id=tenant_id,
                teacher_endpoint=require_reachable_teacher(llm_config),
            )
            results[agent_name] = result
            if checkpointer is not None:
                await checkpointer.record(agent_name, result)
        except Exception as e:
            logger.error(f"Optimization failed for {agent_name}: {e}")
            results[agent_name] = {"status": "failed", "error": str(e)}

    # Strategy distillation: learn reusable strategies from the trigger dataset
    try:
        from cogniverse_agents.optimizer.strategy_learner import StrategyLearner
        from cogniverse_core.memory.manager import Mem0MemoryManager

        if not system_config.backend_url:
            raise ValueError(
                "SystemConfig.backend_url is required for strategy distillation"
            )
        if not system_config.backend_port:
            raise ValueError(
                "SystemConfig.backend_port is required for strategy distillation"
            )
        if not llm_endpoint.api_base:
            raise ValueError(
                "LLMEndpointConfig.api_base is required for strategy distillation"
            )
        denseon_url = system_config.inference_service_urls.get("denseon")
        if not denseon_url:
            raise ValueError(
                "Mem0 strategy distillation requires the denseon inference "
                "service. Available: "
                f"{sorted(system_config.inference_service_urls)}"
            )

        mem_manager = Mem0MemoryManager(tenant_id=tenant_id)
        if mem_manager.memory is None:
            mem_manager.initialize(
                backend_host=system_config.backend_url,
                backend_port=system_config.backend_port,
                llm_model=llm_endpoint.model,
                embedding_model="lightonai/DenseOn",
                llm_base_url=llm_endpoint.api_base,
                embedder_base_url=denseon_url,
                config_manager=config_manager,
                schema_loader=None,
            )

        learner = StrategyLearner(
            memory_manager=mem_manager,
            tenant_id=tenant_id,
            llm_config=llm_endpoint,
        )
        strategies = await learner.learn_from_trigger_dataset(trigger_df)
        results["strategies_distilled"] = len(strategies)
        logger.info(f"Distilled {len(strategies)} strategies from trigger dataset")
    except Exception as e:
        logger.warning(f"Strategy distillation failed (non-fatal): {e}")
        results["strategies_distilled"] = 0

    # Post-optimization: run golden eval to verify improvement (best-effort)
    try:
        from cogniverse_evaluation.quality_monitor import QualityMonitor

        if not llm_endpoint.api_base:
            raise ValueError("LLMEndpointConfig.api_base required for post-eval")

        monitor = QualityMonitor(
            tenant_id=tenant_id,
            runtime_url=system_config.agent_registry_url,
            phoenix_http_endpoint=phoenix_endpoint,
            llm_base_url=llm_endpoint.api_base,
            llm_model=llm_endpoint.model,
            golden_dataset_path="data/testset/evaluation/sample_videos_retrieval_queries.json",
        )
        post_eval = await monitor.evaluate_golden_set()
        results["post_optimization_eval"] = {
            "mrr": post_eval.mean_mrr,
            "ndcg": post_eval.mean_ndcg,
            "precision_at_5": post_eval.mean_precision_at_5,
        }

        # baseline_mrr is the prior baseline snapshotted BEFORE the run was
        # stored — reading the store here would compare the run to itself.
        if post_eval.mean_mrr > (post_eval.baseline_mrr or 0):
            await monitor.update_baseline(golden_result=post_eval)
            results["baseline_updated"] = True

        # Grow golden set with high-scoring live queries
        new_golden_candidates = _golden_set_candidates(trigger_df)
        if new_golden_candidates:
            await monitor.grow_golden_set(new_golden_candidates)
            results["golden_set_growth"] = len(new_golden_candidates)

        await monitor.close()
    except Exception as e:
        logger.warning(f"Post-optimization eval failed (non-fatal): {e}")
        results["post_optimization_eval"] = {"error": str(e)}

    if checkpointer is not None:
        had_failure = any(
            isinstance(results.get(a), dict) and results[a].get("status") == "failed"
            for a in agents
        )
        await checkpointer.finalize(had_failure)

    return results


def _golden_set_candidates(trigger_df) -> list:
    """High-scoring live queries eligible for golden-set growth."""
    candidates = []
    high_scoring = trigger_df[trigger_df["category"] == "high_scoring"]
    for _, row in high_scoring.iterrows():
        try:
            score = float(row.get("score", 0))
        except (TypeError, ValueError):
            logger.warning(
                "Skipping golden-set candidate with non-numeric score %r",
                row.get("score"),
            )
            continue
        if score >= 0.8:
            candidates.append(
                {
                    "query": row.get("query", ""),
                    "expected_videos": [],
                    "ground_truth": "",
                    "query_type": "live_traffic",
                    "source": "quality_monitor",
                }
            )
    return candidates


async def _optimize_agent(
    agent_name: str,
    low_scoring_df,
    high_scoring_df,
    llm_endpoint,
    config_manager,
    telemetry_provider,
    tenant_id: str,
    teacher_endpoint=None,
) -> dict:
    """Run DSPy optimization for a specific agent using scored examples.

    The compile trains on the high-scoring rows; a held-out slice of them
    plus the low-scoring rows (as known-bad probes) score the candidate
    against the currently-active baseline, and only a win by at least
    ``optimization_improvement_threshold`` is promoted to serving.
    """
    import json as _json

    if agent_name not in _SERVE_TARGET:
        return {"status": "skipped", "reason": f"no_signature_for_{agent_name}"}

    # Build labeled examples from the positives FIRST — an all-failure agent
    # has nothing to bootstrap from and must say so, including how many
    # negatives were on the table, before any LM setup.
    import dspy

    trainset = []
    for _, row in high_scoring_df.iterrows():
        query = row.get("query", "")
        output = row.get("output", "{}")
        if isinstance(output, str):
            try:
                output = _json.loads(output)
            except Exception:
                output = {}
        if not isinstance(output, dict):
            output = {}

        if agent_name == "search":
            example = dspy.Example(
                query=query,
                modality="video",
                top_k=10,
                search_strategy="colpali",
                enhanced_query=str(output.get("enhanced_query") or query),
                confidence=row.get("score", 0.8),
            ).with_inputs("query", "modality", "top_k")
        elif agent_name == "summary":
            example = dspy.Example(
                content=_json.dumps(output, default=str),
                summary_type="comprehensive",
                target_audience="general",
                summary=output.get("summary", ""),
                key_points=str(output.get("key_points", [])),
                confidence=row.get("score", 0.8),
            ).with_inputs("content", "summary_type", "target_audience")
        elif agent_name == "report":
            example = dspy.Example(
                search_results=_json.dumps(output, default=str),
                query_context=query,
                analysis_depth="detailed",
                executive_summary=output.get("executive_summary", ""),
                detailed_findings=output.get("detailed_findings", ""),
                recommendations=output.get("recommendations", ""),
                technical_details=output.get("technical_details", ""),
                confidence=row.get("score", 0.8),
            ).with_inputs("search_results", "query_context", "analysis_depth")
        else:
            continue

        trainset.append(example)

    if not trainset:
        return await _reflect_or_skip(
            agent_name=agent_name,
            low_scoring_df=low_scoring_df,
            llm_endpoint=llm_endpoint,
            config_manager=config_manager,
            telemetry_provider=telemetry_provider,
            tenant_id=tenant_id,
            teacher_endpoint=teacher_endpoint,
        )

    from cogniverse_agents.optimizer.dspy_agent_optimizer import (
        DSPyAgentPromptOptimizer,
    )

    optimizer = DSPyAgentPromptOptimizer()
    optimizer.initialize_language_model(
        llm_endpoint, teacher_endpoint_config=teacher_endpoint
    )

    signature = _signature_for_agent(optimizer, agent_name)

    train, holdout = _split_train_holdout(trainset)
    negatives = _negative_probes(agent_name, low_scoring_df)

    from dspy.teleprompt import BootstrapFewShot

    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

    teleprompter = BootstrapFewShot(
        max_bootstrapped_demos=optimizer.optimization_settings[
            "max_bootstrapped_demos"
        ],
        max_labeled_demos=optimizer.optimization_settings["max_labeled_demos"],
        max_rounds=optimizer.optimization_settings["max_rounds"],
        max_errors=optimizer.optimization_settings["max_errors"],
        teacher_settings=optimizer.optimization_settings["teacher_settings"],
    )

    module = dspy.ChainOfThought(signature)

    try:
        # initialize_language_model only sets optimizer.lm, so the compile gets
        # it task-locally — the same binding every other mode uses.
        with dspy.context(lm=optimizer.lm):
            compiled = teleprompter.compile(module, trainset=train)

        artifact_manager = ArtifactManager(telemetry_provider, tenant_id)
        return await _score_and_serve(
            artifact_manager,
            agent_name,
            signature,
            compiled,
            holdout,
            negatives,
            optimizer.lm,
            tenant_id,
            config_manager,
            len(train),
        )

    except Exception as e:
        logger.error(f"DSPy compilation failed for {agent_name}: {e}")
        return {"status": "failed", "error": str(e)}


def _signature_for_agent(optimizer, agent_name: str):
    """The DSPy signature the compile trains for a servable agent."""
    if agent_name == "search":
        return optimizer.create_query_analysis_signature()
    if agent_name == "summary":
        return optimizer.create_summary_generation_signature()
    return optimizer.create_detailed_report_signature()


async def _score_and_serve(
    artifact_manager,
    agent_name: str,
    signature,
    compiled,
    holdout: list,
    negatives: list,
    optimizer_lm,
    tenant_id: str,
    config_manager,
    train_examples: int,
    extra_result: Optional[dict] = None,
) -> dict:
    """Score a compiled candidate against the active baseline and serve a winner.

    Loads the currently-active instructions into a baseline module, scores
    baseline vs candidate on the same ``(holdout, negatives)`` probe set, and
    routes the candidate through ``_serve_compiled_prompts`` — promoted only
    when it beats ``baseline + optimization_improvement_threshold``. Shared by
    the positives-trained path and the reflective all-failure path;
    ``extra_result`` is merged into the returned dict (e.g. ``reflective``).
    """
    import dspy

    served_agent, predictor_attr = _SERVE_TARGET[agent_name]

    # Baseline = the currently-active instructions (or the stock signature when
    # the agent was never optimized), scored against the candidate on the
    # held-out positives + known-bad probes.
    baseline_score = candidate_score = None
    if holdout or negatives:
        baseline_module = dspy.ChainOfThought(signature)
        active_prompts = await artifact_manager.load_prompts(served_agent)
        if active_prompts and active_prompts.get(predictor_attr):
            for _, predictor in baseline_module.named_predictors():
                predictor.signature = predictor.signature.with_instructions(
                    active_prompts[predictor_attr]
                )
                break
        with dspy.context(lm=optimizer_lm):
            baseline_score, candidate_score = _holdout_scores(
                baseline_module, compiled, holdout, negatives, agent_name
            )

    min_improvement = _min_improvement_from_config(tenant_id, config_manager)
    served = await _serve_compiled_prompts(
        artifact_manager,
        agent_name,
        compiled,
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        min_improvement=min_improvement,
        train_examples=train_examples,
    )

    result = {
        "status": "success",
        "training_examples": train_examples,
        "holdout_examples": len(holdout),
        "negative_probes": len(negatives),
    }
    if extra_result:
        result.update(extra_result)
    if served:
        result["served"] = served
    return result


def _min_improvement_from_config(tenant_id: str, config_manager=None) -> float:
    """The tenant's ``optimization_improvement_threshold`` acceptance gate."""
    from cogniverse_runtime.quality_monitor_cli import _load_automation_rules

    rules = _load_automation_rules(tenant_id, config_manager=config_manager)
    return float(rules.optimization_triggers.optimization_improvement_threshold)


def _population_floor_from_config(
    tenant_id: str,
    config_manager=None,
    optimizer_type: str = "query_enhancement",
) -> tuple[int, int]:
    """The tenant's promotion floor: (min_samples, min_unique_queries)."""
    from cogniverse_foundation.config.utils import create_default_config_manager

    manager = config_manager or create_default_config_manager()
    routing = manager.get_routing_config(tenant_id=tenant_id)
    optimizer_floor = routing.optimizer_floors.get(optimizer_type)
    if optimizer_floor is not None:
        tenant_floor = None
        if isinstance(optimizer_floor, dict):
            tenant_floor = _population_floor_from_floor_config(
                optimizer_floor,
                routing.min_samples_for_optimization,
                routing.min_unique_queries,
            )
        if tenant_floor is not None:
            return tenant_floor
        logger.warning(
            "Ignoring malformed optimizer floor for tenant %r optimizer %r: %r",
            tenant_id,
            optimizer_type,
            optimizer_floor,
        )

    shipped_optimizer_floor = _shipped_population_floor_from_config(optimizer_type)
    if shipped_optimizer_floor is not None:
        return shipped_optimizer_floor

    return (
        int(routing.min_samples_for_optimization),
        int(routing.min_unique_queries),
    )


def _population_floor_from_floor_config(
    floor_config: dict,
    fallback_min_samples: int,
    fallback_min_unique: int,
) -> tuple[int, int] | None:
    """Normalize a population-floor mapping into the promoted threshold pair."""
    try:
        return (
            int(floor_config.get("min_samples_for_optimization", fallback_min_samples)),
            int(floor_config.get("min_unique_queries", fallback_min_unique)),
        )
    except (TypeError, ValueError):
        return None


def _shipped_population_floor_from_config(
    optimizer_type: str,
) -> tuple[int, int] | None:
    """Load the shipped optimizer floor for ``optimizer_type`` if present."""
    try:
        raw_config = json.loads(SHIPPED_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Unable to read shipped optimization floors from %s: %s",
            SHIPPED_CONFIG_PATH,
            exc,
        )
        return None

    routing_config = raw_config.get("routing")
    if not isinstance(routing_config, dict):
        logger.warning(
            "Shipped optimization floors at %s are missing routing.optimization_config",
            SHIPPED_CONFIG_PATH,
        )
        return None

    optimization_config = routing_config.get("optimization_config")
    if not isinstance(optimization_config, dict):
        logger.warning(
            "Shipped optimization floors at %s are missing routing.optimization_config",
            SHIPPED_CONFIG_PATH,
        )
        return None

    optimizer_floors = optimization_config.get("optimizer_floors")
    if not isinstance(optimizer_floors, dict):
        logger.warning(
            "Shipped optimization floors at %s are missing routing.optimization_config.optimizer_floors",
            SHIPPED_CONFIG_PATH,
        )
        return None

    shipped_floor = optimizer_floors.get(optimizer_type)
    if shipped_floor is None:
        return None
    if not isinstance(shipped_floor, dict):
        logger.warning(
            "Shipped optimization floor for %s at %s is malformed",
            optimizer_type,
            SHIPPED_CONFIG_PATH,
        )
        return None

    return _population_floor_from_floor_config(
        shipped_floor,
        optimization_config.get(
            "min_samples_for_optimization",
            100,
        ),
        optimization_config.get("min_unique_queries", 3),
    )


def _training_selection_from_config(
    config_manager,
    tenant_id: str,
    optimizer_type: str,
) -> TrainingSelectionKnobs:
    """The tenant's training-selection knobs for a given optimizer."""
    from cogniverse_foundation.config.utils import create_default_config_manager

    manager = config_manager or create_default_config_manager()
    routing = manager.get_routing_config(tenant_id=tenant_id)
    resolved: Dict[str, Any] = {}

    tenant_training_selection = routing.training_selection.get(optimizer_type)
    if tenant_training_selection is not None:
        if isinstance(tenant_training_selection, dict):
            tenant_values, tenant_malformed = _training_selection_values_from_config(
                tenant_training_selection
            )
            resolved.update(tenant_values)
            if tenant_malformed:
                logger.warning(
                    "tenant=%r optimizer=%r has malformed training_selection entry: %r",
                    tenant_id,
                    optimizer_type,
                    tenant_training_selection,
                )
        else:
            logger.warning(
                "tenant=%r optimizer=%r has malformed training_selection entry: %r",
                tenant_id,
                optimizer_type,
                tenant_training_selection,
            )

    shipped_training_selection = _shipped_training_selection_from_config(optimizer_type)
    if shipped_training_selection is not None:
        shipped_values, _ = _training_selection_values_from_config(
            shipped_training_selection
        )
        for field_name, value in shipped_values.items():
            resolved.setdefault(field_name, value)

    defaults = _TRAINING_SELECTION_DEFAULTS
    return TrainingSelectionKnobs(
        int(resolved.get("trainset_cap", defaults.trainset_cap)),
        float(resolved.get("mmr_lambda", defaults.mmr_lambda)),
        int(
            resolved.get(
                "low_confirmation_threshold", defaults.low_confirmation_threshold
            )
        ),
        int(resolved.get("downweight_age_days", defaults.downweight_age_days)),
        float(resolved.get("downweight_factor", defaults.downweight_factor)),
        resolved.get(
            "confirmation_score_threshold",
            defaults.confirmation_score_threshold,
        ),
    )


def _training_selection_values_from_config(
    training_selection_config: dict,
) -> tuple[Dict[str, Any], bool]:
    """Normalize a training-selection mapping into typed knob values."""
    resolved: Dict[str, Any] = {}
    malformed = False
    for field_name, converter in _TRAINING_SELECTION_FIELDS:
        if field_name not in training_selection_config:
            continue
        try:
            resolved[field_name] = converter(training_selection_config[field_name])
        except (TypeError, ValueError):
            malformed = True
    return resolved, malformed


def _shipped_training_selection_from_config(
    optimizer_type: str,
) -> dict | None:
    """Load the shipped training-selection mapping for ``optimizer_type``."""
    try:
        raw_config = json.loads(SHIPPED_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Unable to read shipped training selection from %s: %s",
            SHIPPED_CONFIG_PATH,
            exc,
        )
        return None

    routing_config = raw_config.get("routing")
    if not isinstance(routing_config, dict):
        logger.warning(
            "Shipped training selection at %s are missing routing.optimization_config",
            SHIPPED_CONFIG_PATH,
        )
        return None

    optimization_config = routing_config.get("optimization_config")
    if not isinstance(optimization_config, dict):
        logger.warning(
            "Shipped training selection at %s are missing routing.optimization_config",
            SHIPPED_CONFIG_PATH,
        )
        return None

    training_selection = optimization_config.get("training_selection")
    if not isinstance(training_selection, dict):
        logger.warning(
            "Shipped training selection at %s are missing routing.optimization_config.training_selection",
            SHIPPED_CONFIG_PATH,
        )
        return None

    shipped_training_selection = training_selection.get(optimizer_type)
    if shipped_training_selection is None:
        return None
    if not isinstance(shipped_training_selection, dict):
        logger.warning(
            "Shipped training selection for %s at %s is malformed",
            optimizer_type,
            SHIPPED_CONFIG_PATH,
        )
        return None

    return shipped_training_selection


def _selection_summary(selection_report: SelectionReport) -> Dict[str, Any]:
    return {
        "selection": {
            "pool": selection_report.pool,
            "deduped": selection_report.deduped,
            "cap": selection_report.cap,
            "mmr_applied": selection_report.mmr_applied,
            "decayed_count": selection_report.decayed_count,
            "decayed_example_ids": selection_report.decayed_example_ids,
        }
    }


async def _apply_training_selection(
    *,
    artifact_manager,
    config_manager,
    tenant_id: str,
    optimizer_type: str,
    artifact_key: str,
    train_records: List[Dict[str, Any]],
    embedder_url: Optional[str],
) -> tuple[List[Dict[str, Any]], SelectionReport]:
    lineage = await artifact_manager.get_version_lineage("model", artifact_key)
    knobs = _training_selection_from_config(config_manager, tenant_id, optimizer_type)
    stats = confirmation_stats(
        lineage, score_threshold=knobs.confirmation_score_threshold
    )
    pool = len(train_records)
    if embedder_url is None and pool > knobs.trainset_cap:
        raise RuntimeError(
            "training selection requires --embedder-url when the pool exceeds "
            f"trainset_cap (pool={pool} cap={knobs.trainset_cap})"
        )

    weights = {
        record["example_id"]: decay_weight(
            stats,
            record["example_id"],
            now=datetime.now(timezone.utc),
            knobs=knobs,
        )
        for record in train_records
    }
    selected_records, selection_report = select_training_records(
        train_records,
        weights=weights,
        knobs=knobs,
        embed_fn=lambda texts: embed_texts(embedder_url, texts),
    )
    return selected_records, selection_report


def _reflective_settings_from_config(tenant_id: str, config_manager=None):
    """The tenant's reflective-recompile toggles: (enable, min_failures, budget)."""
    from cogniverse_runtime.quality_monitor_cli import _load_automation_rules

    triggers = _load_automation_rules(
        tenant_id, config_manager=config_manager
    ).optimization_triggers
    return (
        bool(triggers.enable_reflective_recompile),
        int(triggers.min_reflective_failures),
        int(triggers.reflective_max_metric_calls),
    )


async def _reflect_or_skip(
    *,
    agent_name: str,
    low_scoring_df,
    llm_endpoint,
    config_manager,
    telemetry_provider,
    tenant_id: str,
    teacher_endpoint=None,
) -> dict:
    """All-failure fallback: reflective GEPA recompile, or the skip dict.

    Reached when the positives trainset is empty. With reflective recompile
    enabled and enough failing rows, the failing rows are split into a GEPA
    trainset and a held-out negatives slice, GEPA proposes improved
    instructions from the feedback metric, and the candidate goes through the
    SAME ``_score_and_serve`` promotion gate (scored on the held-out negatives,
    no positives). Otherwise the original skip dict is returned unchanged.
    """
    enable, min_failures, max_metric_calls = _reflective_settings_from_config(
        tenant_id, config_manager
    )
    if not enable or agent_name not in _SERVE_TARGET:
        return {
            "status": "skipped",
            "reason": "no_positive_examples",
            "negative_examples": int(len(low_scoring_df)),
        }

    reflect_train_rows, reflect_eval_rows = _split_train_holdout(
        list(low_scoring_df.to_dict("records"))
    )
    if len(reflect_train_rows) < min_failures:
        return {
            "status": "skipped",
            "reason": "insufficient_failures_to_reflect",
            "negative_examples": int(len(low_scoring_df)),
        }

    import pandas as _pd

    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
    from cogniverse_agents.optimizer.dspy_agent_optimizer import (
        DSPyAgentPromptOptimizer,
    )

    optimizer = DSPyAgentPromptOptimizer()
    optimizer.initialize_language_model(
        llm_endpoint, teacher_endpoint_config=teacher_endpoint
    )
    signature = _signature_for_agent(optimizer, agent_name)

    # Eval negatives are HELD OUT — GEPA compiles on reflect_train_rows only and
    # never sees reflect_eval_rows, so the promotion gate scores against unseen
    # failures.
    eval_negatives = _negative_probes(agent_name, _pd.DataFrame(reflect_eval_rows))

    try:
        compiled = _reflective_compile(
            agent_name,
            reflect_train_rows,
            signature,
            optimizer.lm,
            max_metric_calls,
        )
        artifact_manager = ArtifactManager(telemetry_provider, tenant_id)
        return await _score_and_serve(
            artifact_manager,
            agent_name,
            signature,
            compiled,
            [],
            eval_negatives,
            optimizer.lm,
            tenant_id,
            config_manager,
            len(reflect_train_rows),
            extra_result={"reflective": True},
        )
    except Exception as e:
        logger.error(f"Reflective compilation failed for {agent_name}: {e}")
        return {"status": "failed", "error": str(e)}


def _build_gepa(metric, reflection_lm, max_metric_calls: int):
    """Construct the GEPA reflective compiler (isolated seam for testing)."""
    from dspy.teleprompt import GEPA

    return GEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=3,
        candidate_selection_strategy="pareto",
    )


def _reflective_metric(agent_name: str):
    """A 5-arg GEPA feedback metric rewarding a candidate for NOT reproducing
    the recorded failing output.

    ``search`` has no free-text label, so it is scored on enum validity; the
    text agents score ``1 - token_f1`` against the recorded failing output.
    """
    from dspy.teleprompt.gepa.gepa import ScoreWithFeedback

    field = _EVAL_FIELD.get(agent_name)

    def metric(gold, pred, trace, pred_name, pred_trace):
        if agent_name == "search":
            score = _search_validity(pred)
            feedback = (
                f"The recorded analysis for query {getattr(gold, 'query', '')!r} "
                "was malformed. A valid analysis sets primary_intent, "
                "complexity_level, and needs_video_search to well-formed enum "
                "values."
            )
        else:
            bad = str(getattr(gold, "_bad_output", "") or "")
            produced = str(getattr(pred, field, "") or "")
            score = 1.0 - _token_f1(produced, bad)
            feedback = (
                f"The recorded failing {field} was {bad!r}. Produce a distinct, "
                f"accurate {field} that does not reproduce it."
            )
        return ScoreWithFeedback(score=float(score), feedback=feedback)

    return metric


def _reflective_compile(
    agent_name: str,
    reflect_train_rows: list,
    signature,
    reflection_lm,
    max_metric_calls: int,
):
    """Recompile an all-failure agent's prompt with dspy.GEPA.

    No positive demonstrations exist; GEPA's reflection LM reads the failing
    rollouts plus ``_reflective_metric``'s feedback and proposes improved
    instructions. Each failing row becomes a GEPA training example carrying the
    signature's input fields plus a ``_bad_output`` attribute holding the
    recorded failing output the candidate must avoid.
    """
    import dspy

    input_keys = _EVAL_INPUTS[agent_name]
    optional_keys = _EVAL_OPTIONAL_INPUTS[agent_name]
    field = _EVAL_FIELD.get(agent_name)

    trainset = []
    for row in reflect_train_rows:
        query = row.get("query", "")
        output = row.get("output", "{}")
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except Exception:
                output = {}
        if not isinstance(output, dict):
            output = {}

        if agent_name == "search":
            inputs = {"query": query}
            bad_output = ""
        elif agent_name == "summary":
            inputs = {
                "content": json.dumps(output, default=str),
                "summary_type": "comprehensive",
                "target_audience": "general",
            }
            bad_output = str(output.get(field, ""))
        else:
            inputs = {
                "search_results": json.dumps(output, default=str),
                "query_context": query,
                "analysis_depth": "detailed",
            }
            bad_output = str(output.get(field, ""))
        for key in optional_keys:
            inputs.setdefault(key, "")

        example = dspy.Example(**inputs, _bad_output=bad_output).with_inputs(
            *input_keys
        )
        trainset.append(example)

    metric = _reflective_metric(agent_name)
    gepa = _build_gepa(metric, reflection_lm, max_metric_calls)
    module = dspy.ChainOfThought(signature)
    with dspy.context(lm=reflection_lm):
        return gepa.compile(module, trainset=trainset)


# Where a triggered-mode compile is served from: the DISPATCH agent name the
# overlay resolves artefacts under, and the predictor attribute on that
# agent's DSPy module whose instructions the overlay swaps.
_SERVE_TARGET = {
    "search": ("search_agent", "search_optimizer"),
    "summary": ("summarizer_agent", "summarizer"),
    "report": ("detailed_report_agent", "report_generator"),
}

# Held-out eval wiring per agent, matching the REAL DSPy signatures
# (create_query_analysis_signature etc.): required input kwargs taken from
# each example, optional inputs blank-filled, and the primary output field
# scored against the example's label. The search signature emits
# intent/complexity/boolean enums with no free-text labeled output, so search
# is scored label-free on output VALIDITY (are the enums well-formed values).
_EVAL_FIELD = {
    "summary": "summary",
    "report": "executive_summary",
}
_EVAL_INPUTS = {
    "search": ("query",),
    "summary": ("content", "summary_type", "target_audience"),
    "report": ("search_results", "query_context", "analysis_depth"),
}
_EVAL_OPTIONAL_INPUTS = {
    "search": ("context",),
    "summary": ("visual_insights",),
    "report": ("visual_analysis",),
}
_SEARCH_INTENTS = {
    "search",
    "comparison",
    "analysis",
    "summarization",
    "reporting",
    "temporal_search",
    "content_discovery",
    "information_extraction",
    "complex_analysis",
    "meta_query",
}
_SEARCH_COMPLEXITIES = {"simple", "moderate", "complex"}
_BOOL_WORDS = {"true", "false"}


def _search_validity(pred) -> float:
    """Label-free score for the query-analysis signature: fraction of the
    enum-typed outputs that hold a well-formed value."""
    intent = str(getattr(pred, "primary_intent", "") or "").strip().lower()
    complexity = str(getattr(pred, "complexity_level", "") or "").strip().lower()
    needs_video = str(getattr(pred, "needs_video_search", "") or "").strip().lower()
    checks = [
        intent in _SEARCH_INTENTS,
        complexity in _SEARCH_COMPLEXITIES,
        needs_video in _BOOL_WORDS,
    ]
    return sum(checks) / len(checks)


def _probe_score(pred, label: str, agent_name: str) -> float:
    """Score one prediction: search by validity, summary/report by token-F1
    to the label (or plain non-emptiness when the label is empty)."""
    if agent_name == "search":
        return _search_validity(pred)
    text = str(getattr(pred, _EVAL_FIELD[agent_name], "") or "")
    if str(label or "").strip():
        return _token_f1(text, label)
    return 1.0 if text.strip() else 0.0


def _token_f1(predicted: str, label: str) -> float:
    """Whitespace-token set F1 between casefolded predicted and label tokens."""
    pred = set(str(predicted or "").casefold().split())
    lab = set(str(label or "").casefold().split())
    if not pred or not lab:
        return 0.0
    overlap = len(pred & lab)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(lab)
    return 2 * precision * recall / (precision + recall)


def _split_train_holdout(examples: list) -> tuple[list, list]:
    """Deterministic tail holdout: ~25% (min 1) once there are 2+ examples."""
    n = len(examples)
    if n <= 1:
        return list(examples), []
    k = max(1, n // 4)
    return list(examples[:-k]), list(examples[-k:])


def is_scoreable(record: dict) -> bool:
    """True when the served record has any scoreable context."""
    return bool(
        str(record.get("source_text") or "").strip()
        or str(record.get("grounding_context") or "").strip()
    )


def _profile_selection_is_scoreable(record: dict) -> bool:
    """Profile selection records are labeled by ``selected_profile`` itself."""
    return bool(str(record.get("selected_profile") or "").strip())


def _served_scoreable_indices(
    records: list[dict], scoreable_predicate=is_scoreable
) -> list[int]:
    """Indices of served span records that can safely contribute to holdout."""
    return [
        index
        for index, record in enumerate(records)
        if record["example_id"].startswith("span:") and scoreable_predicate(record)
    ]


def _scoreable_first(records: list[dict]) -> tuple[list[dict], int]:
    """Scoreable records first, unscoreable after, each in original order.

    BootstrapFewShot walks the trainset in order and stops once it has its
    demos. A record ``_query_enhancement_quality`` cannot score is a
    guaranteed rejection, so it is walked last and only when the scoreable
    records did not fill the demos. Returns the ordering and the unscoreable
    count.
    """
    scoreable = [record for record in records if is_scoreable(record)]
    unscoreable = [record for record in records if not is_scoreable(record)]
    return scoreable + unscoreable, len(unscoreable)


def _split_served_holdout(
    records: list[dict], min_holdout: int, scoreable_predicate=is_scoreable
) -> tuple[list[dict], list[dict]]:
    """Serve the tail of scoreable span records and keep everything else in train."""
    served_scoreable_indices = _served_scoreable_indices(records, scoreable_predicate)
    if len(served_scoreable_indices) < min_holdout:
        return list(records), []

    holdout_count = max(1, len(served_scoreable_indices) // 4)
    holdout_indices = set(served_scoreable_indices[-holdout_count:])
    train = [
        record for index, record in enumerate(records) if index not in holdout_indices
    ]
    holdout = [
        record for index, record in enumerate(records) if index in holdout_indices
    ]
    return train, holdout


def _negative_probes(agent_name: str, low_scoring_df, limit: int = 20) -> list:
    """Known-bad probes from the human-flagged failures that triggered the
    recompile: ``(inputs, failing_output)`` pairs. For summary/report a
    candidate is rewarded for NOT reproducing the failing output on the same
    inputs; for search (label-free) the candidate is scored on producing a
    VALID analysis where the recorded one failed."""
    import json as _json

    field = _EVAL_FIELD.get(agent_name)
    probes = []
    for _, row in low_scoring_df.iterrows():
        query = row.get("query", "")
        output = row.get("output", "{}")
        if isinstance(output, str):
            try:
                output = _json.loads(output)
            except Exception:
                output = {}
        if not isinstance(output, dict):
            output = {}
        if agent_name == "search":
            if not str(query).strip():
                continue
            probes.append(({"query": query}, ""))
        elif agent_name == "summary":
            bad = str(output.get(field, ""))
            if not bad:
                continue
            probes.append(
                (
                    {
                        "content": _json.dumps(output, default=str),
                        "summary_type": "comprehensive",
                        "target_audience": "general",
                    },
                    bad,
                )
            )
        else:
            bad = str(output.get(field, ""))
            if not bad:
                continue
            probes.append(
                (
                    {
                        "search_results": _json.dumps(output, default=str),
                        "query_context": query,
                        "analysis_depth": "detailed",
                    },
                    bad,
                )
            )
        if len(probes) >= limit:
            break
    return probes


def _holdout_scores(
    baseline_module, candidate_module, holdout, negatives, agent_name: str
) -> tuple[float, float]:
    """Score both modules on the same probe set.

    Held-out positives contribute ``_probe_score`` against the labeled
    output (validity for search); summary/report negatives contribute
    ``1 - F1`` against the recorded failing output, search negatives the
    validity of the fresh analysis. Returns
    ``(baseline_score, candidate_score)`` as means over the probe set.
    """
    input_keys = _EVAL_INPUTS[agent_name]
    optional_keys = _EVAL_OPTIONAL_INPUTS[agent_name]
    label_field = _EVAL_FIELD.get(agent_name)

    def _kwargs(base: dict) -> dict:
        kwargs = dict(base)
        for k in optional_keys:
            kwargs.setdefault(k, "")
        return kwargs

    def _run(module) -> list:
        scores = []
        for ex in holdout:
            pred = module(**_kwargs({k: getattr(ex, k) for k in input_keys}))
            label = getattr(ex, label_field) if label_field else ""
            scores.append(_probe_score(pred, label, agent_name))
        for inputs, bad_output in negatives:
            pred = module(**_kwargs(inputs))
            if agent_name == "search":
                scores.append(_search_validity(pred))
            else:
                field = _EVAL_FIELD[agent_name]
                scores.append(1.0 - _token_f1(getattr(pred, field, ""), bad_output))
        return scores

    baseline_scores = _run(baseline_module)
    candidate_scores = _run(candidate_module)
    if not baseline_scores:
        return 0.0, 0.0
    return (
        sum(baseline_scores) / len(baseline_scores),
        sum(candidate_scores) / len(candidate_scores),
    )


async def _serve_compiled_prompts(
    artifact_manager,
    agent_name: str,
    compiled,
    *,
    baseline_score: Optional[float] = None,
    candidate_score: Optional[float] = None,
    min_improvement: float = 0.0,
    train_examples: Optional[int] = None,
):
    """Publish a compiled module's instructions IF it beats the active baseline.

    Serving goes through ``ArtifactManager.promote_if_better``: only a
    candidate that scores at least ``baseline + min_improvement`` on the
    held-out eval flips active (versioned save → canary → active, so the
    per-request overlay serves it on the next dispatch). A losing candidate
    is recorded in the experiments ledger and never touches live traffic;
    ``--mode rollback`` still restores prior versions.

    Without eval scores nothing is promoted — an ungated promote can regress
    live traffic, which is exactly what the gate exists to prevent.

    Returns ``None`` when the compile produced no instructions, otherwise a
    dict with ``served_agent``/``version``/``active``/``promoted`` plus the
    scores (or a ``reason`` when no eval material was available).
    """
    target = _SERVE_TARGET.get(agent_name)
    if target is None:
        return None
    served_agent, predictor_attr = target

    instructions = None
    named = getattr(compiled, "named_predictors", None)
    for _, predictor in named() if callable(named) else []:
        candidate = getattr(getattr(predictor, "signature", None), "instructions", None)
        if candidate:
            instructions = str(candidate)
            break
    if not instructions:
        logger.warning(
            "Compiled %s module has no instructions — nothing to serve", agent_name
        )
        return None

    if baseline_score is None or candidate_score is None:
        logger.warning(
            "No held-out eval material for %s — compiled prompts NOT promoted",
            agent_name,
        )
        return {
            "served_agent": served_agent,
            "version": None,
            "active": False,
            "promoted": False,
            "reason": "no_eval_material",
        }

    record = await artifact_manager.promote_if_better(
        agent_type=served_agent,
        candidate_prompts={predictor_attr: instructions},
        candidate_demos=None,
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        min_improvement=min_improvement,
        serve_versioned=True,
        optimizer="BootstrapFewShot",
        train_examples=train_examples,
    )
    promoted = bool(record.promoted)
    version = record.extra_metrics.get("served_version") if promoted else None
    logger.info(
        "Compiled %s prompts %s (candidate=%.4f baseline=%.4f min_improvement=%.4f)%s",
        agent_name,
        "PROMOTED" if promoted else "rejected",
        candidate_score,
        baseline_score,
        min_improvement,
        f" — serving as {served_agent} v{version} (active)" if promoted else "",
    )
    return {
        "served_agent": served_agent,
        "version": version,
        "active": promoted,
        "promoted": promoted,
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
    }


def _prune_aged_files(root: str, *, older_than_days: float) -> dict:
    """Delete files under ``root`` whose mtime is older than the cutoff.

    Returns a dict ``{"scanned": N, "deleted": M, "errors": [..]}`` so
    the workflow log captures exact numbers — the assertion contract
    for the daily-cleanup e2e test depends on tight outcome reporting,
    not opaque ``cleanup completed`` markers.

    Silent no-op when ``root`` does not exist or is not a directory —
    the cron container may run on a pod that doesn't mount that path
    (e.g. ``/logs`` only exists when the runtime container mounts a
    log PVC). Logged at INFO so the workflow run records "skipped: no
    such path".
    """
    import time as _t
    from pathlib import Path as _Path

    summary: dict = {"path": root, "scanned": 0, "deleted": 0, "errors": []}
    p = _Path(root)
    if not p.is_dir():
        summary["skipped"] = f"path {root!r} is not a directory"
        return summary

    cutoff = _t.time() - older_than_days * 86400
    for entry in p.rglob("*"):
        if not entry.is_file():
            continue
        summary["scanned"] += 1
        try:
            if entry.stat().st_mtime < cutoff:
                entry.unlink()
                summary["deleted"] += 1
        except OSError as exc:
            summary["errors"].append(f"{entry}: {exc}")
    return summary


def _vacuum_config_metadata(*, keep_versions: int) -> dict:
    """Drain config_metadata version bloat across every config_id.

    Per-write pruning in ``VespaConfigStore.set_config`` keeps fresh
    writes bounded, but a backlog can accumulate when ``keep_versions``
    is bumped or when a backend write path skipped the prune (e.g. an
    older runtime image). One-off sweep here brings stale rows down
    to ``keep_versions`` per config_id and returns the count dropped
    so the workflow log proves the work happened.
    """
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_vespa.config.config_store import VespaConfigStore

    cm = create_default_config_manager()
    store = cm.store
    if not isinstance(store, VespaConfigStore):
        return {
            "skipped": f"store is {type(store).__name__}, expected VespaConfigStore"
        }

    dropped = store.prune_all_configs(keep=keep_versions)
    return {"dropped": dropped, "keep_versions": keep_versions}


async def run_cleanup(
    tenant_id: Optional[str],
    log_retention_days: int,
    memory_retention_days: int,
) -> dict:
    """Daily-cleanup workflow body: memory + logs + temp + config vacuum.

    Per-tenant Mem0 cleanup is schema-driven (per-kind TTLs in the
    KnowledgeRegistry). The other three steps absorbed the
    standalone ``daily-cleanup`` CronWorkflow that the chart didn't
    previously cover:

      * Log rotation under ``LOG_DIR`` (default ``/logs``) — files
        older than ``log_retention_days`` are removed.
      * Temp file cleanup under ``TEMP_DIR`` (default ``/tmp``) —
        files older than 1 day are removed.
      * config_metadata version vacuum — each config_id is pruned to
        the latest ``CONFIG_KEEP_VERSIONS`` (default 10).

    Each section reports exact counts in the result dict so the
    workflow run log proves the work landed — bare "Succeeded" is too
    weak a signal for a maintenance cron.
    """
    from pathlib import Path

    from cogniverse_core.memory.manager import Mem0MemoryManager
    from cogniverse_core.memory.schema import build_default_registry
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_runtime.admin import tenant_manager
    from cogniverse_runtime.memory_init import lazy_init_memory

    # tenant_manager.get_backend() refuses to initialise without a
    # SchemaLoader injected up-front. The daily-cleanup CronWorkflow
    # runs as a standalone process (not via the runtime FastAPI app),
    # so it has no app-startup lifespan to call set_schema_loader for
    # it. Wire it here using the same FilesystemSchemaLoader pattern
    # the synthetic mode uses.
    schemas_dir = Path(os.environ.get("COGNIVERSE_SCHEMAS_DIR", "configs/schemas"))
    tenant_manager.set_schema_loader(FilesystemSchemaLoader(schemas_dir))

    # cleanup_with_schema requires a fully-initialised Mem0 instance
    # (it touches mgr.memory.get_all). The Mem0MemoryManager singleton
    # cache returns a bare object on first construction — without
    # lazy_init_memory every tenant returns "Mem0MemoryManager not
    # initialized" and the workflow appears to Succeed while silently
    # processing nothing. Build a config_manager once and reuse for
    # every tenant in the sweep.
    config_manager = create_default_config_manager()
    registry = build_default_registry()

    results: Dict[str, Any] = {
        "log_retention_days": log_retention_days,
        "memory_retention_days": memory_retention_days,
    }

    def _cleanup_one(tid: str) -> str:
        try:
            mm = Mem0MemoryManager(tenant_id=tid)
            if not lazy_init_memory(mm, tid, config_manager):
                # An init failure means this tenant's cleanup did NOT run —
                # a "failed:" marker so _run_failed trips the exit code rather
                # than a "skipped:" that reads as an intentional no-op.
                return "failed: memory backend init failed (see workflow log)"
            deleted_by_kind = mm.cleanup_with_schema(registry)
            return f"completed: {dict(deleted_by_kind)}"
        except Exception as e:
            return f"failed: {e}"

    # --- Memory cleanup (per tenant) ---
    if tenant_id is not None:
        results["memory_cleanup"] = {tenant_id: _cleanup_one(tenant_id)}
    else:
        per_tenant: Dict[str, str] = {}
        org_ids = await tenant_manager.list_organizations_internal()
        for org_id in org_ids:
            for tenant in await tenant_manager.list_tenants_for_org_internal(org_id):
                tid = tenant.tenant_full_id
                if not tid:
                    continue
                per_tenant[tid] = _cleanup_one(tid)
        results["memory_cleanup"] = per_tenant
        results["tenants_processed"] = len(per_tenant)

    # --- Log rotation ---
    log_dir = os.environ.get("LOG_DIR", "/logs")
    results["log_cleanup"] = _prune_aged_files(
        log_dir, older_than_days=float(log_retention_days)
    )

    # --- Temp file cleanup ---
    temp_dir = os.environ.get("TEMP_DIR", "/tmp")
    temp_age_days = float(os.environ.get("TEMP_RETENTION_DAYS", "1"))
    results["temp_cleanup"] = _prune_aged_files(temp_dir, older_than_days=temp_age_days)

    # --- Config metadata vacuum ---
    keep_versions = int(os.environ.get("CONFIG_KEEP_VERSIONS", "10"))
    try:
        results["config_vacuum"] = _vacuum_config_metadata(keep_versions=keep_versions)
    except Exception as exc:  # noqa: BLE001 — best-effort vacuum
        results["config_vacuum"] = {"failed": str(exc)}

    return results


# Phoenix span-query retry budget. This helper runs in a per-agent loop, so
# the worst case compounds: 2 attempts x 60s per-attempt timeout + one 5s
# backoff = ~125s per call site on a down/hung Phoenix (3 x 120s + sleeps was
# 370s per agent).
_SPAN_QUERY_ATTEMPTS = 2
_SPAN_QUERY_TIMEOUT_S = 60

# Concurrency for the online-eval per-span scoring. Each span runs synchronous
# structural evaluators then one Phoenix annotation write; the writes dominate
# and are independent, so they overlap through a bounded pool instead of one
# serial round-trip at a time.
_ONLINE_EVAL_CONCURRENCY = 8


async def _query_spans_by_name(
    telemetry_manager,
    telemetry_provider,
    tenant_id: str,
    span_name: str,
    lookback_hours: float,
):
    """Query spans from Phoenix filtered by span name.

    Returns a DataFrame of matching spans, or an empty DataFrame if none found.
    """

    project_name = telemetry_manager.config.get_project_name(tenant_id)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=lookback_hours)

    last_exc: Exception | None = None
    for attempt in range(_SPAN_QUERY_ATTEMPTS):
        try:
            spans_df = await asyncio.wait_for(
                telemetry_provider.traces.get_all_spans(
                    project=project_name,
                    start_time=start_time,
                    end_time=end_time,
                    # Server-side name predicate — pulling the whole project
                    # window and filtering client-side costs a full scan of a
                    # project that accumulates thousands of spans a day.
                    filters={"name": span_name},
                ),
                timeout=_SPAN_QUERY_TIMEOUT_S,
            )
            break
        except Exception as e:
            last_exc = e
            logger.warning(
                "Span query for %s failed (attempt %d/%d): %s",
                span_name,
                attempt + 1,
                _SPAN_QUERY_ATTEMPTS,
                e,
            )
            if attempt + 1 < _SPAN_QUERY_ATTEMPTS:
                await asyncio.sleep(5)
    else:
        # A failed query is not "no spans" — reporting no_data here made a
        # Phoenix timeout look like an empty optimization window.
        raise RuntimeError(
            f"Failed to query {span_name} spans from Phoenix after "
            f"{_SPAN_QUERY_ATTEMPTS} attempts"
        ) from last_exc

    if spans_df.empty:
        return spans_df

    return spans_df[spans_df["name"] == span_name]


def _approved_example_exact_metric(example, prediction, trace=None) -> bool:
    """Accept a bootstrapped demonstration only when every reviewed label matches."""
    del trace
    expected = example.labels().toDict()
    return all(
        key in prediction and prediction[key] == value
        for key, value in expected.items()
    )


def _create_teleprompter(
    trainset_size: int,
    teacher_settings: dict | None = None,
    metric=_approved_example_exact_metric,
    metric_threshold: float | None = None,
):
    """Select DSPy optimizer config based on training set size.

    Scales BootstrapFewShot parameters for larger training sets:
    - < 50 examples: 4 bootstrapped demos, 8 labeled, 1 round
    - >= 50 examples: 8 bootstrapped demos, 16 labeled, 2 rounds

    teacher_settings (e.g. ``{"lm": teacher_lm}``) makes DSPy run the
    bootstrap teacher on the configured teacher endpoint instead of the
    student model teaching itself. ``metric`` decides which teacher traces
    become bootstrapped demos; with ``metric_threshold`` a trace is kept when
    the metric's score reaches it.
    """
    from dspy.teleprompt import BootstrapFewShot

    if trainset_size >= 50:
        logger.info(
            "Using scaled BootstrapFewShot for %d examples (>= 50 threshold)",
            trainset_size,
        )
        return BootstrapFewShot(
            metric=metric,
            metric_threshold=metric_threshold,
            max_bootstrapped_demos=8,
            max_labeled_demos=16,
            max_rounds=2,
            max_errors=10,
            teacher_settings=teacher_settings,
        )

    logger.info("Using BootstrapFewShot for %d examples", trainset_size)
    return BootstrapFewShot(
        metric=metric,
        metric_threshold=metric_threshold,
        max_bootstrapped_demos=4,
        max_labeled_demos=8,
        max_rounds=1,
        max_errors=5,
        teacher_settings=teacher_settings,
    )


def _project_approved_optimizer_example(
    optimizer_type: str, example: dict[str, Any]
) -> dict[str, Any]:
    """Project a validated approved record onto its production DSPy signature."""
    if optimizer_type == "query_enhancement":
        return {
            "query": example["query"],
            "enhanced_query": example["enhanced_query"],
            "expansion_terms": ", ".join(example["expansion_terms"]),
            "synonyms": ", ".join(example["synonyms"]),
            "context": example["context"],
            "confidence": "0.0",
            "reasoning": example["reasoning"],
        }
    if optimizer_type == "profile":
        return {
            "query": example["query"],
            "available_profiles": example["available_profiles"],
            "selected_profile": example["selected_profile"],
            "confidence": "0.0",
            "reasoning": example["reasoning"],
            "query_intent": example["query_intent"],
            "modality": example["modality"],
            "complexity": example["complexity"],
        }
    if optimizer_type == "entity_extraction":
        entities = "\n".join(
            f"{entity['text']}|{entity['type']}|1.0" for entity in example["entities"]
        )
        return {
            "query": example["query"],
            "entities": entities,
            "entity_types": example["entity_types"],
        }
    raise ValueError(
        f"optimizer {optimizer_type!r} has no approved DSPy example projection"
    )


async def _load_approved_synthetic_data(
    telemetry_provider,
    tenant_id: str,
    optimizer_type: str,
) -> list[dict[str, Any]]:
    """Load optimizer inputs from the dataset written by human approval."""
    from cogniverse_agents.approval.approval_storage import (
        validate_approved_dataset_record,
    )
    from cogniverse_core.approval.interfaces import (
        ApprovalStatus,
        approved_synthetic_dataset_name,
    )
    from cogniverse_core.approval.training_schema import (
        validate_approved_training_values,
    )
    from cogniverse_foundation.telemetry.providers.base import DatasetNotFoundError
    from cogniverse_synthetic.registry import (
        APPROVED_TRAINING_AGENT_BY_OPTIMIZER,
    )

    expected_agent_type = APPROVED_TRAINING_AGENT_BY_OPTIMIZER.get(optimizer_type)
    if expected_agent_type is None:
        raise ValueError(
            f"optimizer {optimizer_type!r} has no approved training-data consumer"
        )

    dataset_name = approved_synthetic_dataset_name(tenant_id)
    try:
        dataset_df = await telemetry_provider.datasets.get_dataset(name=dataset_name)
    except DatasetNotFoundError:
        return []
    except Exception as exc:
        raise RuntimeError(
            "Failed to load approved synthetic data for "
            f"tenant={tenant_id} optimizer={optimizer_type} "
            f"dataset={dataset_name}"
        ) from exc

    if dataset_df is None:
        raise RuntimeError(
            "Approved synthetic dataset provider returned no frame for "
            f"tenant={tenant_id} optimizer={optimizer_type} "
            f"dataset={dataset_name}"
        )
    if dataset_df.empty:
        return []

    bookkeeping_fields = {
        "item_id",
        "confidence",
        "status",
        "created_at",
        "reviewed_at",
    }
    approved = []
    for position, (_, row) in enumerate(dataset_df.iterrows()):
        record = row.get("input")
        if not isinstance(record, dict):
            raise ValueError(
                f"Approved synthetic dataset row {position} has no input record"
            )
        record = validate_approved_dataset_record(
            record,
            tenant_id=dataset_name.removeprefix("approved_synthetic_data-"),
            dataset_name=dataset_name,
            position=position,
        )
        if record.get("status") != ApprovalStatus.APPROVED.value:
            continue
        if record.get("context.optimizer") != optimizer_type:
            continue

        agent_type = record.get("metadata.agent_type")
        if agent_type != expected_agent_type:
            raise ValueError(
                f"Approved synthetic dataset row {position} for "
                f"optimizer={optimizer_type} requires "
                f"metadata.agent_type={expected_agent_type!r}, got {agent_type!r}"
            )

        example = {}
        for key, value in record.items():
            if (
                key in bookkeeping_fields
                or key.startswith("metadata.")
                or key.startswith("context.")
            ):
                continue
            example[key] = value
        validate_approved_training_values(
            example,
            expected_agent_type,
            context=(
                f"Approved synthetic dataset row {position} for "
                f"optimizer={optimizer_type}"
            ),
        )
        example["example_id"] = f"approved:{record['item_id']}"
        approved.append(example)

    logger.info(
        "Loaded %d/%d approved synthetic examples from %s for %s",
        len(approved),
        len(dataset_df),
        dataset_name,
        optimizer_type,
    )
    return approved


async def run_monthly_reports(
    output_dir: str,
    lookback_hours: float = 24.0 * 30,
    telemetry_otlp_endpoint: str | None = None,
) -> dict:
    """Generate the monthly usage + performance report.

    Replaces the standalone ``monthly-reports`` CronWorkflow that was
    a kubectl-applied stub (echoed empty JSON). This version collects
    real data:

      * **usage**: total orgs, total tenants per org, total schemas
        deployed per tenant (from organization_metadata + tenant_metadata).
      * **performance**: per-tenant span count, mean / p50 / p95 latency
        across every span the project emitted in the lookback window,
        plus error rate (status_code != OK).

    Writes ``usage-YYYYMM.json`` and ``performance-YYYYMM.json`` to
    ``output_dir`` so a follow-up workflow step can upload to MinIO via
    ``mc cp``. Returns a summary the workflow log captures verbatim.
    """
    import json
    from pathlib import Path

    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager
    from cogniverse_runtime.admin import tenant_manager

    create_default_config_manager()  # warm config singletons
    schemas_dir = Path(os.environ.get("COGNIVERSE_SCHEMAS_DIR", "configs/schemas"))
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

    tenant_manager.set_schema_loader(FilesystemSchemaLoader(schemas_dir))

    period = datetime.now().strftime("%Y%m")
    generated_at = datetime.now(timezone.utc).isoformat() + "Z"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- usage ---
    org_ids = await tenant_manager.list_organizations_internal()
    usage_per_org: Dict[str, Any] = {}
    total_tenants = 0
    total_schemas = 0
    for oid in org_ids:
        tenants = await tenant_manager.list_tenants_for_org_internal(oid)
        per_org_tenants = []
        for t in tenants:
            schemas = list(t.schemas_deployed or [])
            per_org_tenants.append(
                {
                    "tenant_full_id": t.tenant_full_id,
                    "tenant_name": t.tenant_name,
                    "status": t.status,
                    "schema_count": len(schemas),
                    "schemas_deployed": schemas,
                }
            )
            total_schemas += len(schemas)
        total_tenants += len(per_org_tenants)
        usage_per_org[oid] = {
            "tenant_count": len(per_org_tenants),
            "tenants": per_org_tenants,
        }
    usage_report = {
        "period": period,
        "generated_at": generated_at,
        "summary": {
            "org_count": len(org_ids),
            "tenant_count": total_tenants,
            "schema_count": total_schemas,
        },
        "organizations": usage_per_org,
    }
    usage_path = out / f"usage-{period}.json"
    usage_path.write_text(json.dumps(usage_report, indent=2, default=str))

    # --- performance ---
    telemetry_manager = get_telemetry_manager(otlp_endpoint=telemetry_otlp_endpoint)
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=lookback_hours)
    perf_per_tenant: Dict[str, Any] = {}
    tenant_ids = [
        t.tenant_full_id
        for oid in org_ids
        for t in await tenant_manager.list_tenants_for_org_internal(oid)
        if t.tenant_full_id
    ]
    for tid in tenant_ids:
        provider = telemetry_manager.get_provider(tenant_id=tid)
        project = telemetry_manager.config.get_project_name(tid)
        try:
            spans_df = await provider.traces.get_all_spans(
                project=project,
                start_time=start,
                end_time=end,
            )
        except Exception as exc:
            perf_per_tenant[tid] = {"error": f"phoenix query failed: {exc}"}
            continue
        if spans_df is None or spans_df.empty:
            perf_per_tenant[tid] = {
                "span_count": 0,
                "latency_ms_mean": None,
                "latency_ms_p50": None,
                "latency_ms_p95": None,
                "error_rate": 0.0,
            }
            continue

        # Pyhoenix dataframes expose `latency_ms` (start_time, end_time)
        # and a status_code column; fall back gracefully if either is
        # absent in older provider versions.
        latencies = []
        if "latency_ms" in spans_df.columns:
            latencies = [v for v in spans_df["latency_ms"].dropna() if v >= 0]
        elif {"start_time", "end_time"}.issubset(spans_df.columns):
            for s, e in zip(spans_df["start_time"], spans_df["end_time"]):
                try:
                    latencies.append((e - s).total_seconds() * 1000.0)
                except Exception:
                    continue
        errors = 0
        if "status_code" in spans_df.columns:
            errors = int(
                spans_df["status_code"].fillna("OK").str.upper().ne("OK").sum()
            )
        n = len(spans_df)
        latencies_sorted = sorted(latencies) if latencies else []

        def _pct(lst: list, q: float):
            if not lst:
                return None
            idx = max(0, min(len(lst) - 1, int(q * (len(lst) - 1))))
            return round(float(lst[idx]), 3)

        perf_per_tenant[tid] = {
            "span_count": int(n),
            "latency_ms_mean": (
                round(sum(latencies) / len(latencies), 3) if latencies else None
            ),
            "latency_ms_p50": _pct(latencies_sorted, 0.50),
            "latency_ms_p95": _pct(latencies_sorted, 0.95),
            "error_rate": round(errors / n, 4) if n else 0.0,
        }
    perf_report = {
        "period": period,
        "generated_at": generated_at,
        "lookback_hours": lookback_hours,
        "tenants": perf_per_tenant,
    }
    perf_path = out / f"performance-{period}.json"
    perf_path.write_text(json.dumps(perf_report, indent=2, default=str))

    result = {
        "period": period,
        "generated_at": generated_at,
        "output_dir": str(out),
        "files_written": [str(usage_path), str(perf_path)],
        "summary": {
            "org_count": len(org_ids),
            "tenant_count": total_tenants,
            "perf_tenants_with_data": sum(
                1
                for v in perf_per_tenant.values()
                if isinstance(v, dict) and v.get("span_count", 0) > 0
            ),
        },
    }
    # Surface per-tenant Phoenix outages at the TOP level so the cron's
    # _run_failed gate sees them and exits non-zero. Without this a total
    # Phoenix outage wrote an all-errors report yet the cron reported
    # Succeeded (the per-tenant "phoenix query failed" strings live only in
    # the file and don't match _run_failed's failed:/error: prefix), so the
    # dropped monthly reports were never regenerated. The usage/Vespa side
    # already exits non-zero by propagating — this aligns the perf side.
    perf_errors = sorted(
        tid
        for tid, v in perf_per_tenant.items()
        if isinstance(v, dict) and v.get("error")
    )
    if perf_errors:
        result["failed"] = perf_errors
    return result


SIMBA_ARTIFACT_KEY = "simba_query_enhancement"
_QUERY_ENHANCEMENT_INPUTS = ("query", "source_text", "grounding_context")


def _query_enhancement_example(record: Dict[str, Any]):
    """A DSPy example carrying a served call's real inputs and outputs."""
    import dspy

    fields = {
        "query": record["query"],
        "source_text": record["source_text"],
        "grounding_context": record["grounding_context"],
        "enhanced_query": record["enhanced_query"],
        "expansion_terms": ", ".join(record["expansion_terms"]),
        "synonyms": ", ".join(record["synonyms"]),
        "context": ", ".join(record["context"]),
        "confidence": str(record["confidence"]),
    }
    if record.get("reasoning"):
        fields["reasoning"] = record["reasoning"]
    return dspy.Example(**fields).with_inputs(*_QUERY_ENHANCEMENT_INPUTS)


_PROFILE_SELECTION_INPUTS = ("query", "available_profiles")


def _profile_selection_example(record: Dict[str, Any]):
    """A DSPy example carrying a profile-selection span or approved record."""
    import dspy

    available_profiles = record.get("available_profiles", "")
    if isinstance(available_profiles, (list, tuple, set)):
        available_profiles = ", ".join(
            str(profile).strip()
            for profile in available_profiles
            if str(profile).strip()
        )
    fields = {
        "query": record["query"],
        "available_profiles": available_profiles,
        "selected_profile": record["selected_profile"],
    }
    for key in ("confidence", "reasoning", "query_intent", "modality", "complexity"):
        if key in record and record[key] not in (None, ""):
            fields[key] = str(record[key]) if key == "confidence" else record[key]
    return dspy.Example(**fields).with_inputs(*_PROFILE_SELECTION_INPUTS)


def _query_enhancement_scores(module, holdout) -> tuple[float, int]:
    """Mean ``_query_enhancement_quality`` over scoreable holdout inputs."""
    scores = []
    for example in holdout:
        score = _query_enhancement_quality(
            module(**{k: getattr(example, k) for k in _QUERY_ENHANCEMENT_INPUTS}),
            example,
        )
        if score is None:
            continue
        scores.append(score)
    scored_count = len(scores)
    return (sum(scores) / scored_count if scored_count else 0.0, scored_count)


async def run_simba_optimization(
    tenant_id: str,
    lookback_hours: float = 24.0,
    telemetry_otlp_endpoint: str | None = None,
    embedder_url: Optional[str] = None,
) -> dict:
    """SIMBA query enhancement optimization.

    Reads cogniverse.query_enhancement spans into served-call records,
    splits them with a served-scoreable holdout, compiles the
    QueryEnhancementAgent's DSPy module via BootstrapFewShot on the trainable
    records, then scores the base module, the persisted artifact and the
    compiled candidate on the holdout with ``_query_enhancement_quality``.
    Below ``min_samples_for_optimization`` or ``min_unique_queries``, the run
    persists an ``insufficient_population`` version and leaves the served
    artifact unchanged.
    """
    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_foundation.telemetry.config import SPAN_NAME_QUERY_ENHANCEMENT
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    logger.info(
        "Starting SIMBA optimization for tenant=%s lookback=%dh",
        tenant_id,
        lookback_hours,
    )

    config_manager = create_default_config_manager()
    telemetry_manager = get_telemetry_manager(otlp_endpoint=telemetry_otlp_endpoint)
    telemetry_provider = telemetry_manager.get_provider(tenant_id=tenant_id)

    spans_df = await _query_spans_by_name(
        telemetry_manager,
        telemetry_provider,
        tenant_id,
        SPAN_NAME_QUERY_ENHANCEMENT,
        lookback_hours,
    )

    logger.info("Found %d query_enhancement spans", len(spans_df))

    records = _query_enhancement_pairs(spans_df)
    production_count = len(records)
    synthetic_demos = await _load_approved_synthetic_data(
        telemetry_provider, tenant_id, "query_enhancement"
    )
    for demo in synthetic_demos:
        projected = _project_approved_optimizer_example("query_enhancement", demo)
        records.append(
            {
                "query": projected["query"],
                "source_text": "",
                "grounding_context": "",
                "enhanced_query": projected["enhanced_query"],
                "expansion_terms": [
                    t.strip()
                    for t in projected["expansion_terms"].split(",")
                    if t.strip()
                ],
                "synonyms": [
                    s.strip() for s in projected["synonyms"].split(",") if s.strip()
                ],
                "context": [
                    c.strip() for c in projected["context"].split(",") if c.strip()
                ],
                "confidence": 0.0,
                "reasoning": projected["reasoning"],
                "example_id": demo["example_id"],
                "trainable": True,
            }
        )
    if not records:
        logger.info("No valid production or approved synthetic examples")
        return {"status": "no_data", "spans_found": len(spans_df), "examples": 0}

    artifact_manager = ArtifactManager(telemetry_provider, tenant_id)
    current_blob = await artifact_manager.load_blob("model", SIMBA_ARTIFACT_KEY)
    min_samples, min_unique_queries = _population_floor_from_config(
        tenant_id, config_manager, "simba_query_enhancement"
    )
    min_holdout = max(1, min_samples // 10)
    population = len(records)
    distinct_queries = len({record["query"] for record in records})
    if population < min_samples or distinct_queries < min_unique_queries:
        logger.warning(
            "SIMBA below population floor for %s: population=%d (min %d) "
            "distinct_queries=%d (min %d)",
            tenant_id,
            population,
            min_samples,
            distinct_queries,
            min_unique_queries,
        )
        consumed_example_ids = [record["example_id"] for record in records]
        _, version = await artifact_manager.save_blob_versioned(
            "model",
            SIMBA_ARTIFACT_KEY,
            current_blob or "{}",
            consumed_example_ids=consumed_example_ids,
            decision="insufficient_population",
            scored=False,
            score=None,
            base_score=None,
            candidate_score=None,
        )
        return {
            "status": "insufficient_population",
            "spans_found": len(spans_df),
            "examples": population,
            "distinct_queries": distinct_queries,
            "min_samples": min_samples,
            "min_unique_queries": min_unique_queries,
            "version": version,
        }

    served_scoreable_examples = len(_served_scoreable_indices(records))
    served_examples = production_count
    approved_examples = len(synthetic_demos)
    train_records, holdout_records = _split_served_holdout(records, min_holdout)
    trainable_records = [r for r in train_records if r["trainable"]]
    non_trainable_examples = len(train_records) - len(trainable_records)
    train_records, selection_report = await _apply_training_selection(
        artifact_manager=artifact_manager,
        config_manager=config_manager,
        tenant_id=tenant_id,
        optimizer_type="simba_query_enhancement",
        artifact_key=SIMBA_ARTIFACT_KEY,
        train_records=trainable_records,
        embedder_url=embedder_url,
    )
    train_records, unscoreable_examples = _scoreable_first(train_records)
    selection_summary = _selection_summary(selection_report)
    trainset = [_query_enhancement_example(r) for r in train_records]
    holdout = [_query_enhancement_example(r) for r in holdout_records]
    logger.info(
        "Merged %d synthetic + %d production = %d records: %d trainable, %d holdout",
        len(synthetic_demos),
        production_count,
        len(records),
        len(trainset),
        len(holdout),
    )
    if not holdout:
        logger.warning(
            "No served-scoreable holdout material for %s query enhancement — "
            "nothing persisted",
            tenant_id,
        )
        return {
            "status": "no_eval_material",
            "spans_found": len(spans_df),
            "examples": len(records),
            "served_scoreable_examples": served_scoreable_examples,
            "non_trainable_examples": non_trainable_examples,
            "unscoreable_examples": unscoreable_examples,
            "training_examples": len(trainset),
            "holdout_examples": 0,
            "holdout_source": "served",
            **selection_summary,
        }

    import dspy

    from cogniverse_agents.query_enhancement_agent import QueryEnhancementModule
    from cogniverse_foundation.config.llm_factory import create_dspy_lm
    from cogniverse_foundation.config.utils import get_config

    config = get_config(tenant_id=tenant_id, config_manager=config_manager)
    llm_config = config.get_llm_config()
    dspy.configure(lm=create_dspy_lm(llm_config.resolve("optimization")))

    try:
        baseline_score, scored_count = _query_enhancement_scores(
            QueryEnhancementModule(), holdout
        )
        if scored_count == 0:
            logger.warning(
                "No served-scoreable holdout material for %s query enhancement — "
                "nothing persisted",
                tenant_id,
            )
            return {
                "status": "no_eval_material",
                "spans_found": len(spans_df),
                "examples": len(records),
                "served_scoreable_examples": served_scoreable_examples,
                "non_trainable_examples": non_trainable_examples,
                "unscoreable_examples": unscoreable_examples,
                "training_examples": len(trainset),
                "holdout_examples": 0,
                "holdout_source": "served",
                **selection_summary,
            }

        current_module = None
        if current_blob:
            current_module = QueryEnhancementModule()
            current_module.load_state(json.loads(current_blob))

        compiled = None
        if trainset:
            teleprompter = _create_teleprompter(
                len(trainset),
                teacher_settings={"lm": teacher_lm_or_raise(llm_config)},
                metric=_query_enhancement_metric,
            )
            compiled = teleprompter.compile(QueryEnhancementModule(), trainset=trainset)

        current_score = (
            _query_enhancement_scores(current_module, holdout)[0]
            if current_module is not None
            else None
        )
        candidate_score = (
            _query_enhancement_scores(compiled, holdout)[0]
            if compiled is not None
            else None
        )
    except Exception as e:
        logger.error("SIMBA compilation failed: %s", e)
        return {"status": "failed", "error": str(e), **selection_summary}

    decision = _select_simba_artifact(
        baseline_score,
        current_score,
        candidate_score,
        _min_improvement_from_config(tenant_id, config_manager),
    )
    # Every run this reached persists a version whose ledger names the examples
    # it consumed; only promote / rollback move the active pointer. The version
    # content is what the run would serve: the candidate for promote / keep, the
    # base state for rollback, the candidate (or base if none compiled) for
    # reject.
    run_module = {
        "promote": compiled,
        "rollback": QueryEnhancementModule(),
    }.get(decision, compiled or QueryEnhancementModule())
    consumed_example_ids = [record["example_id"] for record in records]
    _, version = await artifact_manager.save_blob_versioned(
        "model",
        SIMBA_ARTIFACT_KEY,
        json.dumps(run_module.dump_state(), default=str),
        consumed_example_ids=consumed_example_ids,
        decision=decision,
        scored=candidate_score is not None,
        score=candidate_score,
        base_score=baseline_score,
        candidate_score=candidate_score,
    )
    if decision in ("promote", "rollback"):
        await artifact_manager.activate_version("model", SIMBA_ARTIFACT_KEY, version)
    logger.info(
        "SIMBA optimization %s v%d (baseline=%.3f current=%s candidate=%s, "
        "%d examples)",
        decision,
        version,
        baseline_score,
        current_score,
        candidate_score,
        len(consumed_example_ids),
    )
    return {
        "status": "success",
        "spans_found": len(spans_df),
        "examples": len(records),
        "served_examples": served_examples,
        "approved_examples": approved_examples,
        "served_scoreable_examples": served_scoreable_examples,
        "non_trainable_examples": non_trainable_examples,
        "unscoreable_examples": unscoreable_examples,
        "training_examples": len(trainset),
        "holdout_examples": len(holdout),
        "holdout_source": "served",
        **selection_summary,
        "baseline_score": baseline_score,
        "current_score": current_score,
        "candidate_score": candidate_score,
        "decision": decision,
        "version": version,
        "consumed_example_ids": consumed_example_ids,
    }


async def _save_workflow_learning_state(
    store,
    *,
    tenant_id: str,
    executions: list,
    profiles: list,
    patterns: dict,
    templates: list,
) -> None:
    """Replace one tenant's workflow learning artifacts through its store."""
    await store.replace_learning_state(
        tenant_id,
        executions,
        profiles,
        patterns,
        templates,
    )


async def run_workflow_optimization(
    tenant_id: str,
    lookback_hours: float = 24.0,
    telemetry_otlp_endpoint: str | None = None,
) -> dict:
    """Workflow orchestration optimization.

    Reads cogniverse.orchestration spans, feeds them through
    OrchestrationEvaluator to extract WorkflowExecution records,
    then generates workflow templates and agent performance profiles
    and saves them as artifacts.
    """
    from cogniverse_foundation.config.utils import create_default_config_manager

    logger.info(
        "Starting workflow optimization for tenant=%s lookback=%dh",
        tenant_id,
        lookback_hours,
    )

    create_default_config_manager()

    from cogniverse_agents.workflow.intelligence import WorkflowIntelligence

    intelligence = WorkflowIntelligence(tenant_id=tenant_id)

    from cogniverse_agents.routing.orchestration_evaluator import (
        OrchestrationEvaluator,
    )
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    get_telemetry_manager(otlp_endpoint=telemetry_otlp_endpoint)

    evaluator = OrchestrationEvaluator(
        workflow_intelligence=intelligence,
        tenant_id=tenant_id,
    )

    evaluation_end_time = datetime.now(timezone.utc)
    spans_found = 0
    workflows_extracted = 0
    while True:
        eval_result = await evaluator.evaluate_orchestration_spans(
            lookback_hours=lookback_hours,
            batch_size=50,
            evaluation_end_time=evaluation_end_time,
        )
        spans_found += eval_result["spans_processed"]
        workflows_extracted += eval_result["workflows_extracted"]
        if not eval_result["has_more"]:
            break

    logger.info("Extracted %d workflow executions from spans", workflows_extracted)

    if workflows_extracted == 0:
        logger.info("No orchestration spans found — nothing to optimize")
        return {
            "status": "no_data",
            "spans_found": spans_found,
            "workflows_extracted": 0,
        }

    # Drop executions whose agent_sequence references an agent that no
    # longer exists in the current configuration. Phoenix retains
    # historical spans across schema changes (e.g. an old agent that was
    # renamed or split into multiple agents), and the workflow optimizer
    # would otherwise persist demos that point at deleted agents —
    # those demos can't be replayed and trip downstream consumers
    # asserting ``agent in known_agents``.
    #
    # Read the live set from ``configs/config.json``'s ``agents`` block.
    # That file is the canonical source of which agents the runtime
    # routes to today. AgentRegistry would be the obvious alternative,
    # but it's populated by HTTP self-registration and starts empty in
    # this process (the optimization CLI runs in its own pod with no
    # agents registered against it), so an AgentRegistry-backed filter
    # would drop *every* demo, not just the stale ones.
    from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
    from cogniverse_foundation.config.utils import (
        create_default_config_manager,
        get_config,
    )

    _cfg = get_config(
        tenant_id=SYSTEM_TENANT_ID,
        config_manager=create_default_config_manager(),
    )
    _agents_section = (_cfg or {}).get("agents", {})
    _live_agents = {
        name
        for name, body in _agents_section.items()
        if isinstance(body, dict) and body.get("enabled", True)
    }
    if not _live_agents:
        raise RuntimeError(
            "configs/config.json 'agents' block is empty or unreachable; "
            "cannot filter stale workflow demos. Refusing to save "
            "execution_demos because every demo would be flagged stale "
            "(or every demo would slip through unchecked, depending on "
            "the filter's defensive default) — both are wrong."
        )

    def _agents_live(seq) -> bool:
        if isinstance(seq, str):
            seq = [a.strip() for a in seq.split(",") if a.strip()]
        return bool(seq) and all(a in _live_agents for a in seq)

    # Persist through the workflow store — the same registry-resolved store
    # WorkflowIntelligence reads back at orchestrator startup. Stale demos
    # (agents absent from the live config) are dropped first; the store owns
    # serialization and the demonstration/blob layout.
    from cogniverse_core.registries import WorkflowStoreRegistry

    store = WorkflowStoreRegistry.get(name="telemetry")

    live_executions = [
        execution
        for execution in intelligence.workflow_history
        if _agents_live(execution.agent_sequence)
    ]
    profiles, templates = intelligence.derive_learning_artifacts(live_executions)
    patterns = (
        dict(intelligence.query_type_patterns)
        if intelligence.query_type_patterns
        else {}
    )

    await _save_workflow_learning_state(
        store,
        tenant_id=tenant_id,
        executions=live_executions,
        profiles=profiles,
        patterns=patterns,
        templates=templates,
    )

    logger.info("Workflow optimization complete")
    return {
        "status": "success",
        "spans_found": spans_found,
        "workflows_extracted": workflows_extracted,
        "execution_demos_saved": len(live_executions),
        "agent_profiles_saved": len(profiles),
        "workflow_templates_saved": len(templates),
    }


GATEWAY_DEFAULT_THRESHOLD = 0.4


def _compute_gateway_thresholds(spans_df) -> dict:
    """Pure function: calibrate gateway thresholds from a spans DataFrame.

    Extracted from :func:`run_gateway_thresholds_optimization` so the
    calibration algorithm can be unit-tested against deterministic inputs.
    The async wrapper handles Phoenix I/O and artifact persistence.

    Returns one of:
    - ``{"status": "no_data", "spans_found": N, "reason": ...}`` when the
      input lacks the required attributes or has no confidence values.
    - ``{"status": "ready", "spans_found": N, "thresholds": {...}}`` with
      the calibrated ``fast_path_confidence_threshold``, ``gliner_threshold``
      and an ``analysis`` subdict.

    The ``ready`` status is not yet ``success`` because the artifact hasn't
    been persisted — the wrapper writes the dataset and converts.
    """
    import pandas as _pd

    if spans_df.empty:
        return {"status": "no_data", "spans_found": 0}

    # The gateway decision is on the canonical output.value slot.
    df = spans_df.copy()

    def _gateway_output(row) -> Dict[str, Any]:
        out = read_span_io(row)["output"]
        return out if isinstance(out, dict) else {}

    gw_outputs = df.apply(_gateway_output, axis=1)
    if gw_outputs.map(bool).sum() == 0:
        return {
            "status": "no_data",
            "spans_found": len(spans_df),
            "reason": "no_gateway_attributes",
        }

    df["_complexity"] = gw_outputs.apply(lambda d: d.get("complexity", ""))
    df["_confidence"] = gw_outputs.apply(lambda d: d.get("confidence", None))

    simple_spans = df[df["_complexity"] == "simple"]
    complex_spans = df[df["_complexity"] == "complex"]

    # Coerce, don't cast: one non-numeric confidence (e.g. an LM emitting
    # "high") must drop that row, not abort the whole tenant recompute.
    confidences = _pd.to_numeric(df["_confidence"], errors="coerce").dropna()
    if confidences.empty:
        return {
            "status": "no_data",
            "spans_found": len(df),
            "reason": "no_confidence_data",
        }

    simple_total = len(simple_spans)
    complex_total = len(complex_spans)
    status_col = "status_code"
    simple_errors = 0
    complex_errors = 0
    if status_col in df.columns:
        if simple_total > 0:
            simple_errors = len(simple_spans[simple_spans[status_col] == "ERROR"])
        if complex_total > 0:
            complex_errors = len(complex_spans[complex_spans[status_col] == "ERROR"])

    simple_error_rate = simple_errors / max(simple_total, 1)
    complex_error_rate = complex_errors / max(complex_total, 1)
    mean_confidence = float(confidences.mean())
    p25_confidence = float(confidences.quantile(0.25))

    # Threshold calibration — if simple routing is failing often, raise the
    # threshold so more queries go to orchestrator; if complex routing rarely
    # fails AND mean confidence is high, lower the threshold to keep more
    # queries on the fast path.
    current = GATEWAY_DEFAULT_THRESHOLD
    if simple_error_rate > 0.2:
        optimized_threshold = min(current + 0.1, 0.95)
    elif complex_error_rate < 0.05 and mean_confidence > 0.8:
        # Lower the threshold so more queries stay on the fast path. The floor
        # must sit BELOW the default (0.4) — a 0.5 floor raised the threshold
        # instead, pushing every query with confidence in [0.4, 0.5) to the
        # orchestrator, the opposite of the intended calibration.
        optimized_threshold = max(current - 0.05, 0.3)
    else:
        optimized_threshold = current

    optimized_gliner_threshold = max(0.15, min(p25_confidence * 0.8, 0.5))

    return {
        "status": "ready",
        "spans_found": len(df),
        "thresholds": {
            "fast_path_confidence_threshold": optimized_threshold,
            "gliner_threshold": round(optimized_gliner_threshold, 3),
            "analysis": {
                "total_spans": len(df),
                "simple_count": simple_total,
                "complex_count": complex_total,
                "simple_error_rate": round(simple_error_rate, 4),
                "complex_error_rate": round(complex_error_rate, 4),
                "mean_confidence": round(mean_confidence, 4),
                "p25_confidence": round(p25_confidence, 4),
            },
        },
    }


async def run_gateway_thresholds_optimization(
    tenant_id: str,
    lookback_hours: float = 24.0,
    telemetry_otlp_endpoint: str | None = None,
) -> dict:
    """Gateway confidence threshold tuning.

    Reads cogniverse.gateway spans, analyzes classification accuracy
    (was "simple" routing correct? did "complex" queries actually need
    orchestration?), and updates GLiNER confidence thresholds.
    Saves the threshold config as an artifact.
    """
    import json as _json

    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_foundation.telemetry.config import SPAN_NAME_GATEWAY
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    logger.info(
        "Starting gateway threshold optimization for tenant=%s lookback=%dh",
        tenant_id,
        lookback_hours,
    )

    create_default_config_manager()
    telemetry_manager = get_telemetry_manager(otlp_endpoint=telemetry_otlp_endpoint)
    telemetry_provider = telemetry_manager.get_provider(tenant_id=tenant_id)

    spans_df = await _query_spans_by_name(
        telemetry_manager,
        telemetry_provider,
        tenant_id,
        SPAN_NAME_GATEWAY,
        lookback_hours,
    )

    if spans_df.empty:
        logger.info("No gateway spans found — nothing to optimize")
        return {"status": "no_data", "spans_found": 0}

    logger.info("Found %d gateway spans", len(spans_df))

    result = _compute_gateway_thresholds(spans_df)
    if result["status"] != "ready":
        logger.info("Gateway threshold calibration skipped: %s", result.get("reason"))
        return result

    threshold_config = result["thresholds"]

    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

    artifact_manager = ArtifactManager(telemetry_provider, tenant_id)
    dataset_id = await artifact_manager.save_blob(
        kind="config",
        key="gateway_thresholds",
        content=_json.dumps(threshold_config),
    )

    logger.info(
        "Gateway threshold optimization complete — threshold %.2f -> %.2f, artifact %s",
        GATEWAY_DEFAULT_THRESHOLD,
        threshold_config["fast_path_confidence_threshold"],
        dataset_id,
    )

    return {
        "status": "success",
        "spans_found": result["spans_found"],
        "artifact_id": dataset_id,
        "thresholds": threshold_config,
    }


async def run_online_routing_evaluation(
    tenant_id: str,
    lookback_hours: float = 24.0,
    telemetry_otlp_endpoint: str | None = None,
) -> dict:
    """Online routing-span scoring.

    Reads cogniverse.routing spans and scores each one (routing_outcome +
    confidence_calibration) via OnlineEvaluator, persisting the scores as
    telemetry annotations for drift detection. Sampling rate, evaluator set,
    and persistence are driven by automation_rules.online_evaluation in config.
    """
    from cogniverse_agents.routing.config import OnlineEvaluationConfig
    from cogniverse_evaluation.online_evaluator import OnlineEvaluator
    from cogniverse_foundation.config.utils import (
        create_default_config_manager,
        get_config,
    )
    from cogniverse_foundation.telemetry.config import SPAN_NAME_ROUTING
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    config_manager = create_default_config_manager()
    cfg = get_config(tenant_id=tenant_id, config_manager=config_manager)
    online_dict = (cfg.get_all().get("automation_rules") or {}).get(
        "online_evaluation"
    ) or {}
    online_cfg = OnlineEvaluationConfig(**online_dict)

    if not online_cfg.enabled:
        logger.info("Online routing evaluation disabled in config")
        return {"status": "disabled"}

    telemetry_manager = get_telemetry_manager(otlp_endpoint=telemetry_otlp_endpoint)
    telemetry_provider = telemetry_manager.get_provider(tenant_id=tenant_id)
    project_name = telemetry_manager.config.get_project_name(tenant_id)

    spans_df = await _query_spans_by_name(
        telemetry_manager,
        telemetry_provider,
        tenant_id,
        SPAN_NAME_ROUTING,
        lookback_hours,
    )
    if spans_df.empty:
        logger.info("No routing spans found — nothing to evaluate")
        return {"status": "no_data", "spans_found": 0}

    logger.info("Found %d routing spans", len(spans_df))

    evaluator = OnlineEvaluator(
        provider=telemetry_provider,
        project_name=project_name,
        config=online_cfg,
    )

    scores_persisted = 0
    for _, row in spans_df.iterrows():
        results = await evaluator.evaluate_span(row.to_dict())
        scores_persisted += len(results)

    stats = evaluator.get_statistics()
    logger.info(
        "Online routing evaluation complete — evaluated %d spans, persisted %d scores",
        stats["total_evaluated"],
        scores_persisted,
    )
    return {
        "status": "success",
        "spans_found": len(spans_df),
        "scores_persisted": scores_persisted,
        "statistics": stats,
    }


def _online_eval_agent_types() -> list[str]:
    """Agent types the online span-eval cycle scores: registry entries with
    structural evaluators over their own ``cogniverse.*`` domain spans. The
    ``<ClassName>.process`` base spans carry no canonical payload, so
    structural scoring of them would only produce defaults."""
    from cogniverse_evaluation.evaluators.agent_evaluators import AGENT_EVALUATORS

    return [
        agent_type
        for agent_type, entry in AGENT_EVALUATORS.items()
        if entry.structural and entry.span_name.startswith("cogniverse.")
    ]


async def _score_spans_bounded(evaluator, rows: list[dict]) -> int:
    """Score ``rows`` through ``evaluator.evaluate_span`` concurrently, bounded
    to ``_ONLINE_EVAL_CONCURRENCY`` in-flight, and return the total number of
    scores persisted. Each span's structural evaluation is synchronous; the
    per-span Phoenix annotation write is the awaited part, so overlapping them
    cuts the wall-clock from N serial round-trips to ceil(N/bound). The
    evaluator's counters are incremented in statements with no await between
    read and write, so concurrent tasks cannot lose an increment.
    """
    sem = asyncio.Semaphore(_ONLINE_EVAL_CONCURRENCY)

    async def _score(row_dict: dict) -> int:
        async with sem:
            return len(await evaluator.evaluate_span(row_dict))

    if not rows:
        return 0
    return sum(await asyncio.gather(*(_score(r) for r in rows)))


async def run_online_evaluation(
    tenant_id: str,
    lookback_hours: float | None = None,
    agent_types: list[str] | None = None,
    telemetry_otlp_endpoint: str | None = None,
) -> dict:
    """Online span scoring for every domain-span agent type.

    Generalizes the routing-only cycle: each agent type's spans are scored by
    its registry entry's structural evaluators and persisted as
    ``online_eval.<evaluator>`` annotations. Lookback and per-type batch size
    default from ``automation_rules.optimization_triggers``
    (``span_eval_lookback_hours`` / ``span_eval_batch_size``).
    """
    from cogniverse_agents.routing.config import (
        OnlineEvaluationConfig,
        OptimizationTriggersConfig,
    )
    from cogniverse_evaluation.evaluators.agent_evaluators import get_agent_evaluator
    from cogniverse_evaluation.online_evaluator import OnlineEvaluator
    from cogniverse_foundation.config.utils import (
        create_default_config_manager,
        get_config,
    )
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    config_manager = create_default_config_manager()
    cfg = get_config(tenant_id=tenant_id, config_manager=config_manager)
    automation_rules = cfg.get_all().get("automation_rules") or {}
    online_cfg = OnlineEvaluationConfig(
        **(automation_rules.get("online_evaluation") or {})
    )
    triggers_cfg = OptimizationTriggersConfig(
        **(automation_rules.get("optimization_triggers") or {})
    )

    if not online_cfg.enabled:
        logger.info("Online evaluation disabled in config")
        return {"status": "disabled"}

    if lookback_hours is None:
        lookback_hours = float(triggers_cfg.span_eval_lookback_hours)
    if agent_types is None:
        agent_types = _online_eval_agent_types()

    telemetry_manager = get_telemetry_manager(otlp_endpoint=telemetry_otlp_endpoint)
    telemetry_provider = telemetry_manager.get_provider(tenant_id=tenant_id)
    project_name = telemetry_manager.config.get_project_name(tenant_id)

    per_agent: dict = {}
    total_spans = 0
    total_persisted = 0
    for agent_type in agent_types:
        entry = get_agent_evaluator(agent_type)
        if entry is None or not entry.structural:
            logger.warning("No structural evaluators for %s — skipping", agent_type)
            continue

        spans_df = await _query_spans_by_name(
            telemetry_manager,
            telemetry_provider,
            tenant_id,
            entry.span_name,
            lookback_hours,
        )
        if spans_df.empty:
            per_agent[agent_type] = {"spans_found": 0, "scores_persisted": 0}
            continue
        if len(spans_df) > triggers_cfg.span_eval_batch_size:
            logger.info(
                "%s: capping %d spans to span_eval_batch_size=%d",
                agent_type,
                len(spans_df),
                triggers_cfg.span_eval_batch_size,
            )
            spans_df = spans_df.head(triggers_cfg.span_eval_batch_size)

        evaluator = OnlineEvaluator(
            provider=telemetry_provider,
            project_name=project_name,
            config=online_cfg,
            agent_type=agent_type,
        )
        if agent_type != "routing":
            # config.evaluators names the routing evaluators; every other
            # agent type runs its own registry-defined structural set.
            evaluator.evaluator_names = list(entry.structural.keys())

        rows = [row.to_dict() for _, row in spans_df.iterrows()]
        scores_persisted = await _score_spans_bounded(evaluator, rows)

        per_agent[agent_type] = {
            "spans_found": len(spans_df),
            "scores_persisted": scores_persisted,
            "statistics": evaluator.get_statistics(),
        }
        total_spans += len(spans_df)
        total_persisted += scores_persisted
        logger.info(
            "%s: evaluated %d spans, persisted %d scores",
            agent_type,
            len(spans_df),
            scores_persisted,
        )

    return {
        "status": "success",
        "spans_found": total_spans,
        "scores_persisted": total_persisted,
        "agents": per_agent,
    }


async def run_profile_optimization(
    tenant_id: str,
    lookback_hours: float = 24.0,
    telemetry_otlp_endpoint: str | None = None,
    embedder_url: Optional[str] = None,
) -> dict:
    """Profile selection optimization.

    Reads the shipped sample query corpus, derives one profile label per
    recoverable query by running the tenant's real search backend against each
    candidate profile, compiles the ProfileSelectionAgent's DSPy module, and
    saves the optimized module as an artifact.
    """
    from cogniverse_agents.profile_selection_agent import tenant_usable_profile_names
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.utils import (
        create_default_config_manager,
        get_config,
    )
    from cogniverse_foundation.telemetry.config import SPAN_NAME_PROFILE_SELECTION
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    logger.info(
        "Starting profile selection optimization for tenant=%s lookback=%dh",
        tenant_id,
        lookback_hours,
    )

    config_manager = create_default_config_manager()
    telemetry_manager = get_telemetry_manager(otlp_endpoint=telemetry_otlp_endpoint)
    telemetry_provider = telemetry_manager.get_provider(tenant_id=tenant_id)

    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
    from cogniverse_agents.optimizer.profile_selection_ground_truth import (
        ProfileSelectionGroundTruthMissingError,
        ProfileSelectionGroundTruthStoreUnavailableError,
        load_profile_selection_ground_truth_rows,
    )

    artifact_manager = ArtifactManager(telemetry_provider, tenant_id)
    try:
        ground_truth_rows = await load_profile_selection_ground_truth_rows(
            artifact_manager
        )
    except ProfileSelectionGroundTruthMissingError as exc:
        return exc.to_result()
    except ProfileSelectionGroundTruthStoreUnavailableError as exc:
        return exc.to_result()

    logger.info(
        "Loaded %d profile_selection ground-truth rows for tenant=%s",
        len(ground_truth_rows),
        tenant_id,
    )

    spans_df = await _query_spans_by_name(
        telemetry_manager,
        telemetry_provider,
        tenant_id,
        SPAN_NAME_PROFILE_SELECTION,
        lookback_hours,
    )

    logger.info("Found %d profile_selection spans", len(spans_df))

    config = get_config(tenant_id=tenant_id, config_manager=config_manager)
    candidate_profiles = tenant_usable_profile_names(config_manager, tenant_id)
    schemas_dir = Path(os.environ.get("COGNIVERSE_SCHEMAS_DIR", "configs/schemas"))
    label_source = await asyncio.to_thread(
        _profile_selection_label_source,
        config=config,
        config_manager=config_manager,
        tenant_id=tenant_id,
        candidate_profiles=candidate_profiles,
        schema_loader=FilesystemSchemaLoader(schemas_dir),
    )

    profile_pairs = list(label_source.records)
    synthetic_demos = await _load_approved_synthetic_data(
        telemetry_provider, tenant_id, "profile"
    )
    consumed_example_ids = [pair["example_id"] for pair in profile_pairs]
    records = list(profile_pairs)
    label_exclusions = {
        "count": label_source.excluded_count,
        "queries": list(label_source.excluded_queries),
    }
    for demo in synthetic_demos:
        projected = _project_approved_optimizer_example("profile", demo)
        consumed_example_ids.append(demo["example_id"])
        records.append(
            {
                "query": projected["query"],
                "available_profiles": projected["available_profiles"],
                "selected_profile": projected["selected_profile"],
                "confidence": 0.0,
                "reasoning": projected["reasoning"],
                "query_intent": projected["query_intent"],
                "modality": projected["modality"],
                "complexity": projected["complexity"],
                "example_id": demo["example_id"],
            }
        )
    if not records:
        logger.info("No valid production or approved synthetic examples")
        return {
            "status": "no_data",
            "spans_found": len(spans_df),
            "examples": 0,
            "label_exclusions": label_exclusions,
        }

    current_blob = await artifact_manager.load_blob("model", "profile_selection")
    min_samples, min_unique_queries = _population_floor_from_config(
        tenant_id, config_manager, "profile_selection"
    )
    population = len(records)
    distinct_queries = len({record["query"] for record in records})
    if population < min_samples or distinct_queries < min_unique_queries:
        logger.warning(
            "Profile selection below population floor for %s: population=%d "
            "(min %d) distinct_queries=%d (min %d)",
            tenant_id,
            population,
            min_samples,
            distinct_queries,
            min_unique_queries,
        )
        _, version = await artifact_manager.save_blob_versioned(
            "model",
            "profile_selection",
            current_blob or "{}",
            consumed_example_ids=consumed_example_ids,
            decision="insufficient_population",
            scored=False,
            score=None,
            base_score=None,
            candidate_score=None,
        )
        return {
            "status": "insufficient_population",
            "spans_found": len(spans_df),
            "examples": population,
            "distinct_queries": distinct_queries,
            "min_samples": min_samples,
            "min_unique_queries": min_unique_queries,
            "version": version,
            "label_exclusions": label_exclusions,
        }

    min_holdout = max(1, min_samples // 10)
    served_records = [dict(record) for record in records]

    served_scoreable_examples = len(
        _served_scoreable_indices(
            served_records,
            scoreable_predicate=_profile_selection_is_scoreable,
        )
    )
    served_examples = len(profile_pairs)
    approved_examples = len(synthetic_demos)
    train_records, holdout_records = _split_served_holdout(
        served_records,
        min_holdout,
        scoreable_predicate=_profile_selection_is_scoreable,
    )
    train_records, selection_report = await _apply_training_selection(
        artifact_manager=artifact_manager,
        config_manager=config_manager,
        tenant_id=tenant_id,
        optimizer_type="profile_selection",
        artifact_key="profile_selection",
        train_records=train_records,
        embedder_url=embedder_url,
    )
    selection_summary = _selection_summary(selection_report)
    trainset = [_profile_selection_example(record) for record in train_records]
    holdout = [_profile_selection_example(record) for record in holdout_records]
    logger.info(
        "Merged %d synthetic + %d production = %d total training examples",
        len(synthetic_demos),
        len(profile_pairs),
        len(records),
    )
    if not holdout:
        logger.warning(
            "No served-scoreable holdout material for %s profile selection — "
            "nothing persisted",
            tenant_id,
        )
        return {
            "status": "no_eval_material",
            "spans_found": len(spans_df),
            "served_scoreable_examples": served_scoreable_examples,
            "training_examples": len(trainset),
            "holdout_examples": 0,
            "holdout_source": "derived_labels",
            "label_exclusions": label_exclusions,
            **selection_summary,
        }

    import dspy

    from cogniverse_agents.profile_selection_agent import ProfileSelectionModule
    from cogniverse_foundation.config.llm_factory import create_dspy_lm

    llm_config = config.get_llm_config()
    llm_endpoint = llm_config.resolve("optimization")

    dspy.configure(lm=create_dspy_lm(llm_endpoint))

    try:
        baseline_score = _profile_selection_scores(ProfileSelectionModule(), holdout)

        current_module = None
        if current_blob:
            current_module = ProfileSelectionModule()
            current_module.load_state(json.loads(current_blob))

        compiled = None
        if trainset:
            teleprompter = _create_teleprompter(
                len(trainset),
                teacher_settings={"lm": teacher_lm_or_raise(llm_config)},
                metric=_profile_selection_metric,
            )
            compiled = teleprompter.compile(ProfileSelectionModule(), trainset=trainset)

        current_score = (
            _profile_selection_scores(current_module, holdout)
            if current_module is not None
            else None
        )
        candidate_score = (
            _profile_selection_scores(compiled, holdout)
            if compiled is not None
            else None
        )
    except Exception as e:
        logger.error("Profile DSPy compilation failed: %s", e)
        return {
            "status": "failed",
            "error": str(e),
            "label_exclusions": label_exclusions,
            **selection_summary,
        }

    decision = _select_simba_artifact(
        baseline_score,
        current_score,
        candidate_score,
        _min_improvement_from_config(tenant_id, config_manager),
    )
    run_module = {
        "promote": compiled,
        "rollback": ProfileSelectionModule(),
    }.get(decision, compiled or ProfileSelectionModule())
    _, version = await artifact_manager.save_blob_versioned(
        "model",
        "profile_selection",
        json.dumps(run_module.dump_state(), default=str),
        consumed_example_ids=consumed_example_ids,
        decision=decision,
        scored=candidate_score is not None,
        score=candidate_score,
        base_score=baseline_score,
        candidate_score=candidate_score,
    )
    if decision in ("promote", "rollback"):
        await artifact_manager.activate_version("model", "profile_selection", version)

    logger.info(
        "Profile optimization %s v%d (baseline=%.3f current=%s candidate=%s, "
        "%d examples)",
        decision,
        version,
        baseline_score,
        current_score,
        candidate_score,
        len(consumed_example_ids),
    )
    return {
        "status": "success",
        "spans_found": len(spans_df),
        "served_examples": served_examples,
        "approved_examples": approved_examples,
        "served_scoreable_examples": served_scoreable_examples,
        "training_examples": len(trainset),
        "holdout_examples": len(holdout),
        "holdout_source": "derived_labels",
        "label_exclusions": label_exclusions,
        **selection_summary,
        "baseline_score": baseline_score,
        "current_score": current_score,
        "candidate_score": candidate_score,
        "decision": decision,
        "version": version,
        "consumed_example_ids": consumed_example_ids,
    }


async def run_entity_extraction_optimization(
    tenant_id: str,
    lookback_hours: float = 24.0,
    telemetry_otlp_endpoint: str | None = None,
    embedder_url: Optional[str] = None,
) -> dict:
    """Entity extraction optimization.

    Reads cogniverse.entity_extraction spans, builds training examples
    from (query) -> (entities) pairs, compiles the EntityExtractionModule's
    DSPy module, and saves the optimized module as an artifact.
    """
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_foundation.telemetry.config import SPAN_NAME_ENTITY_EXTRACTION
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    logger.info(
        "Starting entity extraction optimization for tenant=%s lookback=%dh",
        tenant_id,
        lookback_hours,
    )

    config_manager = create_default_config_manager()
    telemetry_manager = get_telemetry_manager(otlp_endpoint=telemetry_otlp_endpoint)
    telemetry_provider = telemetry_manager.get_provider(tenant_id=tenant_id)

    spans_df = await _query_spans_by_name(
        telemetry_manager,
        telemetry_provider,
        tenant_id,
        SPAN_NAME_ENTITY_EXTRACTION,
        lookback_hours,
    )

    logger.info("Found %d entity_extraction spans", len(spans_df))

    import json as _json

    entity_pairs = _entity_extraction_pairs(spans_df)
    consumed_example_ids = [pair["example_id"] for pair in entity_pairs]
    records = [
        {
            "query": pair["query"],
            "entities": pair["entities"],
            "entity_types": "",
            "example_id": pair["example_id"],
        }
        for pair in entity_pairs
    ]
    synthetic_demos = await _load_approved_synthetic_data(
        telemetry_provider, tenant_id, "entity_extraction"
    )
    for demo in synthetic_demos:
        projected = _project_approved_optimizer_example("entity_extraction", demo)
        consumed_example_ids.append(demo["example_id"])
        records.append(
            {
                "query": projected["query"],
                "entities": projected["entities"],
                "entity_types": projected["entity_types"],
                "example_id": demo["example_id"],
            }
        )
    if not records:
        logger.info("No valid production or approved synthetic examples")
        return {"status": "no_data", "spans_found": len(spans_df), "examples": 0}

    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

    artifact_manager = ArtifactManager(telemetry_provider, tenant_id)
    current_blob = await artifact_manager.load_blob("model", "entity_extraction")
    min_samples, min_unique_queries = _population_floor_from_config(
        tenant_id, config_manager, "entity_extraction"
    )
    population = len(records)
    distinct_queries = len({record["query"] for record in records})
    if population < min_samples or distinct_queries < min_unique_queries:
        logger.warning(
            "Entity extraction below population floor for %s: population=%d "
            "(min %d) distinct_queries=%d (min %d)",
            tenant_id,
            population,
            min_samples,
            distinct_queries,
            min_unique_queries,
        )
        _, version = await artifact_manager.save_blob_versioned(
            "model",
            "entity_extraction",
            current_blob or "{}",
            consumed_example_ids=consumed_example_ids,
            decision="insufficient_population",
            scored=False,
            score=None,
            base_score=None,
            candidate_score=None,
        )
        return {
            "status": "insufficient_population",
            "spans_found": len(spans_df),
            "examples": population,
            "distinct_queries": distinct_queries,
            "min_samples": min_samples,
            "min_unique_queries": min_unique_queries,
            "version": version,
        }

    min_holdout = max(1, min_samples // 10)
    served_scoreable_examples = len(
        _served_scoreable_indices(
            records,
            scoreable_predicate=_entity_extraction_is_scoreable,
        )
    )
    served_examples = len(entity_pairs)
    approved_examples = len(synthetic_demos)
    train_records, holdout_records = _split_served_holdout(
        records,
        min_holdout,
        scoreable_predicate=_entity_extraction_is_scoreable,
    )
    train_records, selection_report = await _apply_training_selection(
        artifact_manager=artifact_manager,
        config_manager=config_manager,
        tenant_id=tenant_id,
        optimizer_type="entity_extraction",
        artifact_key="entity_extraction",
        train_records=train_records,
        embedder_url=embedder_url,
    )
    selection_summary = _selection_summary(selection_report)

    import dspy

    trainset = [_entity_extraction_example(record) for record in train_records]
    holdout = [_entity_extraction_example(record) for record in holdout_records]
    logger.info(
        "Merged %d synthetic + %d production = %d total training examples",
        len(synthetic_demos),
        len(entity_pairs),
        len(records),
    )
    if not holdout:
        logger.warning(
            "No served-scoreable holdout material for %s entity extraction — "
            "nothing persisted",
            tenant_id,
        )
        return {
            "status": "no_eval_material",
            "spans_found": len(spans_df),
            "served_scoreable_examples": served_scoreable_examples,
            "training_examples": len(trainset),
            "holdout_examples": 0,
            "holdout_source": "served",
            **selection_summary,
        }
    from cogniverse_agents.entity_extraction_agent import EntityExtractionModule
    from cogniverse_foundation.config.llm_factory import create_dspy_lm
    from cogniverse_foundation.config.utils import get_config

    config = get_config(tenant_id=tenant_id, config_manager=config_manager)
    llm_config = config.get_llm_config()
    llm_endpoint = llm_config.resolve("optimization")

    dspy.configure(lm=create_dspy_lm(llm_endpoint))

    try:
        baseline_score = _entity_extraction_scores(EntityExtractionModule(), holdout)

        current_module = None
        if current_blob:
            current_module = EntityExtractionModule()
            current_module.load_state(_json.loads(current_blob))
        current_score = (
            _entity_extraction_scores(current_module, holdout)
            if current_module is not None
            else None
        )

        compiled = None
        bootstrap = None
        if trainset:
            recorder = BootstrapMetricRecorder(
                _entity_extraction_quality,
                threshold=_entity_bootstrap_threshold(baseline_score, current_score),
            )
            teleprompter = _create_teleprompter(
                len(trainset),
                teacher_settings={"lm": teacher_lm_or_raise(llm_config)},
                metric=recorder,
                metric_threshold=recorder.threshold,
            )
            compiled = teleprompter.compile(EntityExtractionModule(), trainset=trainset)
            bootstrap = _bootstrap_report(
                recorder, teleprompter, compiled, len(trainset)
            )
            logger.info("Entity extraction bootstrap for %s: %s", tenant_id, bootstrap)

        candidate_score = (
            _entity_extraction_scores(compiled, holdout)
            if compiled is not None
            else None
        )
    except Exception as e:
        logger.error("Entity extraction DSPy compilation failed: %s", e)
        return {"status": "failed", "error": str(e), **selection_summary}

    decision = _select_simba_artifact(
        baseline_score,
        current_score,
        candidate_score,
        _min_improvement_from_config(tenant_id, config_manager),
    )
    run_module = {
        "promote": compiled,
        "rollback": EntityExtractionModule(),
    }.get(decision, compiled or EntityExtractionModule())
    _, version = await artifact_manager.save_blob_versioned(
        "model",
        "entity_extraction",
        _json.dumps(run_module.dump_state(), default=str),
        consumed_example_ids=consumed_example_ids,
        decision=decision,
        scored=candidate_score is not None,
        score=candidate_score,
        base_score=baseline_score,
        candidate_score=candidate_score,
    )
    if decision in ("promote", "rollback"):
        await artifact_manager.activate_version("model", "entity_extraction", version)

    logger.info(
        "Entity extraction optimization %s v%d (baseline=%.3f current=%s "
        "candidate=%s, %d examples)",
        decision,
        version,
        baseline_score,
        current_score,
        candidate_score,
        len(consumed_example_ids),
    )
    return {
        "status": "success",
        "spans_found": len(spans_df),
        "served_examples": served_examples,
        "approved_examples": approved_examples,
        "served_scoreable_examples": served_scoreable_examples,
        "training_examples": len(trainset),
        "holdout_examples": len(holdout),
        "holdout_source": "served",
        **selection_summary,
        "bootstrap": bootstrap,
        "baseline_score": baseline_score,
        "current_score": current_score,
        "candidate_score": candidate_score,
        "decision": decision,
        "version": version,
        "consumed_example_ids": consumed_example_ids,
    }


def _synthetic_aggregate_status(results: dict[str, dict[str, Any]]) -> str:
    statuses = {
        optimizer_type: result.get("status")
        for optimizer_type, result in results.items()
    }
    unsupported = {
        optimizer_type: status
        for optimizer_type, status in statuses.items()
        if status not in {"success", "no_data", "failed", "error"}
    }
    if unsupported:
        raise ValueError(f"Unsupported synthetic result statuses: {unsupported}")
    if not statuses:
        raise ValueError("Synthetic generation requires at least one optimizer result")
    if any(status in {"failed", "error"} for status in statuses.values()):
        return "failed"
    if any(status == "success" for status in statuses.values()):
        return "success"
    return "no_data"


async def _build_cli_entity_extractor(
    *,
    config_manager,
    telemetry_manager,
    tenant_id: str,
):
    """Build the production entity agent used to label synthetic examples."""
    from cogniverse_agents.entity_extraction_agent import (
        EntityExtractionAgent,
        EntityExtractionDeps,
        EntityExtractionInput,
    )

    system_config = config_manager.get_system_config()
    service_urls = system_config.inference_service_urls
    gliner_url = service_urls.get("gliner") if isinstance(service_urls, dict) else None
    if not isinstance(gliner_url, str) or not gliner_url.strip():
        raise ValueError(
            "GLiNER inference endpoint is required for synthetic entity extraction "
            f"for tenant={tenant_id!r}"
        )

    def build_agent():
        agent = EntityExtractionAgent(
            deps=EntityExtractionDeps(gliner_inference_url=gliner_url)
        )
        agent.telemetry_manager = telemetry_manager
        agent._config_manager = config_manager
        agent._artifact_tenant_id = tenant_id
        agent._load_artifact()
        return agent

    try:
        agent = await asyncio.to_thread(build_agent)
    except Exception as exc:
        raise RuntimeError(
            "Entity extraction agent initialization failed for "
            f"tenant={tenant_id!r} endpoint={gliner_url!r}: {exc}"
        ) from exc

    async def extract_entities(source_text: str, request_tenant_id: str):
        if request_tenant_id != tenant_id:
            raise ValueError(
                "Entity extraction agent tenant mismatch: "
                f"configured={tenant_id!r} requested={request_tenant_id!r}"
            )
        typed_input = EntityExtractionInput(
            query=source_text,
            tenant_id=request_tenant_id,
        )
        try:
            return await agent.process(typed_input)
        except Exception as exc:
            raise RuntimeError(
                "Entity extraction agent failed for "
                f"tenant={request_tenant_id!r} source_text={source_text!r}: {exc}"
            ) from exc

    return extract_entities


async def _build_cli_routing_decider(
    *, config_manager: Any, telemetry_manager: Any, tenant_id: str
):
    """Build a tenant-bound production GatewayAgent routing callback."""
    from cogniverse_agents.gateway_agent import GatewayAgent, GatewayDeps, GatewayInput

    system_config = config_manager.get_system_config()
    service_urls = system_config.inference_service_urls
    gliner_url = service_urls.get("gliner") if isinstance(service_urls, dict) else None
    if not isinstance(gliner_url, str) or not gliner_url.strip():
        raise ValueError(
            "GLiNER inference endpoint is required for synthetic routing "
            f"for tenant={tenant_id!r}"
        )

    def build_agent():
        agent = GatewayAgent(deps=GatewayDeps(gliner_inference_url=gliner_url))
        agent.telemetry_manager = telemetry_manager
        agent._config_manager = config_manager
        agent._artifact_tenant_id = tenant_id
        agent._load_artifact()
        return agent

    try:
        agent = await asyncio.to_thread(build_agent)
    except Exception as exc:
        raise RuntimeError(
            "Gateway agent initialization failed for "
            f"tenant={tenant_id!r} endpoint={gliner_url!r}: {exc}"
        ) from exc

    async def route_query(query: str, request_tenant_id: str):
        if request_tenant_id != tenant_id:
            raise ValueError(
                "Gateway agent tenant mismatch: "
                f"configured={tenant_id!r} requested={request_tenant_id!r}"
            )
        try:
            return await agent.process(
                GatewayInput(query=query, tenant_id=request_tenant_id)
            )
        except Exception as exc:
            raise RuntimeError(
                "Gateway agent failed for "
                f"tenant={request_tenant_id!r} query={query!r}: {exc}"
            ) from exc

    return route_query


async def _build_cli_query_enhancer(
    *, config_manager: Any, telemetry_manager: Any, tenant_id: str
):
    """Build a tenant-bound production QueryEnhancementAgent callback."""
    from cogniverse_agents.query_enhancement_agent import (
        QueryEnhancementAgent,
        QueryEnhancementDeps,
        QueryEnhancementInput,
    )

    agent = QueryEnhancementAgent(deps=QueryEnhancementDeps())
    agent.telemetry_manager = telemetry_manager
    agent._config_manager = config_manager
    agent._artifact_tenant_id = tenant_id
    agent._load_artifact()

    async def enhance_query(query: str, request_tenant_id: str, source_text: str):
        if request_tenant_id != tenant_id:
            raise ValueError(
                "Query enhancement agent tenant mismatch: "
                f"configured={tenant_id!r} requested={request_tenant_id!r}"
            )
        try:
            return await agent.process(
                QueryEnhancementInput(
                    query=query,
                    source_text=source_text,
                    tenant_id=request_tenant_id,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "Query enhancement agent failed for "
                f"tenant={request_tenant_id!r} query={query!r}: {exc}"
            ) from exc

    return enhance_query


async def _build_cli_profile_labeler(
    *, config_manager: Any, telemetry_manager: Any, tenant_id: str
):
    """Build a tenant-bound production ProfileSelectionAgent callback."""
    from cogniverse_agents.profile_selection_agent import (
        ProfileSelectionAgent,
        ProfileSelectionDeps,
        ProfileSelectionInput,
    )

    def build_agent():
        agent = ProfileSelectionAgent(deps=ProfileSelectionDeps(available_profiles=[]))
        agent.telemetry_manager = telemetry_manager
        agent._config_manager = config_manager
        agent._artifact_tenant_id = tenant_id
        agent._load_artifact()
        return agent

    try:
        agent = await asyncio.to_thread(build_agent)
    except Exception as exc:
        raise RuntimeError(
            "Profile selection agent initialization failed for "
            f"tenant={tenant_id!r}: {exc}"
        ) from exc

    async def label_profile(
        query: str, available_profiles: list[str], request_tenant_id: str
    ):
        if request_tenant_id != tenant_id:
            raise ValueError(
                "Profile selection agent tenant mismatch: "
                f"configured={tenant_id!r} requested={request_tenant_id!r}"
            )
        try:
            return await agent.process(
                ProfileSelectionInput(
                    query=query,
                    available_profiles=list(available_profiles),
                    tenant_id=request_tenant_id,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "Profile selection agent failed for "
                f"tenant={request_tenant_id!r} query={query!r}: {exc}"
            ) from exc

    return label_profile


async def run_synthetic_generation(
    tenant_id: str,
    optimizer_types: list[str] | None = None,
    count: int = 50,
    telemetry_otlp_endpoint: str | None = None,
) -> dict:
    """Generate synthetic training data for optimizer types.

    Uses SyntheticDataService to create training examples, then persists
    them as pending review batches for later human approval.
    """
    from cogniverse_core.common.tenant_utils import require_tenant_id
    from cogniverse_foundation.config.utils import (
        create_default_config_manager,
        get_config,
    )
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager
    from cogniverse_synthetic.registry import (
        APPROVED_TRAINING_AGENT_BY_OPTIMIZER,
    )

    tenant_id = require_tenant_id(tenant_id, source="run_synthetic_generation")

    if optimizer_types is None:
        optimizer_types = list(APPROVED_TRAINING_AGENT_BY_OPTIMIZER)
    unsupported_types = [
        optimizer_type
        for optimizer_type in optimizer_types
        if optimizer_type not in APPROVED_TRAINING_AGENT_BY_OPTIMIZER
    ]
    if unsupported_types:
        raise ValueError(
            "synthetic optimizer types have no approved training-data consumer: "
            f"{unsupported_types}"
        )

    logger.info(
        "Starting synthetic generation for tenant=%s types=%s count=%d",
        tenant_id,
        optimizer_types,
        count,
    )

    config_manager = create_default_config_manager()
    config = get_config(tenant_id=tenant_id, config_manager=config_manager)

    from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config

    try:
        synthetic_runtime_config = parse_synthetic_runtime_config(
            config,
            tenant_id=tenant_id,
        )
    except ValueError as exc:
        error = str(exc)
        logger.error("Synthetic configuration rejected: %s", error)
        results = {
            optimizer_type: {"status": "failed", "error": error}
            for optimizer_type in optimizer_types
        }
        return {"status": "failed", "results": results}

    telemetry_manager = get_telemetry_manager(otlp_endpoint=telemetry_otlp_endpoint)
    entity_extractor = None
    routing_decider = None
    query_enhancer = None
    profile_labeler = None
    entity_dependent_types = {"routing", "entity_extraction"}
    results = {}
    if set(optimizer_types) & entity_dependent_types:
        try:
            entity_extractor = await _build_cli_entity_extractor(
                config_manager=config_manager,
                telemetry_manager=telemetry_manager,
                tenant_id=tenant_id,
            )
        except (RuntimeError, ValueError) as exc:
            error = str(exc)
            logger.error("Synthetic entity extraction unavailable: %s", error)
            results.update(
                {
                    optimizer_type: {"status": "failed", "error": error}
                    for optimizer_type in optimizer_types
                    if optimizer_type in entity_dependent_types
                }
            )

    if "routing" in optimizer_types and "routing" not in results:
        try:
            routing_decider = await _build_cli_routing_decider(
                config_manager=config_manager,
                telemetry_manager=telemetry_manager,
                tenant_id=tenant_id,
            )
        except (RuntimeError, ValueError) as exc:
            error = str(exc)
            logger.error("Synthetic routing unavailable: %s", error)
            results["routing"] = {"status": "failed", "error": error}

    if "query_enhancement" in optimizer_types:
        try:
            query_enhancer = await _build_cli_query_enhancer(
                config_manager=config_manager,
                telemetry_manager=telemetry_manager,
                tenant_id=tenant_id,
            )
        except (RuntimeError, ValueError) as exc:
            error = str(exc)
            logger.error("Synthetic query enhancement unavailable: %s", error)
            results["query_enhancement"] = {"status": "failed", "error": error}

    if "profile" in optimizer_types:
        try:
            profile_labeler = await _build_cli_profile_labeler(
                config_manager=config_manager,
                telemetry_manager=telemetry_manager,
                tenant_id=tenant_id,
            )
        except (RuntimeError, ValueError) as exc:
            error = str(exc)
            logger.error("Synthetic profile selection unavailable: %s", error)
            results["profile"] = {"status": "failed", "error": error}

    # Synthetic generators that wrap DSPy modules need an LM scoped to this
    # async task. A global DSPy configuration can only be changed by its owner
    # task and would also leak the tenant's binding into concurrent runs.
    import dspy

    from cogniverse_foundation.config.llm_factory import create_dspy_lm

    llm_endpoint = config.get_llm_config().primary
    synthetic_lm = create_dspy_lm(llm_endpoint)

    for opt_type in optimizer_types:
        if opt_type in results:
            continue
        try:
            from pathlib import Path

            from cogniverse_core.registries.backend_registry import BackendRegistry
            from cogniverse_core.schemas.filesystem_loader import (
                FilesystemSchemaLoader,
            )
            from cogniverse_synthetic.schemas import SyntheticDataRequest
            from cogniverse_synthetic.service import SyntheticDataService

            # BackendRegistry is a singleton — its __new__ takes no args.
            # get_search_backend is the public accessor; tenant isolation
            # is per-query via tenant_id in query_dict, so we don't pass
            # tenant_id here. Backend name comes from the resolved
            # backend config (defaults to "vespa"). schema_loader is
            # required for backend init — match what the ingestion v2
            # worker does (see ingestion_worker/worker.py).
            schemas_dir = Path(
                os.environ.get("COGNIVERSE_SCHEMAS_DIR", "configs/schemas")
            )
            try:
                registry = BackendRegistry()
                backend = registry.get_search_backend(
                    name=synthetic_runtime_config.backend_config.backend_type,
                    config_manager=config_manager,
                    schema_loader=FilesystemSchemaLoader(schemas_dir),
                )
            except Exception as exc:
                raise RuntimeError(
                    "Synthetic backend access failed for "
                    f"tenant={tenant_id!r} "
                    "backend="
                    f"{synthetic_runtime_config.backend_config.backend_type!r}: {exc}"
                ) from exc

            service = SyntheticDataService(
                backend=backend,
                config_manager=config_manager,
                backend_config=synthetic_runtime_config.backend_config,
                generator_config=synthetic_runtime_config.generator_config,
                agents_config=synthetic_runtime_config.agents_config,
                entity_extractor=entity_extractor,
                routing_decider=routing_decider,
                query_enhancer=query_enhancer,
                profile_labeler=profile_labeler,
            )

            request = SyntheticDataRequest(
                optimizer=opt_type,
                count=count,
                tenant_id=tenant_id,
            )
            with dspy.context(lm=synthetic_lm):
                response = await service.generate(request)

            if response.data:
                from cogniverse_agents.approval.approval_storage import (
                    ApprovalStorageImpl,
                )
                from cogniverse_core.approval.interfaces import (
                    ApprovalBatch,
                    ApprovalStatus,
                    ReviewItem,
                )
                from cogniverse_synthetic.approval.confidence_extractor import (
                    SyntheticDataConfidenceExtractor,
                )

                system_config = config_manager.get_system_config()
                if not system_config.redis_url:
                    raise ValueError(
                        "redis_url is required to persist synthetic review batches"
                    )
                grpc_endpoint = system_config.telemetry_collector_endpoint
                if not grpc_endpoint.startswith("http"):
                    grpc_endpoint = f"http://{grpc_endpoint}"
                storage = ApprovalStorageImpl(
                    grpc_endpoint=grpc_endpoint,
                    http_endpoint=system_config.telemetry_url,
                    tenant_id=tenant_id,
                    telemetry_manager=telemetry_manager,
                    redis_url=system_config.redis_url,
                )
                batch_id = f"synthetic_{opt_type}_{uuid.uuid4().hex}"
                agent_type = APPROVED_TRAINING_AGENT_BY_OPTIMIZER[opt_type]
                confidence_extractor = SyntheticDataConfidenceExtractor()
                review_items = [
                    ReviewItem(
                        item_id=f"{batch_id}_{index}",
                        data=dict(item),
                        confidence=confidence_extractor.extract(item),
                        status=ApprovalStatus.PENDING_REVIEW,
                        metadata={
                            "agent_type": agent_type,
                            "optimizer_type": opt_type,
                            "synthetic": True,
                        },
                    )
                    for index, item in enumerate(response.data)
                ]
                batch = ApprovalBatch(
                    batch_id=batch_id,
                    items=review_items,
                    context={
                        "tenant_id": tenant_id,
                        "agent_type": agent_type,
                        "optimizer": opt_type,
                        "purpose": "optimizer_training",
                    },
                )
                persisted_batch_id = await storage.save_batch(batch)
                results[opt_type] = {
                    "status": "success",
                    "examples_generated": len(review_items),
                    "batch_id": persisted_batch_id,
                    "pending_review": len(review_items),
                }
            else:
                results[opt_type] = {"status": "no_data", "examples_generated": 0}

            logger.info(
                "Generated %d synthetic examples for %s",
                len(response.data),
                opt_type,
            )

        except Exception as e:
            logger.error("Synthetic generation failed for %s: %s", opt_type, e)
            results[opt_type] = {"status": "failed", "error": str(e)}

    return {
        "status": _synthetic_aggregate_status(results),
        "results": results,
    }


async def run_ab_compare(
    *,
    tenant_id: str,
    queries_dataset: str,
    judge_substring: Optional[str] = None,
    rlm_max_iterations: int = 10,
    rlm_max_llm_calls: int = 30,
    telemetry_otlp_endpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """run RLMABRunner over a Phoenix queries dataset.

    The dataset must contain rows with at least ``query`` and ``context``
    columns (Phoenix wraps these under ``input``/``output`` dicts when
    saved with input_keys; we flatten on load). For each row we run both
    arms and emit a Phoenix span (``rlm.ab_compare``) with the harness's
    ``to_telemetry_dict()`` as attributes — that's what the dashboard
    tile will read.

    Optional ``judge_substring`` enables a deterministic substring-match
    judge (1.0 if the substring appears in the answer, 0.0 otherwise).
    Real eval-time judges should be wired by the caller; this is the
    minimum viable judge for getting a `judge_delta` populated in CI.

    Returns aggregated stats so the operator can see per-dataset trends
    without tailing Phoenix.
    """
    from opentelemetry import trace
    from phoenix.client import Client as PhoenixSyncClient

    from cogniverse_agents.inference.ab_harness import RLMABRunner
    from cogniverse_foundation.config.utils import (
        create_default_config_manager,
        get_config,
    )

    config_manager = create_default_config_manager()
    cfg = get_config(tenant_id=tenant_id, config_manager=config_manager)
    llm_primary = cfg.get_llm_config().primary

    phoenix_http = os.environ.get("PHOENIX_HTTP_ENDPOINT", "http://localhost:6006")
    sync_client = PhoenixSyncClient(base_url=phoenix_http)

    try:
        dataset = sync_client.datasets.get_dataset(dataset=queries_dataset)
        df = dataset.to_dataframe()
    except Exception as exc:
        logger.error("ab-compare: dataset %r not loadable: %s", queries_dataset, exc)
        return {"status": "failed", "error": str(exc)}

    # Flatten input/output dicts the way run_triggered_optimization does.
    if "input" in df.columns and "query" not in df.columns:
        import pandas as _pd

        flat = []
        for _, row in df.iterrows():
            inp = row.get("input", {}) or {}
            out = row.get("output", {}) or {}
            flat.append({**inp, **out})
        df = _pd.DataFrame(flat)

    if "query" not in df.columns or "context" not in df.columns:
        return {
            "status": "failed",
            "error": (
                f"dataset {queries_dataset!r} must expose 'query' and 'context' "
                f"columns; got {list(df.columns)}"
            ),
        }

    judge = None
    if judge_substring:
        token = judge_substring

        def _substring_judge(_q: str, _ctx: str, ans: str) -> float:
            return 1.0 if token.lower() in (ans or "").lower() else 0.0

        judge = _substring_judge

    runner = RLMABRunner(
        llm_config=llm_primary,
        judge=judge,
        rlm_max_iterations=rlm_max_iterations,
        rlm_max_llm_calls=rlm_max_llm_calls,
        tenant_id=tenant_id,
        config_manager=config_manager,
    )

    # Emit the comparison span through the tenant's own provider so it lands
    # in that tenant's project -- not the global no-op provider, which
    # discards it. Falls back to the global tracer only when telemetry is
    # disabled and no tenant tracer exists.
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    telemetry_manager = get_telemetry_manager(otlp_endpoint=telemetry_otlp_endpoint)
    tracer = telemetry_manager._get_tracer_for_project(
        tenant_id, None
    ) or trace.get_tracer("cogniverse.ab_compare")

    rows: list = []
    for _, r in df.iterrows():
        query = str(r["query"])
        context = str(r["context"])
        try:
            result = runner.run(query=query, context=context)
        except Exception as exc:
            logger.warning("ab-compare: arm failure on query=%r: %s", query[:60], exc)
            continue

        # Emit a Phoenix span with the comparison attributes — the dashboard
        # tile (when added) will aggregate over these.
        with tracer.start_as_current_span("rlm.ab_compare") as span:
            for k, v in result.to_telemetry_dict().items():
                if v is None:
                    continue
                span.set_attribute(f"openinference.{k}", v)
            span.set_attribute("openinference.tenant_id", tenant_id)
            span.set_attribute("openinference.queries_dataset", queries_dataset)
        rows.append(result)

    # This is a short-lived job: flush batched spans before returning so the
    # emitted rlm.ab_compare spans reach Phoenix rather than being dropped on
    # process exit.
    telemetry_manager.force_flush()

    if not rows:
        return {
            "status": "failed",
            "error": "no rows produced both arms successfully",
            "queries_dataset": queries_dataset,
        }

    n = len(rows)
    avg_latency_delta = sum(r.comparison.latency_delta_ms for r in rows) / n
    avg_tokens_delta = sum(r.comparison.tokens_delta for r in rows) / n
    judge_deltas = [
        r.comparison.judge_delta for r in rows if r.comparison.judge_delta is not None
    ]
    avg_judge_delta = sum(judge_deltas) / len(judge_deltas) if judge_deltas else None
    fallback_count = sum(1 for r in rows if r.with_rlm.was_fallback)

    summary = {
        "status": "ok",
        "queries_dataset": queries_dataset,
        "tenant_id": tenant_id,
        "rows_compared": n,
        "avg_latency_delta_ms": avg_latency_delta,
        "avg_tokens_delta": avg_tokens_delta,
        "avg_judge_delta": avg_judge_delta,
        "rlm_fallback_rate": fallback_count / n,
        "ab_ids": [r.ab_id for r in rows],
    }
    logger.info("A/B compare complete: %s", summary)
    return summary


def run_egress_netpol(
    *,
    policy_dir: str,
    output_dir: str,
    service_map: Dict[str, str],
    namespace: str = "cogniverse",
    pod_app_label: str = "cogniverse",
    helm_conditional: Optional[str] = None,
    unified_pod_selector: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """emit k8s NetworkPolicy CRDs from agent policy YAMLs.

    Reads every YAML in ``policy_dir`` whose
    ``network_policies.deny_all_other`` is true, translates the egress
    list into NetworkPolicy egress rules.

    Two emit modes:
      * **per-agent (default)**: writes one NetworkPolicy per agent under
        ``output_dir/<agent>-egress-netpol.yaml`` selecting on
        ``app=<pod_app_label>, cogniverse-agent=<agent>``. Use this when
        each agent runs in its own Deployment so the labels match.
      * **unified-runtime** (``unified_pod_selector`` set): emits ONE
        NetworkPolicy named ``runtime-egress-netpol.yaml`` whose
        ``spec.egress`` is the de-duplicated UNION of every agent's
        allowed destinations and whose ``spec.podSelector.matchLabels``
        come from ``unified_pod_selector``. Use this when every agent
        runs inside a single shared runtime pod (the default helm chart
        topology) — per-agent L4 enforcement is impossible there, but
        cluster-wide deny-all-other-egress with a union allowlist is
        still real defense-in-depth on top of the application-layer
        OpenShell sandbox enforcement.

    Why this exists: the agent policy YAMLs declare per-agent egress
    constraints (Vespa for SearchAgent, the configured LM for SummarizerAgent,
    etc.) but in-process Python enforcement is fundamentally weak — a
    compromised process can ``socket.connect`` past any httpx wrapper.
    NetworkPolicy is enforced in the kernel by the cluster's CNI
    plugin (Cilium / Calico / etc.), so it's process-bypass-proof and
    independent of which HTTP library the agent uses.

    Args:
        policy_dir: Where the agent policy YAMLs live (default
            ``configs/agent_policies/``).
        output_dir: Where to write the generated NetworkPolicy YAMLs.
            Operators check these into the helm chart's
            ``templates/agent-egress/`` so helm applies them at
            deploy time.
        service_map: Logical service name → ``namespace/service-name:port``
            mapping (e.g. ``vespa=cogniverse/vespa-service:8080``). The
            policy YAML's ``localhost:N`` entries are matched by port
            against this map's values; the resulting NetworkPolicy uses
            podSelectors that target those services.
        namespace: k8s namespace the NetworkPolicy lives in.
        pod_app_label: ``app=`` label that selects cogniverse pods in
            per-agent mode. Ignored when ``unified_pod_selector`` is
            set.
        helm_conditional: Wrap each emitted YAML in a helm ``{{- if X }}``
            … ``{{- end }}`` so a values flag toggles application.
        unified_pod_selector: When provided, emit a single union policy
            selecting on these labels (e.g.
            ``{"app.kubernetes.io/component": "runtime"}``).

    Returns a summary dict suitable for the CLI's stdout JSON.
    """
    from pathlib import Path as _Path

    import yaml as _yaml

    out = _Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build a port → (svc-namespace, svc-name, svc-port) lookup so the
    # localhost:N entries in the YAMLs can be resolved to the right
    # in-cluster service.
    port_to_service: Dict[int, Dict[str, Any]] = {}
    for logical, target in service_map.items():
        # Format: "namespace/service-name:port"
        try:
            ns_part, port_str = target.rsplit(":", 1)
            svc_namespace, svc_name = ns_part.split("/", 1)
            port = int(port_str)
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"--service-map {logical}={target!r} is malformed; expected "
                "'namespace/service:port'"
            ) from exc
        port_to_service[port] = {
            "logical": logical,
            "namespace": svc_namespace,
            "service": svc_name,
            "port": port,
        }

    written: List[str] = []
    skipped: List[Dict[str, str]] = []

    # First pass: read every eligible policy + collect (agent, [egress_rules]).
    per_agent_rules: List[tuple] = []  # (agent_name, [egress_rules])
    for yaml_path in sorted(_Path(policy_dir).glob("*.yaml")):
        agent_name = yaml_path.stem
        with open(yaml_path) as f:
            policy_blob = _yaml.safe_load(f) or {}

        netpols = policy_blob.get("network_policies") or {}
        if not netpols.get("deny_all_other"):
            skipped.append({"agent": agent_name, "reason": "deny_all_other not set"})
            continue

        egress_rules: List[Dict[str, Any]] = []
        unmapped_ports: List[int] = []
        for rule in netpols.get("egress") or []:
            port = int(rule.get("port", 0))
            svc = port_to_service.get(port)
            if svc is None:
                unmapped_ports.append(port)
                continue
            egress_rules.append(
                {
                    "to": [
                        {
                            "namespaceSelector": {
                                "matchLabels": {
                                    "kubernetes.io/metadata.name": svc["namespace"]
                                }
                            },
                            "podSelector": {"matchLabels": {"app": svc["service"]}},
                        }
                    ],
                    "ports": [
                        {
                            "port": svc["port"],
                            "protocol": str(rule.get("protocol", "tcp")).upper(),
                        }
                    ],
                }
            )

        if unmapped_ports:
            skipped.append(
                {
                    "agent": agent_name,
                    "reason": (
                        f"egress ports {sorted(set(unmapped_ports))} not in "
                        "--service-map"
                    ),
                }
            )
            continue

        # DNS is mandatory for any egress to resolve service names.
        egress_rules.append(
            {
                "to": [
                    {
                        "namespaceSelector": {
                            "matchLabels": {
                                "kubernetes.io/metadata.name": "kube-system"
                            }
                        },
                        "podSelector": {"matchLabels": {"k8s-app": "kube-dns"}},
                    }
                ],
                "ports": [
                    {"port": 53, "protocol": "UDP"},
                    {"port": 53, "protocol": "TCP"},
                ],
            }
        )

        per_agent_rules.append((agent_name, egress_rules))

    # Second pass: emit either one union policy (unified mode) or one
    # policy per agent (per-agent mode).
    if unified_pod_selector:
        # De-duplicate egress rules across agents — two agents that both
        # need DNS or both need vespa shouldn't produce duplicate yaml
        # entries.
        union: List[Dict[str, Any]] = []
        seen_keys = set()
        for _agent, rules in per_agent_rules:
            for rule in rules:
                key = _yaml.safe_dump(rule, sort_keys=True)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                union.append(rule)

        netpol_doc = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "cogniverse-runtime-egress",
                "namespace": namespace,
                "labels": {"cogniverse-component": "runtime"},
            },
            "spec": {
                "podSelector": {"matchLabels": dict(unified_pod_selector)},
                "policyTypes": ["Egress"],
                "egress": union,
            },
        }
        out_path = out / "runtime-egress-netpol.yaml"
        with open(out_path, "w") as f:
            if helm_conditional:
                f.write("{{- if " + helm_conditional + " }}\n")
            _yaml.safe_dump(netpol_doc, f, sort_keys=False, default_flow_style=False)
            if helm_conditional:
                f.write("{{- end }}\n")
        written.append(str(out_path))
        logger.info(
            "Wrote unified NetworkPolicy → %s (%d egress rules from %d agents)",
            out_path,
            len(union),
            len(per_agent_rules),
        )
    else:
        for agent_name, egress_rules in per_agent_rules:
            netpol_doc = {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "NetworkPolicy",
                "metadata": {
                    "name": f"cogniverse-{agent_name.replace('_', '-')}-egress",
                    "namespace": namespace,
                    "labels": {"cogniverse-agent": agent_name},
                },
                "spec": {
                    "podSelector": {
                        "matchLabels": {
                            "app": pod_app_label,
                            "cogniverse-agent": agent_name,
                        }
                    },
                    "policyTypes": ["Egress"],
                    "egress": egress_rules,
                },
            }

            out_path = out / f"{agent_name}-egress-netpol.yaml"
            with open(out_path, "w") as f:
                if helm_conditional:
                    f.write("{{- if " + helm_conditional + " }}\n")
                _yaml.safe_dump(
                    netpol_doc, f, sort_keys=False, default_flow_style=False
                )
                if helm_conditional:
                    f.write("{{- end }}\n")
            written.append(str(out_path))
            logger.info(
                "Wrote NetworkPolicy for %s → %s (%d egress rules)",
                agent_name,
                out_path,
                len(egress_rules),
            )

    return {
        "status": "ok",
        "policy_dir": str(policy_dir),
        "output_dir": str(output_dir),
        "written": written,
        "skipped": skipped,
        "service_map": service_map,
        "mode": "unified" if unified_pod_selector else "per-agent",
    }


def _build_phoenix_provider_for_cli(tenant_id: str):
    """Construct a PhoenixProvider directly from env vars for CLI runs.

    Operators (and integration tests) set ``PHOENIX_HTTP_ENDPOINT`` and
    ``PHOENIX_GRPC_ENDPOINT`` to point at the Phoenix instance the CLI
    should talk to. We build the provider directly here rather than
    going through ``get_telemetry_manager()`` so a CLI invocation can
    target a specific Phoenix without the global telemetry config (which
    is loaded from ConfigManager and pinned to the cluster's primary).
    """
    from cogniverse_telemetry_phoenix.provider import PhoenixProvider

    http_endpoint = os.environ.get("PHOENIX_HTTP_ENDPOINT", "http://localhost:6006")
    grpc_endpoint = os.environ.get("PHOENIX_GRPC_ENDPOINT", "localhost:4317")
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": http_endpoint,
            "grpc_endpoint": grpc_endpoint,
        }
    )
    return provider


async def run_rollback(
    *,
    tenant_id: str,
    agent_type: str,
    prompts_version: Optional[int] = None,
    demos_version: Optional[int] = None,
) -> Dict[str, Any]:
    """restore active artefacts to a previously-snapshotted version.

    Wraps :meth:`ArtifactManager.rollback_to_version` so an operator can
    run e.g. ``cogniverse-optim --mode rollback --tenant-id acme
    --agent search_agent --prompts-version 3``.

    The current active artefacts are themselves snapshotted before the
    rollback (the manager method does this) so the rollback is itself
    reversible — the returned ``backup_versions`` dict contains the
    versions you'd pass to ``rollback`` again to undo this operation.
    """
    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

    telemetry_provider = _build_phoenix_provider_for_cli(tenant_id)
    am = ArtifactManager(telemetry_provider, tenant_id)
    logger.info(
        "Rollback: tenant=%s agent=%s prompts_v=%s demos_v=%s",
        tenant_id,
        agent_type,
        prompts_version,
        demos_version,
    )
    summary = await am.rollback_to_version(
        agent_type=agent_type,
        prompts_version=prompts_version,
        demos_version=demos_version,
    )
    logger.info("Rollback complete: %s", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the optimization CLI argument parser.

    Exposed separately from ``main`` so tests assert against the REAL parser
    (its modes, defaults, required flags) instead of a hand-built copy that
    can silently drift from production.
    """
    parser = argparse.ArgumentParser(description="Cogniverse Optimization CLI")
    parser.add_argument(
        "--mode",
        choices=[
            "cleanup",
            "triggered",
            "simba",
            "workflow",
            "gateway-thresholds",
            "online-routing-eval",
            "online-eval",
            "profile",
            "entity-extraction",
            "synthetic",
            "rollback",
            "ab-compare",
            "egress-netpol",
            "monthly-reports",
        ],
        required=True,
    )
    # monthly-reports writes its JSON output here for a follow-up
    # workflow step to upload via mc. Inside the cron pod this is a
    # mounted emptyDir / PVC; local CLI runs default to ./reports.
    parser.add_argument(
        "--reports-output-dir",
        default="./reports",
        help="Output directory for monthly-reports mode (default: ./reports)",
    )
    # --tenant-id is required for most modes; cleanup + monthly-reports
    # are the exceptions and run globally when omitted, so the
    # daily-cleanup / monthly-reports CronWorkflows (no tenant) don't
    # exit 2 on argparse.
    parser.add_argument(
        "--tenant-id",
        default=None,
        help=(
            "Tenant ID (required for all modes except --mode cleanup / "
            "--mode monthly-reports)"
        ),
    )
    parser.add_argument(
        "--agents",
        help="Comma-separated agent names for triggered mode",
    )
    parser.add_argument(
        "--trigger-dataset",
        help="Phoenix dataset name containing trigger payload",
    )
    parser.add_argument("--log-retention-days", type=int, default=7)
    parser.add_argument("--memory-retention-days", type=int, default=30)
    # rollback mode args. Operators run e.g.
    #   cogniverse-optim --mode rollback --tenant-id acme \
    #       --agent search_agent --prompts-version 3
    # to restore search_agent's active prompts to v3. Demos rollback is
    # independent so a caller can roll back just one or both.
    parser.add_argument(
        "--agent",
        help="Single agent name (rollback mode)",
    )
    parser.add_argument(
        "--prompts-version",
        type=int,
        help="Prompts version to restore (rollback mode)",
    )
    parser.add_argument(
        "--demos-version",
        type=int,
        help="Demonstrations version to restore (rollback mode)",
    )
    # ab-compare mode args. Operators run e.g.
    #   cogniverse-optim --mode ab-compare --tenant-id acme \
    #       --queries-dataset golden_eval_v1 [--judge-substring 'Paris']
    parser.add_argument(
        "--queries-dataset",
        help="Phoenix dataset of (query, context) rows (ab-compare mode)",
    )
    parser.add_argument(
        "--judge-substring",
        help="Optional substring judge for ab-compare mode (1.0 if present)",
    )
    parser.add_argument(
        "--rlm-max-iterations",
        type=int,
        default=10,
        help="Per-arm RLM iteration cap (ab-compare mode)",
    )
    parser.add_argument(
        "--rlm-max-llm-calls",
        type=int,
        default=30,
        help="Per-arm RLM total LLM call cap (ab-compare mode)",
    )
    # egress-netpol mode args. Generates k8s NetworkPolicy CRDs from
    # the agent policy YAMLs in configs/agent_policies/. Operators run e.g.
    #   cogniverse-optim --mode egress-netpol \
    #       --policy-dir configs/agent_policies/ \
    #       --output-dir charts/cogniverse/templates/networkpolicies/ \
    #       --service-map vespa=cogniverse/vespa-service:8080 \
    #       --service-map llm=cogniverse/llm-service:11434
    parser.add_argument(
        "--policy-dir",
        default="configs/agent_policies",
        help="Source directory of agent policy YAMLs (egress-netpol mode)",
    )
    parser.add_argument(
        "--output-dir",
        help="Where to write generated NetworkPolicy YAMLs (egress-netpol mode)",
    )
    parser.add_argument(
        "--service-map",
        action="append",
        default=[],
        help=(
            "Logical service mapping `name=namespace/service:port` "
            "(repeatable; egress-netpol mode)"
        ),
    )
    parser.add_argument(
        "--netpol-namespace",
        default="cogniverse",
        help="k8s namespace for the generated NetworkPolicies",
    )
    parser.add_argument(
        "--netpol-app-label",
        default="cogniverse",
        help="Pod `app=` label that scopes the policies to cogniverse pods",
    )
    parser.add_argument(
        "--helm-conditional",
        default=None,
        help=(
            "When set, wrap each emitted YAML in `{{- if <expr> }}` ... "
            "`{{- end }}` so the helm chart's values.yaml flag toggles "
            "whether the NetworkPolicy applies. Example: "
            "`.Values.networkPolicy.agentEgress.enabled`."
        ),
    )
    parser.add_argument(
        "--unified-pod-selector",
        action="append",
        default=[],
        help=(
            "key=value (repeatable). When set, emit ONE NetworkPolicy "
            "selecting on these labels with the de-duplicated UNION of "
            "every agent's egress destinations. Use this for the "
            "default unified-runtime topology where all agents run in "
            "the same pod. Example: "
            "`--unified-pod-selector app.kubernetes.io/component=runtime`. "
            "When omitted, emits one NetworkPolicy per agent "
            "(per-agent-pod topology)."
        ),
    )
    parser.add_argument(
        "--lookback-hours",
        type=float,
        default=24.0,
        help="Hours of span history to analyze. Accepts fractions (e.g. 0.1 "
        "= 6 minutes) so e2e tests can scope to the current fixture "
        "window without picking up spans from earlier runs.",
    )
    parser.add_argument(
        "--embedder-url",
        default=None,
        help="DenseOn embeddings endpoint for training selection.",
    )
    return parser


def _run_failed(result: Any) -> bool:
    """Whether a mode result reports failure — drives the exit code, which
    is the only success signal Argo sees for a workflow step.

    A top-level status never masks a failed requested result. Batch and cleanup
    modes also encode per-entry failure as a nested ``{"status": ...}`` dict,
    a ``{"failed": ...}`` dict, or a free-form ``"failed: ..."`` /
    ``"error: ..."`` string. Recurse so any failure shape at any depth fails
    the run.
    """
    if isinstance(result, str):
        marker = result.strip().lower()
        return marker.startswith("failed:") or marker.startswith("error:")
    if isinstance(result, (list, tuple)):
        return any(_run_failed(item) for item in result)
    if not isinstance(result, dict):
        return False
    if result.get("status") in ("failed", "error"):
        return True
    if result.get("failed"):
        return True
    return any(_run_failed(value) for value in result.values())


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from cogniverse_runtime.entrypoint_env import resolve_library_env_defaults
    from cogniverse_runtime.inference_services import parse_inference_service_urls

    runtime_env = resolve_library_env_defaults()
    telemetry_otlp_endpoint = runtime_env["telemetry_otlp_endpoint"]
    inference_service_urls = parse_inference_service_urls(
        os.environ.get("INFERENCE_SERVICE_URLS")
    )
    embedder_url = args.embedder_url
    if embedder_url is None and inference_service_urls is not None:
        embedder_url = inference_service_urls.get("denseon")

    if (
        args.mode not in ("cleanup", "egress-netpol", "monthly-reports")
        and not args.tenant_id
    ):
        parser.error(f"--tenant-id is required for mode={args.mode!r}")
    if args.tenant_id:
        # One canonical tenant everywhere — span projects and artifacts are
        # keyed by the canonical form, so the compile must read the same
        # project the runtime writes.
        from cogniverse_core.common.tenant_utils import canonical_tenant_id

        args.tenant_id = canonical_tenant_id(args.tenant_id)

    # Keep stdout reserved for the final JSON document.
    with _redirect_stdout_to_stderr():
        if args.mode == "cleanup":
            result = asyncio.run(
                run_cleanup(
                    args.tenant_id, args.log_retention_days, args.memory_retention_days
                )
            )
        elif args.mode == "monthly-reports":
            result = asyncio.run(
                run_monthly_reports(
                    output_dir=args.reports_output_dir,
                    lookback_hours=args.lookback_hours,
                    telemetry_otlp_endpoint=telemetry_otlp_endpoint,
                )
            )
        elif args.mode == "triggered":
            if not args.agents or not args.trigger_dataset:
                parser.error(
                    "--agents and --trigger-dataset are required for triggered mode"
                )
            agents = [a.strip() for a in args.agents.split(",")]
            result = asyncio.run(
                run_triggered_optimization(
                    tenant_id=args.tenant_id,
                    agents=agents,
                    trigger_dataset=args.trigger_dataset,
                    telemetry_otlp_endpoint=telemetry_otlp_endpoint,
                )
            )
        elif args.mode == "simba":
            result = asyncio.run(
                run_simba_optimization(
                    tenant_id=args.tenant_id,
                    lookback_hours=args.lookback_hours,
                    telemetry_otlp_endpoint=telemetry_otlp_endpoint,
                    embedder_url=embedder_url,
                )
            )
        elif args.mode == "workflow":
            result = asyncio.run(
                run_workflow_optimization(
                    tenant_id=args.tenant_id,
                    lookback_hours=args.lookback_hours,
                    telemetry_otlp_endpoint=telemetry_otlp_endpoint,
                )
            )
        elif args.mode == "gateway-thresholds":
            result = asyncio.run(
                run_gateway_thresholds_optimization(
                    tenant_id=args.tenant_id,
                    lookback_hours=args.lookback_hours,
                    telemetry_otlp_endpoint=telemetry_otlp_endpoint,
                )
            )
        elif args.mode == "online-routing-eval":
            result = asyncio.run(
                run_online_routing_evaluation(
                    tenant_id=args.tenant_id,
                    lookback_hours=args.lookback_hours,
                    telemetry_otlp_endpoint=telemetry_otlp_endpoint,
                )
            )
        elif args.mode == "online-eval":
            result = asyncio.run(
                run_online_evaluation(
                    tenant_id=args.tenant_id,
                    lookback_hours=args.lookback_hours,
                    telemetry_otlp_endpoint=telemetry_otlp_endpoint,
                )
            )
        elif args.mode == "profile":
            result = asyncio.run(
                run_profile_optimization(
                    tenant_id=args.tenant_id,
                    lookback_hours=args.lookback_hours,
                    telemetry_otlp_endpoint=telemetry_otlp_endpoint,
                    embedder_url=embedder_url,
                )
            )
        elif args.mode == "entity-extraction":
            result = asyncio.run(
                run_entity_extraction_optimization(
                    tenant_id=args.tenant_id,
                    lookback_hours=args.lookback_hours,
                    telemetry_otlp_endpoint=telemetry_otlp_endpoint,
                    embedder_url=embedder_url,
                )
            )
        elif args.mode == "rollback":
            if not args.agent or (
                args.prompts_version is None and args.demos_version is None
            ):
                parser.error(
                    "--agent is required for rollback mode, plus at least one of "
                    "--prompts-version or --demos-version"
                )
            result = asyncio.run(
                run_rollback(
                    tenant_id=args.tenant_id,
                    agent_type=args.agent,
                    prompts_version=args.prompts_version,
                    demos_version=args.demos_version,
                )
            )
        elif args.mode == "egress-netpol":
            if not args.output_dir:
                parser.error("--output-dir is required for egress-netpol mode")
            if not args.service_map:
                parser.error(
                    "at least one --service-map is required for egress-netpol mode"
                )
            # Parse `name=ns/svc:port` pairs into a dict.
            sm: Dict[str, str] = {}
            for pair in args.service_map:
                if "=" not in pair:
                    parser.error(f"--service-map {pair!r} missing '=' separator")
                k, v = pair.split("=", 1)
                sm[k.strip()] = v.strip()
            unified_selectors: Optional[Dict[str, str]] = None
            if args.unified_pod_selector:
                unified_selectors = {}
                for pair in args.unified_pod_selector:
                    if "=" not in pair:
                        parser.error(
                            f"--unified-pod-selector {pair!r} missing '=' separator"
                        )
                    k, v = pair.split("=", 1)
                    unified_selectors[k.strip()] = v.strip()
            result = run_egress_netpol(
                policy_dir=args.policy_dir,
                output_dir=args.output_dir,
                service_map=sm,
                namespace=args.netpol_namespace,
                pod_app_label=args.netpol_app_label,
                helm_conditional=args.helm_conditional,
                unified_pod_selector=unified_selectors,
            )
        elif args.mode == "ab-compare":
            if not args.queries_dataset:
                parser.error("--queries-dataset is required for ab-compare mode")
            result = asyncio.run(
                run_ab_compare(
                    tenant_id=args.tenant_id,
                    queries_dataset=args.queries_dataset,
                    judge_substring=args.judge_substring,
                    rlm_max_iterations=args.rlm_max_iterations,
                    rlm_max_llm_calls=args.rlm_max_llm_calls,
                    telemetry_otlp_endpoint=telemetry_otlp_endpoint,
                )
            )
        elif args.mode == "synthetic":
            from cogniverse_synthetic.registry import (
                APPROVED_TRAINING_AGENT_BY_OPTIMIZER,
            )

            optimizer_types = list(APPROVED_TRAINING_AGENT_BY_OPTIMIZER)
            if args.agents:
                optimizer_types = [a.strip() for a in args.agents.split(",")]
            result = asyncio.run(
                run_synthetic_generation(
                    tenant_id=args.tenant_id,
                    optimizer_types=optimizer_types,
                    telemetry_otlp_endpoint=telemetry_otlp_endpoint,
                )
            )
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

    print(json.dumps(result, indent=2, default=str))
    sys.exit(1 if _run_failed(result) else 0)


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        # Configuration errors (e.g. BACKEND_URL unset) exit with a clean
        # one-line message instead of a traceback.
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
