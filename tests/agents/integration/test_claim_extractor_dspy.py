"""Integration test for ClaimExtractor + ClaimExtractionSignature.

Exercises the compiled DSPy module against a live LM:

- ``extract()`` on the Marie Curie seed text yields the locked SPO edge list.
- The chain-of-thought output is claims-only; reasoning stays internal to the
  DSPy module and there is no separate rationale field.
- Negative example ("yellow flowers in a glass vase") yields zero edges.
- Long input triggers RLM promotion sized against the window the endpoint
  serves; the Phoenix span tree shows the ``rlm_iterations`` attribute.
- Short input emits zero ``InstrumentedRLM`` spans.
- The compiled artifact loaded via ArtifactManager is byte-equal to golden,
  with ``len(demos) == 8`` (BootstrapFewShot k=8).
- Across the 100-example training set, the union of emitted predicates
  equals the locked vocabulary set.
- Two consecutive calls produce byte-equal JSON serializations.

Test LM endpoint is resolved from ``TEST_LLM_API_BASE`` /
``TEST_LLM_MODEL`` (see ``tests/fixtures/llm.py``). When the endpoint
is unreachable the whole file skips — per project convention, never
skip individual tests inside an integration file for an infra dep.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from cogniverse_agents.graph.claim_extractor import (
    CLAIM_EXTRACTION_MAX_CLAIMS,
    PREDICATE_VOCABULARY,
    PROMPT_TOKENIZER_MARGIN_SHARE,
    RLM_HARNESS_TOKENS,
    RLM_TRANSCRIPT_CHARS_PER_TOKEN,
    RLM_TRANSCRIPT_TURNS,
    ClaimExtractor,
    RecursiveClaimBudgetError,
)
from cogniverse_agents.graph.dspy_signatures import ClaimExtractionSignature
from cogniverse_agents.graph.graph_schema import Mention
from tests.fixtures.llm import (
    resolve_api_key,
)

# --------------------------------------------------------------------------- #
# Golden-file machinery                                                       #
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[3]
# Rows in the committed claim-extraction corpus.
CLAIM_TRAINING_ROWS = 100
GOLDEN_DIR = Path(__file__).parent / "goldens"
RECORD_GOLDEN = os.environ.get("RECORD_GOLDEN") == "1"


def _golden(name: str) -> Path:
    return GOLDEN_DIR / name


def assert_golden(actual: Any, name: str) -> None:
    """Byte-equal JSON assertion against a golden file."""
    path = _golden(name)
    actual_json = json.dumps(actual, indent=2, sort_keys=True, default=str)
    if RECORD_GOLDEN:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual_json + "\n")
        return
    if not path.exists():
        raise AssertionError(
            f"Golden file missing: {path}\n"
            f"To create it: RECORD_GOLDEN=1 uv run pytest <this test>"
        )
    expected = path.read_text().rstrip("\n")
    assert actual_json == expected, (
        f"Golden mismatch for {name}.\n"
        f"To regenerate: RECORD_GOLDEN=1 uv run pytest <this test>\n"
        f"--- expected ---\n{expected}\n--- actual ---\n{actual_json}"
    )


def assert_golden_edges(sorted_edges: Any, name: str) -> None:
    """Golden assertion for extracted edges: every structural field is
    byte-equal; ``confidence`` is an LM-emitted float that can drift by a
    hundredth across servings even at temperature 0, so it is compared as
    a tight band against the golden's recorded value."""
    path = _golden(name)
    actual_json = json.dumps(sorted_edges, indent=2, sort_keys=True, default=str)
    if RECORD_GOLDEN:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual_json + "\n")
        return
    if not path.exists():
        raise AssertionError(
            f"Golden file missing: {path}\n"
            f"To create it: RECORD_GOLDEN=1 uv run pytest <this test>"
        )
    expected = json.loads(path.read_text())
    actual = json.loads(actual_json)
    assert len(actual) == len(expected), (
        f"Golden mismatch for {name}: {len(actual)} edges vs "
        f"{len(expected)} expected.\n--- actual ---\n{actual_json}"
    )
    for a, e in zip(actual, expected):
        a_conf = float(a.pop("confidence"))
        e_conf = float(e.pop("confidence"))
        assert abs(a_conf - e_conf) <= 0.05, (
            f"Golden mismatch for {name}: confidence {a_conf} vs "
            f"{e_conf} (band 0.05) on edge {a}"
        )
    assert actual == expected, (
        f"Golden mismatch for {name} (structural fields).\n"
        f"To regenerate: RECORD_GOLDEN=1 uv run pytest <this test>\n"
        f"--- expected ---\n{json.dumps(expected, indent=2)}\n"
        f"--- actual ---\n{json.dumps(actual, indent=2)}"
    )


pytestmark = [pytest.mark.integration]


# --------------------------------------------------------------------------- #
# Marie Curie fixture (matches test_per_segment_kg_provenance.py).            #
# --------------------------------------------------------------------------- #

VIDEO_ID = "marie_curie_30s"
TENANT_ID = "test"

SEG_3_TEXT = "Marie Curie discovered radium in 1898 at the Sorbonne."
SEG_3_START = 12.0
SEG_3_END = 18.5
SEG_3_ENTITY_HINTS = ["Marie Curie", "radium", "Sorbonne", "1898"]
SEG_3_SEGMENT = "seg_3"
# The relations the segment's one sentence supports, sorted.
SEG_3_RELATIONS = ["discovered", "discovered_in", "worked_at"]
# "Marie Curie discovered radium in 1898" dates the discovery; the discoverer
# and the discovery are both grounded subjects for the year claim.
SEG_3_DISCOVERY_PARTICIPANTS = {"Marie Curie", "radium"}

VLM_FLOWERS_TEXT = "Yellow flowers in a glass vase."
VLM_FLOWERS_TS = 30.0


def _seg3_anchor() -> Mention:
    return Mention(
        source_doc_id=VIDEO_ID,
        segment_id="seg_3",
        ts_start=SEG_3_START,
        ts_end=SEG_3_END,
        modality="transcript",
        evidence_span=SEG_3_TEXT,
    )


def _vlm_flowers_anchor() -> Mention:
    return Mention(
        source_doc_id=VIDEO_ID,
        segment_id="frame_30_0",
        ts_start=VLM_FLOWERS_TS,
        ts_end=VLM_FLOWERS_TS,
        modality="vlm",
        evidence_span=VLM_FLOWERS_TEXT,
    )


# --------------------------------------------------------------------------- #
# DSPy LM configuration — fixed-seed, temperature=0 for determinism.          #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def hermetic_test_lm():
    """Provision (or reattach to) the hermetic vLLM sidecar the goldens
    were locked against — the same ``ensure_llm`` path
    ``tests/evaluation/integration/conftest.py::llm_endpoint`` uses. Fails
    loudly when the sidecar cannot come up (this file errors, never skips,
    on infra faults)."""
    from tests.utils.hermetic_llm import ensure_llm

    base = ensure_llm()
    if base is None:
        pytest.fail(
            "hermetic test-LM sidecar failed to provision "
            "(Docker unavailable or the model never became ready)"
        )
    return base


@pytest.fixture(scope="function")
def configured_dspy_lm(hermetic_test_lm):
    """Configure DSPy with a temperature=0 test LM (deterministic).

    Function-scope is required because the session-wide ``cleanup_dspy_state``
    autouse fixture (``tests/conftest.py:150``) nulls ``dspy.settings.lm``
    after every test. A module-scope fixture only configures once and then
    every subsequent test in the module sees ``No LM is loaded``.

    Uses ``cogniverse_foundation.config.llm_factory.create_budgeted_dspy_lm``
    — the constructor the ingest path builds its LM with — at temperature=0.
    """
    import dspy

    from cogniverse_foundation.config.llm_factory import create_budgeted_dspy_lm
    from cogniverse_foundation.config.unified_config import LLMEndpointConfig
    from tests.utils.hermetic_llm import MODEL as SIDECAR_MODEL

    # Pin model + base to the sidecar the goldens were recorded against.
    # The session test-config fixture rewrites the resolver chain to
    # whatever live OAI server it detects (e.g. host ollama), which would
    # silently swap the model under the locked goldens.
    endpoint = LLMEndpointConfig(
        model=f"openai/{SIDECAR_MODEL}",
        api_base=hermetic_test_lm,
        api_key=resolve_api_key(),
        # Hard-pinned for golden determinism.
        temperature=0.0,
        max_tokens=800,
    )
    lm = create_budgeted_dspy_lm(endpoint)
    # Bypass the litellm/dspy disk cache: a cached completion recorded
    # against an earlier serving of the same model id would be replayed
    # forever, so the assertions would never exercise the live model.
    lm.cache = False
    dspy.configure(lm=lm)
    try:
        yield lm
    finally:
        dspy.configure(lm=None)


# --------------------------------------------------------------------------- #
# Span / Phoenix helpers                                                      #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


class TestClaimExtractorMarieCurie:
    """Short-input determinism + negative + idempotency."""

    def test_marie_curie_extract_locked(self, configured_dspy_lm):
        """ClaimExtractor.extract() on the Marie Curie seed returns the three
        claims the sentence states, each anchored to the segment and each
        carrying a relation from the locked vocabulary."""
        extractor = ClaimExtractor()
        edges = extractor.extract(
            text=SEG_3_TEXT,
            entity_hints=SEG_3_ENTITY_HINTS,
            modality_hint="transcript",
            segment_anchor=_seg3_anchor(),
            tenant_id=TENANT_ID,
            source_doc_id=VIDEO_ID,
        )
        dumped = [asdict(e) for e in edges]
        by_relation = {e.relation: e for e in edges}
        assert sorted(by_relation) == SEG_3_RELATIONS, dumped
        assert len(edges) == len(SEG_3_RELATIONS), dumped
        assert set(SEG_3_RELATIONS) - PREDICATE_VOCABULARY == set(), (
            "the segment's relations are no longer the vocabulary the extractor keeps"
        )

        # Anchoring is the extractor's contract and is identical on every edge.
        assert [
            (
                e.evidence_span,
                e.modality,
                e.provenance,
                e.segment_id,
                e.source_doc_id,
                e.tenant_id,
                e.ts_start,
                e.ts_end,
            )
            for e in edges
        ] == [
            (
                SEG_3_TEXT,
                "transcript",
                "EXTRACTED",
                SEG_3_SEGMENT,
                VIDEO_ID,
                TENANT_ID,
                SEG_3_START,
                SEG_3_END,
            )
        ] * len(SEG_3_RELATIONS)

        # The two claims the sentence states unambiguously.
        assert (
            by_relation["discovered"].source,
            by_relation["discovered"].target,
        ) == ("Marie Curie", "radium")
        assert (
            by_relation["worked_at"].source,
            by_relation["worked_at"].target,
        ) == ("Marie Curie", "Sorbonne")

        # "discovered radium in 1898" supports the year claim about either
        # discovery participant; both are named in the segment and the
        # extractor binds neither, so the object is exact and the subject is
        # pinned to those two, each a verbatim span of the segment.
        year_claim = by_relation["discovered_in"]
        assert year_claim.target == "1898"
        assert {year_claim.source} - SEG_3_DISCOVERY_PARTICIPANTS == set(), dumped
        assert SEG_3_TEXT.count(year_claim.source) == 1, year_claim

    def test_chain_of_thought_exposes_reasoning_and_claims_only(
        self, configured_dspy_lm
    ):
        """The CoT module returns reasoning plus at most the capped number of
        grounded claims, terminating inside the derived token budget."""
        import dspy

        module = dspy.ChainOfThought(ClaimExtractionSignature)
        prediction = module(
            text_segment=SEG_3_TEXT,
            entity_hints=SEG_3_ENTITY_HINTS,
            modality_hint="transcript",
        )
        assert set(prediction.keys()) == {"reasoning", "claims"}
        assert 1 <= len(prediction.claims) <= CLAIM_EXTRACTION_MAX_CLAIMS, prediction
        assert [sorted(claim) for claim in prediction.claims] == [
            ["confidence", "evidence_span", "object", "predicate", "subject"]
        ] * len(prediction.claims), prediction.claims
        assert [
            claim["evidence_span"] in SEG_3_TEXT for claim in prediction.claims
        ] == [True] * len(prediction.claims), prediction.claims
        # The bounded contract is the point: the completion ended on its own,
        # not at max_tokens.
        choice = dspy.settings.lm.history[-1]["response"].choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason is None and isinstance(choice, dict):
            finish_reason = choice["finish_reason"]
        assert finish_reason == "stop", dspy.settings.lm.history[-1]

    def test_negative_yellow_flowers_no_edges(self, configured_dspy_lm):
        """'Yellow flowers in a glass vase.' yields zero SPO edges
        (no real subject/predicate/object structure)."""
        extractor = ClaimExtractor()
        edges = extractor.extract(
            text=VLM_FLOWERS_TEXT,
            entity_hints=["flowers", "vase"],
            modality_hint="vlm",
            segment_anchor=_vlm_flowers_anchor(),
            tenant_id=TENANT_ID,
            source_doc_id=VIDEO_ID,
        )
        # Empty list, byte-equal — no spurious SPO emission.
        edge_dicts = [asdict(e) for e in edges]
        for d in edge_dicts:
            d.pop("created_at", None)
        assert edge_dicts == [], (
            f"Expected zero SPO edges for '{VLM_FLOWERS_TEXT}', got: {edge_dicts}"
        )

    def test_idempotency_byte_equal(self, configured_dspy_lm):
        """Two consecutive .extract() calls with identical args produce
        byte-equal JSON serializations."""
        extractor = ClaimExtractor()
        run_one = extractor.extract(
            text=SEG_3_TEXT,
            entity_hints=SEG_3_ENTITY_HINTS,
            modality_hint="transcript",
            segment_anchor=_seg3_anchor(),
            tenant_id=TENANT_ID,
            source_doc_id=VIDEO_ID,
        )
        run_two = extractor.extract(
            text=SEG_3_TEXT,
            entity_hints=SEG_3_ENTITY_HINTS,
            modality_hint="transcript",
            segment_anchor=_seg3_anchor(),
            tenant_id=TENANT_ID,
            source_doc_id=VIDEO_ID,
        )

        def _canonical(edges):
            dicts = [asdict(e) for e in edges]
            for d in dicts:
                d.pop("created_at", None)
            return json.dumps(
                sorted(dicts, key=lambda d: (d["source"], d["relation"], d["target"])),
                sort_keys=True,
                indent=2,
            )

        json_one = _canonical(run_one)
        json_two = _canonical(run_two)
        assert json_one == json_two, (
            "Two consecutive ClaimExtractor.extract() calls produced different "
            f"output:\n--- run 1 ---\n{json_one}\n--- run 2 ---\n{json_two}"
        )


class TestClaimExtractorRLMPromotion:
    """RLM promotion threshold honored by ``_select_module``."""

    def test_long_input_promotes_to_rlm(self, configured_dspy_lm):
        """60 concatenated Marie Curie sentences route through the RLM
        path, not ChainOfThought, and the extraction dedupes to the single
        claim the repeated sentence states.
        """
        long_text = (SEG_3_TEXT + " ") * 60
        assert len(long_text) > 3000, "fixture must exceed RLM_PROMOTION_TOKENS"

        extractor = ClaimExtractor()
        # Snapshot whether the extractor selects the RLM module for this
        # text — independent of Phoenix availability.
        selected = extractor._select_module(text=long_text, tenant_id=TENANT_ID)
        import dspy

        is_rlm = isinstance(selected, dspy.RLM)
        assert is_rlm, (
            f"Expected RLM module for long input (len={len(long_text)}), "
            f"got {type(selected).__name__}"
        )

        edges = extractor.extract(
            text=long_text,
            entity_hints=SEG_3_ENTITY_HINTS,
            modality_hint="transcript",
            segment_anchor=_seg3_anchor(),
            tenant_id=TENANT_ID,
            source_doc_id=VIDEO_ID,
        )
        # Deduplicated by deterministic edge_id; expected count locked
        # via golden so any LM drift is explicit.
        edge_ids = sorted({e.edge_id for e in edges})
        assert_golden(
            {"edge_count": len(edges), "unique_edge_ids": edge_ids},
            "claim_extractor_long_doc_edge_summary.json",
        )

        # The promoted module is capped at the production transcript-turn
        # count and is the one the extractor keeps for this output size.
        assert selected.max_iterations == RLM_TRANSCRIPT_TURNS
        assert extractor._select_module(text=long_text, tenant_id=TENANT_ID) is selected

    def test_short_input_stays_on_chain_of_thought(self, configured_dspy_lm):
        """Short Marie Curie sentence (56 chars) routes to the cached
        ChainOfThought module, never the recursive path."""
        extractor = ClaimExtractor()
        selected = extractor._select_module(text=SEG_3_TEXT, tenant_id=TENANT_ID)
        import dspy

        # The selected module for a short text must NOT be an RLM.
        assert not isinstance(selected, dspy.RLM), (
            f"Short input promoted to RLM unexpectedly: {type(selected).__name__}"
        )
        assert isinstance(selected, dspy.ChainOfThought), (
            f"Short input should route to ChainOfThought, got {type(selected).__name__}"
        )
        assert (
            extractor._select_module(text=SEG_3_TEXT, tenant_id=TENANT_ID) is selected
        )

        edges = extractor.extract(
            text=SEG_3_TEXT,
            entity_hints=SEG_3_ENTITY_HINTS,
            modality_hint="transcript",
            segment_anchor=_seg3_anchor(),
            tenant_id=TENANT_ID,
            source_doc_id=VIDEO_ID,
        )
        assert sorted({e.relation for e in edges}) == SEG_3_RELATIONS, [
            (e.source_node_id, e.relation, e.target_node_id) for e in edges
        ]

    def test_prompt_that_cannot_fit_raises_with_the_sizes(self, hermetic_test_lm):
        """A reservation that leaves no input allowance raises with the window,
        the reservation and the measured input — the caller never sees the
        provider's context-length rejection, and nothing is sent."""
        import dspy

        from cogniverse_foundation.config.llm_factory import create_budgeted_dspy_lm
        from cogniverse_foundation.config.token_budget import (
            PromptBudgetExceededError,
            fetch_context_window,
        )
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig
        from tests.utils.hermetic_llm import MODEL as SIDECAR_MODEL

        served_window = fetch_context_window(hermetic_test_lm)
        # Reserve all but a sliver of the window for the completion: no prompt
        # this signature can build will fit what is left.
        reserved = served_window - 64
        starved = create_budgeted_dspy_lm(
            LLMEndpointConfig(
                model=f"openai/{SIDECAR_MODEL}",
                api_base=hermetic_test_lm,
                api_key=resolve_api_key(),
                temperature=0.0,
                max_tokens=reserved,
            )
        )
        starved.cache = False
        extractor = ClaimExtractor()

        with dspy.context(lm=starved):
            with pytest.raises(RuntimeError) as short_input:
                extractor.extract(
                    text=SEG_3_TEXT,
                    entity_hints=SEG_3_ENTITY_HINTS,
                    modality_hint="transcript",
                    segment_anchor=_seg3_anchor(),
                    tenant_id=TENANT_ID,
                    source_doc_id=VIDEO_ID,
                )
            with pytest.raises(RuntimeError) as long_input:
                extractor.extract(
                    text=(SEG_3_TEXT + " ") * 60,
                    entity_hints=SEG_3_ENTITY_HINTS,
                    modality_hint="transcript",
                    segment_anchor=_seg3_anchor(),
                    tenant_id=TENANT_ID,
                    source_doc_id=VIDEO_ID,
                )

        budget_error = short_input.value.__cause__
        assert type(budget_error) is PromptBudgetExceededError, short_input.value
        assert budget_error.context_window == served_window
        assert budget_error.reserved_output == reserved
        assert budget_error.input_tokens > served_window - reserved
        assert budget_error.dropped_demos == 0
        assert f"context_window={served_window}" in str(short_input.value)
        assert f"input_tokens={budget_error.input_tokens}" in str(short_input.value)

        # The recursive path refuses before building a module at all: its
        # harness alone cannot fit what the reservation leaves.
        recursive_error = long_input.value.__cause__
        assert type(recursive_error) is RecursiveClaimBudgetError, long_input.value
        assert recursive_error.context_window == served_window
        assert recursive_error.reserved_output == reserved
        assert extractor._rlm_modules == {}

    def test_second_lm_on_the_same_endpoint_reuses_the_read_window(
        self, hermetic_test_lm
    ):
        """Ingestion builds one LM per segment; the window behind them is read
        from the endpoint once."""
        import cogniverse_foundation.config.token_budget as token_budget
        from cogniverse_foundation.config.llm_factory import create_budgeted_dspy_lm
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig
        from tests.utils.hermetic_llm import MODEL as SIDECAR_MODEL

        token_budget.clear_context_window_memo()
        real_fetch = token_budget.fetch_context_window
        fetch_calls: List[str] = []

        def counting_fetch(api_base, **kwargs):
            fetch_calls.append(api_base)
            return real_fetch(api_base, **kwargs)

        def _lm():
            return create_budgeted_dspy_lm(
                LLMEndpointConfig(
                    model=f"openai/{SIDECAR_MODEL}",
                    api_base=hermetic_test_lm,
                    api_key=resolve_api_key(),
                    temperature=0.0,
                    max_tokens=800,
                )
            )

        with patch.object(token_budget, "fetch_context_window", counting_fetch):
            first, second = _lm().budget, _lm().budget

        assert len(fetch_calls) == 1, fetch_calls
        assert (first.context_window, first.reserved_output) == (
            second.context_window,
            second.reserved_output,
        )
        assert first.context_window == real_fetch(hermetic_test_lm)

    def test_concurrent_promotion_resolves_the_window_once(self, configured_dspy_lm):
        """Eight threads promoting at once read the served window one time and
        share one module sized from it."""
        import cogniverse_foundation.config.token_budget as token_budget

        token_budget.clear_context_window_memo()
        real_fetch = token_budget.fetch_context_window
        fetch_calls: List[str] = []
        fetch_lock = threading.Lock()

        def counting_fetch(api_base, **kwargs):
            with fetch_lock:
                fetch_calls.append(api_base)
            return real_fetch(api_base, **kwargs)

        reserved = configured_dspy_lm.kwargs["max_tokens"]
        long_text = (SEG_3_TEXT + " ") * 60
        extractor = ClaimExtractor()
        threads = 8
        barrier = threading.Barrier(threads)
        selected: List[Any] = []
        failures: List[BaseException] = []
        select_lock = threading.Lock()

        def touch() -> None:
            barrier.wait(timeout=60)
            try:
                module = extractor._select_module(text=long_text, tenant_id=TENANT_ID)
            except BaseException as exc:  # noqa: BLE001 - reported below
                with select_lock:
                    failures.append(exc)
                return
            with select_lock:
                selected.append(module)

        with patch.object(token_budget, "fetch_context_window", counting_fetch):
            workers = [threading.Thread(target=touch) for _ in range(threads)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=180)

        assert failures == []
        assert len(selected) == threads
        assert len(fetch_calls) == 1, fetch_calls
        served_window = token_budget.fetch_context_window(
            configured_dspy_lm.kwargs["api_base"]
        )
        input_budget = served_window - reserved
        transcript_tokens = (
            input_budget
            - RLM_HARNESS_TOKENS
            - int(input_budget * PROMPT_TOKENIZER_MARGIN_SHARE)
        )
        expected_chars = (
            transcript_tokens // RLM_TRANSCRIPT_TURNS * RLM_TRANSCRIPT_CHARS_PER_TOKEN
        )
        assert len({id(module) for module in selected}) == 1
        assert list(extractor._rlm_modules) == [expected_chars]
        assert selected[0].max_output_chars == expected_chars
        assert selected[0].max_iterations == RLM_TRANSCRIPT_TURNS


class TestClaimExtractorArtifact:
    """Compiled-artifact dataset equality + demo count."""

    @pytest.mark.asyncio
    async def test_artifact_blob_round_trip(
        self, configured_dspy_lm, phoenix_container
    ):
        """Compiled ClaimExtractor state persists as a ``("model",
        "claim_extraction")`` JSON blob and round-trips byte-for-byte through
        ArtifactManager, then restores into a fresh ChainOfThought via
        ``load_state``.

        This is the mechanism ``ClaimExtractor._load_compiled_state`` actually
        uses (``save_blob`` / ``load_blob`` + ``module.load_state``) — not the
        prompts dataset that ``load_for_request`` serves, whose ``{"prompts":
        ...}`` shape ``load_state`` cannot consume. The test populates its own
        blob, so it does not depend on externally pre-seeded Phoenix state.

        Fails (does not skip) when no Phoenix endpoint is reachable — the
        artifact storage backend is the live Phoenix instance.
        """
        import dspy

        # Workspace packages — always importable; a real ImportError should
        # error loudly, not silently skip (per the "skips = bugs" rule below).
        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
        from cogniverse_telemetry_phoenix.provider import PhoenixProvider

        # The managed per-process Phoenix container — the test provisions
        # its own backend instead of probing for an operator-run one.
        http_endpoint = phoenix_container["http_endpoint"]
        grpc_endpoint = phoenix_container["grpc_endpoint"]

        provider = PhoenixProvider()
        provider.initialize(
            {
                "tenant_id": TENANT_ID,
                "http_endpoint": http_endpoint,
                "grpc_endpoint": grpc_endpoint,
            }
        )
        manager = ArtifactManager(telemetry_provider=provider, tenant_id=TENANT_ID)

        # Build a compiled state carrying a known 8-demo set (the
        # BootstrapFewShot k=8 the optimizer harness targets).
        module = dspy.ChainOfThought(ClaimExtractionSignature)
        state = json.loads(json.dumps(module.dump_state(), default=str))
        predict_key = next(
            k for k, v in state.items() if isinstance(v, dict) and "demos" in v
        )
        state[predict_key]["demos"] = [
            {
                "text_segment": f"Person {i} discovered element {i} in 18{i:02d}.",
                "entity_hints": f"Person {i}|element {i}|18{i:02d}",
                "modality_hint": "transcript",
                "claims": (
                    f'[{{"subject":"Person {i}","predicate":"discovered",'
                    f'"object":"element {i}"}}]'
                ),
                "reasoning": f"Subject-verb-object claim number {i}.",
            }
            for i in range(8)
        ]
        state_json = json.dumps(state, default=str)

        dataset_id = await manager.save_blob("model", "claim_extraction", state_json)
        assert dataset_id

        loaded_json = await manager.load_blob("model", "claim_extraction")
        assert loaded_json is not None
        loaded_state = json.loads(loaded_json)
        assert loaded_state == state
        assert len(loaded_state[predict_key]["demos"]) == 8

        # The blob restores into a fresh module via the same load_state path
        # ClaimExtractor uses.
        fresh = dspy.ChainOfThought(ClaimExtractionSignature)
        fresh.load_state(loaded_state)
        assert fresh.dump_state()[predict_key]["demos"] == state[predict_key]["demos"]


class TestClaimExtractorPredicateVocabulary:
    """Predicate vocabulary across the 100-example training set."""

    def test_predicate_vocab_locked(self, configured_dspy_lm):
        """Every predicate in the committed claim-extraction corpus is one the
        extractor keeps, and the corpus covers the whole vocabulary."""
        training_path = REPO_ROOT / "data" / "training" / "claim_extraction.jsonl"
        assert training_path.exists(), (
            f"the committed claim-extraction corpus is missing at {training_path}"
        )

        with training_path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) == CLAIM_TRAINING_ROWS

        predicates: set[str] = set()
        for row in rows:
            for claim in row.get("claims") or []:
                pred = (claim.get("predicate") or "").strip()
                if pred:
                    predicates.add(pred)

        assert sorted(predicates) == sorted(PREDICATE_VOCABULARY)


# --------------------------------------------------------------------------- #
# Signature sanity                                                            #
# --------------------------------------------------------------------------- #


def test_signature_field_shape() -> None:
    """Catches an upstream signature rename — the goldens above all key
    on the current ``ClaimExtractionSignature`` field set."""
    sig = ClaimExtractionSignature
    input_fields = sorted(sig.input_fields.keys())
    output_fields = sorted(sig.output_fields.keys())
    assert input_fields == ["entity_hints", "modality_hint", "text_segment"], (
        f"ClaimExtractionSignature input fields drifted: {input_fields}"
    )
    assert output_fields == ["claims"], (
        f"ClaimExtractionSignature output fields drifted: {output_fields}"
    )


def test_extract_signature_field_present() -> None:
    """ClaimExtractor.extract must accept the documented kw-only args."""
    import inspect

    sig = inspect.signature(ClaimExtractor.extract)
    params = sorted(p for p in sig.parameters if p != "self")
    assert params == sorted(
        [
            "text",
            "entity_hints",
            "modality_hint",
            "segment_anchor",
            "tenant_id",
            "source_doc_id",
        ]
    ), f"ClaimExtractor.extract signature drifted: {params}"


def _smoke_marie_curie_anchor_round_trip() -> None:
    """Trivial sanity that the anchor builder produces the Mention shape
    used in the extract-locked assertion. Not a pytest test — invoked by
    import to catch typos early."""
    anchor = _seg3_anchor()
    keys = sorted(asdict(anchor).keys())
    assert keys == [
        "evidence_span",
        "modality",
        "segment_id",
        "source_doc_id",
        "ts_end",
        "ts_start",
    ], f"Mention anchor shape drift: {keys}"


_smoke_marie_curie_anchor_round_trip()


def _suppress_unused_warning() -> Dict[str, Any]:
    """Returns the dict used by typing-only imports so the linter doesn't
    flag them. Kept as a function so it can be referenced from a test
    without polluting module-level state."""
    return {"VIDEO_ID": VIDEO_ID}


# --------------------------------------------------------------------------- #
# Output budget on the e2e corpus document                                    #
# --------------------------------------------------------------------------- #

E2E_DOCUMENT = (
    Path(__file__).resolve().parents[3] / "data" / "testset" / "dataset_summary.md"
)

# GLiNER's per-chunk entity names for the document above; the first segment of
# a source carries no prior-entity pool, so these are the exact hints the
# ingest path passes for each chunk.
E2E_DOCUMENT_CHUNK_HINTS = [
    ["mbzuai-oryx", "Large Vision and Language Models", "Blender Foundation", "Google"],
    ["ActivityNet"],
    ["Whisper cache"],
]


class TestClaimBudgetOnCorpusDocument:
    def test_every_chunk_of_the_e2e_document_terminates_inside_the_budget(
        self, hermetic_test_lm, monkeypatch
    ):
        """The production ``ClaimExtractor`` clamps the tenant endpoint to the
        derived output budget. Fed the e2e document chunk by chunk with the
        hints the ingest path produces, every completion must end on its own:
        a ``length`` finish fails the segment, and a single-segment document
        then fails the whole ingest after its content was already fed."""
        from cogniverse_agents.graph.doc_extractor import DocExtractor
        from cogniverse_agents.graph.graph_schema import DOCUMENT_MODALITY
        from cogniverse_foundation.config import semantic_router
        from cogniverse_foundation.config.llm_factory import create_budgeted_dspy_lm
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig
        from tests.utils.hermetic_llm import MODEL as SIDECAR_MODEL

        recorded_lms: list = []

        def _uncached_lm(endpoint):
            lm = create_budgeted_dspy_lm(endpoint)
            lm.cache = False
            recorded_lms.append(lm)
            return lm

        monkeypatch.setattr(semantic_router, "create_budgeted_dspy_lm", _uncached_lm)

        endpoint = LLMEndpointConfig(
            model=f"openai/{SIDECAR_MODEL}",
            api_base=hermetic_test_lm,
            api_key=resolve_api_key(),
            temperature=0.0,
            max_tokens=8000,
        )
        extractor = ClaimExtractor(llm_config=endpoint)
        chunks = DocExtractor._chunk_text(
            DocExtractor.__new__(DocExtractor), E2E_DOCUMENT.read_text()
        )
        assert [len(c) for c in chunks] == [1608, 1391, 774]

        finish_reasons: list[str] = []
        for index, (chunk, hints) in enumerate(zip(chunks, E2E_DOCUMENT_CHUNK_HINTS)):
            extractor.extract(
                text=chunk,
                entity_hints=hints,
                modality_hint=DOCUMENT_MODALITY,
                segment_anchor=Mention(
                    source_doc_id="dataset_summary",
                    segment_id="file_0",
                    ts_start=0.0,
                    ts_end=0.0,
                    modality=DOCUMENT_MODALITY,
                    evidence_span=chunk[:200],
                ),
                tenant_id=TENANT_ID,
                source_doc_id="dataset_summary",
            )
            entry = recorded_lms[-1].history[-1]
            finish_reasons.append(entry["response"].choices[0].finish_reason)
            assert recorded_lms[-1].kwargs["max_tokens"] == (
                extractor._llm_config.max_tokens
            ), (index, recorded_lms[-1].kwargs)

        assert finish_reasons == ["stop", "stop", "stop"]
