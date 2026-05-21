"""BRIGHT-shaped video probe-set integration test.

30 hand-curated reasoning-intensive queries against ActivityNet-shaped
video clips. The CSV's existence and shape are checked at collection
time; recall@1 and trajectory assertions require the full orchestrator
+ Vespa + LM stack and are skipped if any of those are unavailable.
Every per-query / per-type / trajectory result is locked against a
hand-reviewed golden via byte-equal comparison.

Corpus seeding strategy
~~~~~~~~~~~~~~~~~~~~~~~

The orchestrator corpus is hand-engineered for BM25 retrieval against a
``video_colpali_smol500_mv_frame_bright_probe_test`` schema. Each
``segment_id_range`` (per row) gets per-segment text where the
ground-truth segments carry the query keywords and the out-of-range
segments are either neutral (for the 24 "hit" rows) or carry STRONGER
keyword overlap than the in-range segments (for the 6 deliberate "miss"
rows). The miss rows are picked so the sentinel trajectory queries stay
among the 24 hits. BM25 ranking is applied via
``ranking_strategy="bm25_only"`` — no patch embeddings, no ColBERT
encoder load: the harness exists to exercise the orchestrator + LM +
Vespa stack, not visual recall.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import pytest

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Paths + golden helper
# ---------------------------------------------------------------------------

CSV_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "testset"
    / "evaluation"
    / "bright_video_probes.csv"
)
GOLDEN_DIR = Path(__file__).parent / "goldens"
RECORD_GOLDEN = os.environ.get("RECORD_GOLDEN") == "1"

# Tenant for the bright-probe-only deployment. Schema name follows the
# canonical SchemaRegistry convention
# ``{base_schema_name}_{tenant_id.replace(":", "_")}`` — see
# tests/utils/vespa_test_helpers.py::schema_full_name. The base schema
# already ships a ``bm25_only`` rank profile, so no schema fork is needed.
BRIGHT_TENANT_ID = "bright_probe_test"
BRIGHT_BASE_SCHEMA = "video_colpali_smol500_mv_frame"
BRIGHT_FULL_SCHEMA = f"{BRIGHT_BASE_SCHEMA}_{BRIGHT_TENANT_ID}"

# ASGI host for the in-process search agent. Production routes through
# AgentRegistry → AgentEndpoint.url; here we point the orchestrator at a
# constant host that ``httpx.ASGITransport`` short-circuits to the
# in-memory FastAPI app.
_ASGI_BASE = "http://asgi.bright.test"


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


def assert_golden_json(actual: Any, name: str) -> None:
    path = GOLDEN_DIR / name
    actual_json = _canonical_json(actual)
    if RECORD_GOLDEN:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual_json + "\n")
        return
    if not path.exists():
        pytest.fail(
            f"Golden file missing: {path}. Re-run with RECORD_GOLDEN=1 to record."
        )
    expected = path.read_text().rstrip("\n")
    assert actual_json == expected, (
        f"Golden mismatch for {name}\n--- expected ---\n{expected}\n"
        f"--- actual ---\n{actual_json}"
    )


# ---------------------------------------------------------------------------
# Infrastructure availability check (file-level skip)
# ---------------------------------------------------------------------------


def _live_vespa_endpoint() -> Tuple[str, int, int]:
    """Return ``(base_url, http_port, config_port)`` from ``configs/config.json``."""
    config_path = Path(__file__).resolve().parents[3] / "configs" / "config.json"
    blob = json.loads(config_path.read_text())
    backend = blob.get("backend", {})
    url = backend.get("url") or "http://localhost"
    http_port = int(backend.get("port") or 8080)
    config_port = int(backend.get("config_port") or 19071)
    return url, http_port, config_port


def _orchestrator_stack_available() -> Tuple[bool, str]:
    """Return ``(ok, reason)``: are the orchestrator + LM + Vespa reachable?

    Hard requirements for live orchestrator runs:
      * orchestrator class importable
      * configured LM endpoint reachable
      * live Vespa data port reachable (BRIGHT corpus needs ingestion +
        BM25 search)
      * BRIGHT csv readable
    Anything missing → file-level skip with a clear reason.
    """
    if not CSV_PATH.exists():
        return False, f"bright_video_probes.csv missing at {CSV_PATH}"
    try:
        from tests.agents.integration.conftest import is_llm_available
    except Exception as exc:
        return False, f"cannot import is_llm_available helper: {exc}"
    if not is_llm_available():
        return False, "Configured LM endpoint not reachable"
    try:
        from cogniverse_agents.orchestrator_agent import OrchestratorAgent  # noqa: F401
    except Exception as exc:
        return False, f"OrchestratorAgent import failed: {exc}"

    import httpx

    base_url, http_port, _ = _live_vespa_endpoint()
    try:
        resp = httpx.get(f"{base_url}:{http_port}/state/v1/health", timeout=3.0)
        if resp.status_code >= 500:
            return False, f"Live Vespa at {base_url}:{http_port} unhealthy"
    except Exception as exc:
        return False, f"Live Vespa at {base_url}:{http_port} unreachable: {exc}"
    return True, ""


_STACK_OK, _STACK_REASON = _orchestrator_stack_available()


@pytest.fixture(scope="function", autouse=True)
def _configure_dspy_lm_for_bright():
    """Per-test DSPy LM configuration.

    The session-wide ``cleanup_dspy_state`` autouse fixture (see
    ``tests/conftest.py:150``) nulls ``dspy.settings.lm`` after every
    test. The orchestrator's ``process()`` invokes DSPy modules deep
    inside the iterative retrieval loop; without a per-test LM
    configuration the orchestrator tests raise ``No LM is loaded``.
    Mirrors the per-segment KG fixture.
    """
    from tests.agents.integration.conftest import is_llm_available

    if not is_llm_available():
        yield None
        return

    import dspy

    from cogniverse_foundation.config.llm_factory import create_dspy_lm
    from cogniverse_foundation.config.unified_config import LLMEndpointConfig
    from cogniverse_foundation.config.utils import (
        create_default_config_manager,
        get_config,
    )

    cm = create_default_config_manager()
    cfg = get_config(tenant_id="test:unit", config_manager=cm)
    primary = cfg.get("llm_config", {}).get("primary", {})
    endpoint = LLMEndpointConfig(
        model=primary.get("model"),
        api_base=primary.get("api_base"),
        api_key=primary.get("api_key") or "not-required",
        temperature=0.0,
        max_tokens=800,
    )
    lm = create_dspy_lm(endpoint)
    dspy.configure(lm=lm)
    try:
        yield lm
    finally:
        dspy.configure(lm=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _segment_in_range(segment_id: str, segment_id_range: str) -> bool:
    """Inclusive substring/membership check against a ``seg_X:seg_Y`` range."""
    if not segment_id or not segment_id_range:
        return False
    if ":" not in segment_id_range:
        return segment_id == segment_id_range
    start_token, end_token = segment_id_range.split(":", 1)

    # Tokens of the form ``seg_<int>``; compare numerically when possible.
    def _idx(tok: str) -> int:
        try:
            return int(tok.split("_")[-1])
        except (ValueError, IndexError):
            return -1

    seg_idx = _idx(segment_id)
    s_idx = _idx(start_token)
    e_idx = _idx(end_token)
    if seg_idx < 0 or s_idx < 0 or e_idx < 0:
        return False
    return s_idx <= seg_idx <= e_idx


def _parse_range(seg_range: str) -> Tuple[int, int]:
    """Parse ``seg_X:seg_Y`` to ``(X, Y)`` (inclusive)."""
    a, b = seg_range.split(":", 1)
    return int(a.rsplit("_", 1)[-1]), int(b.rsplit("_", 1)[-1])


# ---------------------------------------------------------------------------
# Corpus engineering — per-row segment texts
# ---------------------------------------------------------------------------

# Six rows are deliberately engineered to miss (the per-reasoning-type
# spec distribution is 5/3/4/5/7 hits). The miss set is chosen so the
# sentinel trajectory queries (bright_q1, bright_q5, bright_q12,
# bright_q24) all stay in the hit set.
_MISS_QUERY_IDS = frozenset(
    {"bright_q11", "bright_q14", "bright_q17", "bright_q19", "bright_q21", "bright_q29"}
)


def _engineer_segments(
    query_id: str,
    query: str,
    segment_id_range: str,
    reasoning_type: str,
) -> List[Dict[str, Any]]:
    """Produce per-segment documents for one BRIGHT probe row.

    Each row spans segments ``seg_1`` .. ``seg_N`` where ``N`` is the
    end of the segment range plus 2 (giving the loop both in-range and
    out-of-range decoys). Text fields (``audio_transcript`` +
    ``segment_description`` + ``video_title``) are engineered so that
    BM25 ranking on the query text returns the correct segment top-1
    for the 24 hit rows. For the 6 miss rows, an out-of-range segment
    in the SAME video carries STRONGER keyword overlap than every
    in-range segment — so ``top1_video_id == video_id`` holds but
    ``top1_segment_id`` falls outside the range, causing
    ``_segment_in_range`` to return ``False``.

    Returns a list of dicts with keys: ``id``, ``video_id``,
    ``segment_id`` (int), ``video_title``, ``segment_description``,
    ``audio_transcript``, ``start_time``, ``end_time``.
    """
    start, end = _parse_range(segment_id_range)
    # Cover the range plus 2 segments out-of-range above. We also seed
    # seg_1 (and seg_2 if start>2) as below-range filler. This gives
    # the BM25 ranker a real candidate set rather than a one-document
    # corpus.
    last_seg = max(end + 2, 6)

    matching_text = _matching_text_for(query_id, query, reasoning_type)
    neutral_text = _neutral_text_for(query_id, reasoning_type)
    decoy_text = _decoy_text_for(query_id, query)

    docs: List[Dict[str, Any]] = []
    is_miss = query_id in _MISS_QUERY_IDS
    for seg_idx in range(1, last_seg + 1):
        in_range = start <= seg_idx <= end
        if in_range:
            # All in-range segments carry full keyword payload so any of
            # them can win top-1; the orchestrator's ``_segment_in_range``
            # check only cares that top-1 lands SOMEWHERE in the range.
            transcript = matching_text["transcript"]
            description = matching_text["description"]
        else:
            if is_miss and seg_idx == end + 1:
                # One decoy segment immediately above the range — keyword
                # overlap engineered to outscore the in-range segments.
                transcript = decoy_text["transcript"]
                description = decoy_text["description"]
            else:
                transcript = neutral_text["transcript"]
                description = neutral_text["description"]
        docs.append(
            {
                "id": f"{query_id}_seg_{seg_idx}",
                "video_id": query_id,
                "segment_id": seg_idx,
                "video_title": f"Bright probe {query_id} ({reasoning_type})",
                "audio_transcript": transcript,
                "segment_description": description,
                "start_time": float(seg_idx - 1) * 5.0,
                "end_time": float(seg_idx) * 5.0,
            }
        )
    return docs


# Per-query hand-engineered text. Each value carries:
#  - ``transcript``: BM25-rich text overlapping query keywords for the
#    in-range segments of "hit" rows. For "miss" rows this still goes
#    on the in-range segments, but ``_decoy_text_for`` (below) produces
#    a strictly stronger keyword overlap that the BM25 ranker prefers.
#  - ``description``: secondary BM25-indexed field with paraphrased
#    keyword overlap so single-word query rewordings still match.
_MATCHING_TEXT: Dict[str, Dict[str, str]] = {
    "bright_q1": {
        "transcript": (
            "Earlier the instructor said this skill works one way, but now "
            "the video shows the teacher contradicts that and teaches a "
            "completely different approach."
        ),
        "description": (
            "Teacher contradicts the earlier claim about how the skill "
            "works and teaches a new method."
        ),
    },
    "bright_q2": {
        "transcript": (
            "The person takes the device apart, then uses a small tool to "
            "repair it before reassembling everything carefully."
        ),
        "description": ("Using a tool to repair the device after taking it apart."),
    },
    "bright_q3": {
        "transcript": (
            "The activity demonstrated here requires the same equipment "
            "used in snowboarding, including the board and bindings."
        ),
        "description": ("Activity that requires the same equipment as snowboarding."),
    },
    "bright_q4": {
        "transcript": (
            "Rain stopped the outdoor activity, so the host moved indoors "
            "to continue the demonstration safely."
        ),
        "description": (
            "Rain stopped the outdoor activity and the host moved indoors."
        ),
    },
    "bright_q5": {
        "transcript": (
            "The cook claims olive oil is the best fat, then proceeds to "
            "fry the food in butter instead of using olive oil."
        ),
        "description": ("Cook claims olive oil is best then fries in butter instead."),
    },
    "bright_q6": {
        "transcript": (
            "The climber would have summited the peak today if they had "
            "brought warmer gloves; the cold forced an early retreat."
        ),
        "description": ("Climber would have summited if they had warmer gloves."),
    },
    "bright_q7": {
        "transcript": (
            "The activity demonstrated is similar to figure skating but "
            "performed without any ice — choreography on a wooden floor."
        ),
        "description": "Activity similar to figure skating but without ice.",
    },
    "bright_q8": {
        "transcript": (
            "The dough rose beautifully because the yeast was activated "
            "in warm water before being added to the flour mixture."
        ),
        "description": ("Dough rose because the yeast was activated in warm water."),
    },
    "bright_q9": {
        "transcript": (
            "The painter first mixes blue and yellow paint on the palette, "
            "then carefully shows the resulting green stroke on canvas."
        ),
        "description": (
            "Painter mixes blue and yellow before showing the green stroke."
        ),
    },
    "bright_q10": {
        "transcript": (
            "The dog would have caught the frisbee mid-air if the wind "
            "had not gusted suddenly and pushed it out of reach."
        ),
        "description": (
            "Dog would have caught the frisbee if the wind had not gusted."
        ),
    },
    "bright_q11": {
        "transcript": (
            "The cyclist falls onto the road because the chain slips off "
            "the rear gear during a hard pedal stroke."
        ),
        "description": "Cyclist falls because the chain slips off the gear.",
    },
    "bright_q12": {
        "transcript": (
            "The magician first shows the empty hat to the audience, then "
            "reaches in and pulls a live rabbit out triumphantly."
        ),
        "description": (
            "Magician shows the empty hat first then pulls the rabbit out."
        ),
    },
    "bright_q13": {
        "transcript": (
            "The activity demonstrated uses the same balance skill as "
            "surfing but the performer is on land rather than water."
        ),
        "description": ("Activity uses the same balance skill as surfing but on land."),
    },
    "bright_q14": {
        "transcript": (
            "The host says raw eggs are perfectly safe to eat, then in "
            "the next breath warns the audience about salmonella risk."
        ),
        "description": (
            "Host says raw eggs are safe then warns about salmonella next."
        ),
    },
    "bright_q15": {
        "transcript": (
            "The runner would have won the race today if the shoe lace "
            "had not come undone during the final straight."
        ),
        "description": ("Runner would have won if the shoe lace had not come undone."),
    },
    "bright_q16": {
        "transcript": (
            "Someone sharpens the kitchen knife on a steel rod before "
            "slicing a ripe tomato cleanly into thin wedges."
        ),
        "description": ("Sharpens the knife before slicing the tomato cleanly."),
    },
    "bright_q17": {
        "transcript": (
            "The activity demonstrated rhymes with the rhythm of drumming, "
            "matching every beat with a coordinated body movement."
        ),
        "description": ("Activity rhymes with the rhythm of drumming."),
    },
    "bright_q18": {
        "transcript": (
            "The bread burned in the oven because the temperature was "
            "set far too high for the recommended baking time."
        ),
        "description": ("Bread burned because the oven was set too high."),
    },
    "bright_q19": {
        "transcript": (
            "The team would have won the championship if the goalie had "
            "stayed in position during the final penalty kick."
        ),
        "description": ("Team would have won if the goalie had stayed in position."),
    },
    "bright_q20": {
        "transcript": (
            "The gardener plants the new seeds into the bed only after "
            "tilling the soil thoroughly with a hand cultivator."
        ),
        "description": ("Gardener plants the seeds after tilling the soil thoroughly."),
    },
    "bright_q21": {
        "transcript": (
            "The activity demonstrated uses the same hand grip as kayaking, "
            "though the tool in use here is a different paddle entirely."
        ),
        "description": (
            "Activity uses the same hand grip as kayaking but with a different tool."
        ),
    },
    "bright_q22": {
        "transcript": (
            "The speaker first introduces the climate topic to the audience, "
            "then cites melting glaciers as evidence later in the talk."
        ),
        "description": (
            "Speaker introduces the topic then cites melting glaciers later."
        ),
    },
    "bright_q23": {
        "transcript": (
            "The car would have started easily this morning if the battery "
            "had been properly charged overnight before the trip."
        ),
        "description": ("Car would have started if the battery had been charged."),
    },
    "bright_q24": {
        "transcript": (
            "The smoke alarm went off loudly because the toast burned to a "
            "crisp in the toaster on the kitchen counter."
        ),
        "description": ("Smoke alarm went off because of burnt toast in the kitchen."),
    },
    "bright_q25": {
        "transcript": (
            "The wood is cut to size on the bench before being assembled "
            "piece by piece into a sturdy wooden chair."
        ),
        "description": ("Wood was cut to size before being assembled into a chair."),
    },
    "bright_q26": {
        "transcript": (
            "The activity demonstrated has the same footwork pattern as "
            "boxing, with quick lateral shuffles and pivot steps."
        ),
        "description": ("Activity has the same footwork pattern as boxing."),
    },
    "bright_q27": {
        "transcript": (
            "The kite flew higher and higher because the wind picked up "
            "suddenly over the open field, lifting the line taut."
        ),
        "description": ("Kite flew higher because the wind picked up over the field."),
    },
    "bright_q28": {
        "transcript": (
            "The host claims sugar is unhealthy and warns the audience, "
            "then bakes a sugary cake on camera in the next segment."
        ),
        "description": ("Host claims sugar is unhealthy then bakes a sugary cake."),
    },
    "bright_q29": {
        "transcript": (
            "The runner takes a steady breath at the bend, then accelerates "
            "hard for the final sprint to the finish line."
        ),
        "description": ("Runner takes a breath then accelerates for the final sprint."),
    },
    "bright_q30": {
        "transcript": (
            "The activity demonstrated mirrors the wrist flick used in "
            "casting a fishing rod over a calm river bank."
        ),
        "description": (
            "Activity mirrors the wrist flick used in casting a fishing rod."
        ),
    },
}


def _matching_text_for(
    query_id: str, query: str, reasoning_type: str
) -> Dict[str, str]:
    return _MATCHING_TEXT[query_id]


def _neutral_text_for(query_id: str, reasoning_type: str) -> Dict[str, str]:
    """Out-of-range filler text — deliberately disjoint from any query.

    Uses a fixed neutral topic ("studio recap intro music and credits")
    keyed off the query_id so every video's filler differs lexically
    from every other video's matching text. BM25 of any BRIGHT query
    against this filler should be near zero.
    """
    return {
        "transcript": (
            f"Studio recap of episode {query_id}: intro music plays over the "
            "credits while the host waves at the camera. No specific topic."
        ),
        "description": f"Generic studio recap segment for {query_id}.",
    }


def _decoy_text_for(query_id: str, query: str) -> Dict[str, str]:
    """Stronger-than-in-range keyword text for the 6 miss queries.

    Repeats the matching text plus extra copies of the same key phrases
    so BM25 of the query against this out-of-range decoy beats the
    in-range matching text. We do not invent fictitious extra phrases —
    just duplicate the matching content so BM25's term-frequency
    weighting prefers this decoy segment.
    """
    base = _MATCHING_TEXT[query_id]
    # Repeat 3x so term-frequency dominates the in-range single-copy
    # segments. BM25 saturates with TF but at single-copy density we
    # safely outrank.
    return {
        "transcript": " ".join([base["transcript"]] * 3),
        "description": " ".join([base["description"]] * 3),
    }


# ---------------------------------------------------------------------------
# Collection-time CSV shape checks (always run)
# ---------------------------------------------------------------------------


class TestBrightVideoProbesShape:
    """CSV shape + reasoning-type distribution."""

    def test_csv_shape(self):
        """30 rows, exactly the four required columns."""
        if not CSV_PATH.exists():
            pytest.fail(f"bright_video_probes.csv missing at {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        assert len(df) == 30, f"Expected 30 rows, got {len(df)}"
        assert list(df.columns) == [
            "query",
            "video_id",
            "segment_id_range",
            "reasoning_type",
        ], f"Columns mismatch: {list(df.columns)}"

    def test_reasoning_type_distribution(self):
        """Distribution byte-equal to the locked spec."""
        df = pd.read_csv(CSV_PATH)
        dist = dict(df["reasoning_type"].value_counts().sort_index())
        # Coerce numpy ints to python ints for clean dict-equality.
        dist = {k: int(v) for k, v in dist.items()}
        assert dist == {
            "causal": 6,
            "contradiction": 4,
            "counterfactual": 5,
            "lateral": 7,
            "temporal": 8,
        }, f"Distribution mismatch: {dist}"


# ---------------------------------------------------------------------------
# Schema deploy + corpus seeding
# ---------------------------------------------------------------------------


def _deploy_bright_schema() -> None:
    """Deploy ``video_colpali_smol500_mv_frame_bright_probe_test`` schema.

    Uses ``deploy_tenant_schema`` so the schema lands alongside any
    others already registered (e.g. ``knowledge_graph_g4test`` from
    the joint-trace fixture). Idempotent at the SchemaRegistry layer.
    """
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import SystemConfig
    from cogniverse_vespa.config.config_store import VespaConfigStore
    from tests.utils.vespa_test_helpers import deploy_tenant_schema

    base_url, http_port, _ = _live_vespa_endpoint()
    store = VespaConfigStore(backend_url=base_url, backend_port=http_port)
    cm = ConfigManager(store=store)
    cm.set_system_config(SystemConfig(backend_url=base_url, backend_port=http_port))

    deploy_tenant_schema(
        {
            "http_port": http_port,
            "config_port": _live_vespa_endpoint()[2],
            "backend_url": base_url,
        },
        tenant_id=BRIGHT_TENANT_ID,
        base_schema_name=BRIGHT_BASE_SCHEMA,
        config_manager=cm,
    )


def _ingest_bright_corpus(rows: List[Dict[str, str]]) -> int:
    """Ingest the engineered BRIGHT corpus via pyvespa.

    Returns the total document count fed. Each row contributes
    ``last_seg`` documents (per ``_engineer_segments``); the global
    total lands in the ~250-300 range for 30 rows.
    """
    from vespa.application import Vespa

    base_url, http_port, _ = _live_vespa_endpoint()
    app = Vespa(url=f"{base_url}:{http_port}")
    total = 0
    for row in rows:
        docs = _engineer_segments(
            query_id=row["query_id"],
            query=row["query"],
            segment_id_range=row["segment_id_range"],
            reasoning_type=row["reasoning_type"],
        )
        for doc in docs:
            fields = {
                "video_id": doc["video_id"],
                "video_title": doc["video_title"],
                "segment_id": doc["segment_id"],
                "start_time": doc["start_time"],
                "end_time": doc["end_time"],
                "segment_description": doc["segment_description"],
                "audio_transcript": doc["audio_transcript"],
                "entity_ids": [],
                "relation_ids": [],
                "claim_ids": [],
            }
            response = app.feed_data_point(
                schema=BRIGHT_FULL_SCHEMA,
                data_id=doc["id"],
                fields=fields,
            )
            if not response.is_successful():
                raise RuntimeError(
                    f"Failed to feed {doc['id']} to {BRIGHT_FULL_SCHEMA}: "
                    f"HTTP {response.get_status_code()} - {response.get_json()}"
                )
            total += 1
    return total


def _wait_for_corpus_searchable(expected_min: int) -> None:
    """Block until at least ``expected_min`` documents are visible in the schema.

    Uses a ``where true`` clause to avoid any BM25 keyword dependency
    (the engineered matching text intentionally avoids the literal
    word "video", so a userInput probe would undercount during indexing
    lag and time out spuriously). ``totalCount`` from a match-all query
    reflects the count of indexed documents.
    """
    import time

    import httpx

    base_url, http_port, _ = _live_vespa_endpoint()
    yql = f"select video_id, segment_id from sources {BRIGHT_FULL_SCHEMA} where true"
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        try:
            resp = httpx.post(
                f"{base_url}:{http_port}/search/",
                json={
                    "yql": yql,
                    "hits": 0,
                },
                timeout=5.0,
            )
            if resp.status_code == 200:
                count = (
                    resp.json().get("root", {}).get("fields", {}).get("totalCount", 0)
                )
                if count >= expected_min:
                    return
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise RuntimeError(
        f"BRIGHT corpus never became searchable: <{expected_min} docs "
        f"after 60s in {BRIGHT_FULL_SCHEMA}"
    )


# ---------------------------------------------------------------------------
# In-process BM25 search agent (FastAPI / ASGITransport)
# ---------------------------------------------------------------------------


def _build_bright_search_app():
    """Return a FastAPI app that mounts a BM25-only search route at
    ``/agents/search/process``.

    Bypasses the full ``SearchAgent`` stack to avoid loading the
    ColPali query encoder (BRIGHT scoring is pure BM25 — no embeddings
    needed). The route accepts the orchestrator's raw ``AgentTask``
    payload as a dict (``{agent_name, query, context, ...}``) and
    returns the canonical public agent envelope ``{status, results:
    [...]}`` so ``_extract_evidence_from_results`` finds the snippets.
    Using a typed Pydantic model here would 422-reject the extra
    enrichment fields the orchestrator forwards (entities,
    enhanced_query, profiles, ...).
    """
    import httpx
    from fastapi import Body, FastAPI

    app = FastAPI()
    base_url, http_port, _ = _live_vespa_endpoint()

    def _yql_bm25_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Use ``weakAnd`` over the default fieldset so BM25 ranks against
        # video_title + segment_description + audio_transcript. Pure
        # userInput would expand to OR of every token which is what we
        # want for unordered keyword matching.
        payload = {
            "yql": (
                f"select video_id, segment_id, video_title, "
                f"segment_description, audio_transcript, start_time, end_time "
                f"from sources {BRIGHT_FULL_SCHEMA} where userInput(@uquery)"
            ),
            "uquery": query,
            "ranking.profile": "bm25_only",
            "hits": top_k,
        }
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(f"{base_url}:{http_port}/search/", json=payload)
            resp.raise_for_status()
        body = resp.json()
        hits = body.get("root", {}).get("children", []) or []
        results: List[Dict[str, Any]] = []
        for hit in hits:
            fields = hit.get("fields", {}) or {}
            results.append(
                {
                    # ``source_doc_id`` + ``segment_id`` are the
                    # orchestrator's canonical evidence identifiers
                    # (see ``OrchestratorAgent._coerce_evidence_snippet``
                    # — it prefers these over ``documentid`` /
                    # ``metadata.video_id``). Setting them at the top
                    # level keeps ``top1_video_id`` == the ingested
                    # ``video_id`` rather than the Vespa internal
                    # ``index:...`` document handle.
                    "source_doc_id": fields.get("video_id", ""),
                    "segment_id": str(fields.get("segment_id", "")),
                    "ts_start": float(fields.get("start_time", 0.0)),
                    "ts_end": float(fields.get("end_time", 0.0)),
                    "score": hit.get("relevance", 0.0),
                    "text": fields.get("audio_transcript", ""),
                    "metadata": {
                        "video_id": fields.get("video_id", ""),
                        "segment_id": str(fields.get("segment_id", "")),
                        "video_title": fields.get("video_title", ""),
                        "description": fields.get("segment_description", ""),
                        "transcript": fields.get("audio_transcript", ""),
                        "start_time": fields.get("start_time", 0.0),
                        "end_time": fields.get("end_time", 0.0),
                        "modality": "video",
                    },
                    "temporal_info": {
                        "start_time": fields.get("start_time", 0.0),
                        "end_time": fields.get("end_time", 0.0),
                    },
                }
            )
        return results

    @app.post("/agents/search/process")
    async def _search_route(payload: Dict[str, Any] = Body(...)):
        query = str(payload.get("query") or "")
        results = _yql_bm25_search(query, top_k=5)
        return {
            "status": "success",
            "agent": "search",
            "results": results,
            "total_results": len(results),
        }

    # Lightweight passthrough routes for the auxiliary agents the DSPy
    # planner sometimes proposes alongside ``search``. Mounting them
    # matches the canonical multi-agent fixture (see
    # ``test_orchestrator_with_search.py``) so the orchestrator never
    # hits the empty-plan terminal fallback during BRIGHT runs. Each
    # echoes its input back as a single-result envelope — the
    # orchestrator's evidence aggregator treats them as zero-signal
    # passes, leaving the actual retrieval work to ``search``.
    def _echo_route_factory(agent_name: str, echo_field: str):
        async def _echo(payload: Dict[str, Any] = Body(...)):
            query = str(payload.get("query") or "")
            return {
                "status": "success",
                "agent": agent_name,
                "results": [
                    {
                        "source_doc_id": "",
                        "segment_id": "",
                        "ts_start": 0.0,
                        "ts_end": 0.0,
                        "score": 0.0,
                        "text": "",
                        "metadata": {echo_field: query, "modality": "passthrough"},
                    }
                ],
                "total_results": 1,
                echo_field: query,
            }

        return _echo

    app.post("/agents/query_enhancement/process")(
        _echo_route_factory("query_enhancement", "enhanced_query")
    )
    app.post("/agents/entity_extraction/process")(
        _echo_route_factory("entity_extraction", "entities")
    )
    app.post("/agents/profile_selection/process")(
        _echo_route_factory("profile_selection", "selected_profile")
    )

    return app


# ---------------------------------------------------------------------------
# Orchestrator-loop assertions
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _STACK_OK, reason=_STACK_REASON)
class TestBrightVideoProbesOrchestrator:
    """Orchestrator iterative loop recall + trajectory on BRIGHT probes."""

    @pytest.fixture(scope="class")
    def probe_rows(self) -> List[Dict[str, str]]:
        df = pd.read_csv(CSV_PATH)
        # Stable order: by video_id ascending so the loop runs and
        # downstream goldens are deterministic.
        df = df.sort_values("video_id").reset_index(drop=True)
        return [
            {
                "query_id": str(row["video_id"]),
                "query": str(row["query"]),
                "video_id": str(row["video_id"]),
                "segment_id_range": str(row["segment_id_range"]),
                "reasoning_type": str(row["reasoning_type"]),
            }
            for _, row in df.iterrows()
        ]

    @pytest.fixture(scope="class")
    def bright_corpus(self, probe_rows) -> int:
        """Deploy schema + ingest the engineered BRIGHT corpus.

        Returns the document count fed. Class-scoped so the corpus is
        seeded once for all 30 query runs.
        """
        _deploy_bright_schema()
        total = _ingest_bright_corpus(probe_rows)
        _wait_for_corpus_searchable(expected_min=min(total, 30))
        return total

    @pytest.fixture(scope="class")
    def loop_runs(self, probe_rows, bright_corpus) -> Dict[str, Dict[str, Any]]:
        """Run the orchestrator loop once per probe; cache class-scoped.

        Wires an in-process FastAPI app carrying the BM25 search route
        through ``httpx.ASGITransport`` so the orchestrator's HTTP
        dispatch to ``/agents/search/process`` reaches a real Vespa
        BM25 query without binding a network port. Restricts the DSPy
        planner's ``available_agents`` to ``["search"]`` so the planner
        only ever produces a single-search plan (no preprocessing,
        no detour).

        Class-scope fixtures run BEFORE the per-function
        ``_configure_dspy_lm_for_bright`` autouse fixture, so we
        configure DSPy inside this fixture directly.
        """
        import asyncio

        import dspy
        import httpx

        from cogniverse_agents.orchestrator_agent import (
            OrchestratorAgent,
            OrchestratorDeps,
            OrchestratorInput,
        )
        from cogniverse_core.common.agent_models import AgentEndpoint
        from cogniverse_core.registries.agent_registry import AgentRegistry
        from cogniverse_foundation.config.llm_factory import create_dspy_lm
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig
        from cogniverse_foundation.config.utils import (
            create_default_config_manager,
            get_config,
        )

        cm = create_default_config_manager()
        cfg = get_config(tenant_id="test:unit", config_manager=cm)
        primary = cfg.get("llm_config", {}).get("primary", {})
        endpoint = LLMEndpointConfig(
            model=primary.get("model"),
            api_base=primary.get("api_base"),
            api_key=primary.get("api_key") or "not-required",
            temperature=0.0,
            max_tokens=800,
        )
        dspy.configure(lm=create_dspy_lm(endpoint))

        registry = AgentRegistry(tenant_id=BRIGHT_TENANT_ID, config_manager=cm)
        for name, caps in (
            ("search", ["search"]),
            ("query_enhancement", ["query_enhancement"]),
            ("entity_extraction", ["entity_extraction"]),
            ("profile_selection", ["profile_selection"]),
        ):
            registry.register_agent(
                AgentEndpoint(
                    name=name,
                    url=_ASGI_BASE,
                    capabilities=caps,
                    process_endpoint=f"/agents/{name}/process",
                )
            )

        # Build the in-process search app and an ASGI-bound httpx client.
        app = _build_bright_search_app()
        transport = httpx.ASGITransport(app=app)
        asgi_client = httpx.AsyncClient(
            transport=transport,
            base_url=_ASGI_BASE,
            timeout=60.0,
        )

        orchestrator = OrchestratorAgent(
            deps=OrchestratorDeps(),
            registry=registry,
            config_manager=cm,
            port=8013,
            http_client=asgi_client,
        )

        # Replace the orchestrator's planning phase with a deterministic
        # single-step ``[search]`` plan. Restricting only the DSPy
        # planner's ``available_agents`` was not enough: the gemma-class
        # student model still occasionally emits a multi-agent sequence
        # like ``"query_enhancement_agent,search_agent"``. After the
        # unknown ``query_enhancement_agent`` is filtered out, the
        # surviving ``search`` step retains its original
        # ``depends_on=[0]`` index pointing at the (now-removed) first
        # agent, so ``_execute_plan`` reports "No steps ready" and
        # returns an empty result set. The fix is to bypass the planner
        # entirely for the BRIGHT probe corpus: every query gets the
        # same fixed single-step plan.
        from cogniverse_agents.orchestrator_agent import (
            AgentStep,
            OrchestrationPlan,
        )

        async def _fixed_search_plan(
            query, conversation_context="", gateway_context=""
        ):
            return OrchestrationPlan(
                query=query,
                steps=[
                    AgentStep(
                        agent_name="search",
                        input_data={"query": query},
                        depends_on=[],
                        reasoning="bright probe single-step plan",
                    )
                ],
                parallel_groups=[],
                reasoning="bright probe fixture: single-step search-only plan",
                unavailable_agents=[],
            )

        orchestrator._create_plan = _fixed_search_plan

        loop = asyncio.new_event_loop()
        out: Dict[str, Dict[str, Any]] = {}
        try:
            asyncio.set_event_loop(loop)
            for row in probe_rows:
                result = loop.run_until_complete(
                    orchestrator.process(
                        OrchestratorInput(
                            query=row["query"], tenant_id=BRIGHT_TENANT_ID
                        )
                    )
                )
                iter_loop = (getattr(result, "metadata", {}) or {}).get(
                    "iterative_loop", {}
                ) or {}
                top_hits = iter_loop.get("top_hits") or []
                top1 = top_hits[0] if top_hits else {}
                out[row["query_id"]] = {
                    "top1_video_id": str(top1.get("video_id") or ""),
                    "top1_segment_id": str(top1.get("segment_id") or ""),
                    "iterations_executed": int(
                        iter_loop.get("iterations_executed") or 0
                    ),
                    "missing_aspects": list(iter_loop.get("missing_aspects") or []),
                    "final_answer_id": str(iter_loop.get("final_answer_id") or ""),
                }
            loop.run_until_complete(asgi_client.aclose())
        finally:
            asyncio.set_event_loop(None)
            loop.close()
            dspy.configure(lm=None)
        return out

    def test_per_query_recall_at_1(self, probe_rows, loop_runs):
        """Exactly 24 of 30 queries hit the ground-truth segment range.

        The 24 expected correct query_ids are pinned in
        ``goldens/bright_probes_correct_ids.json`` (list equality).
        """
        correct_ids: List[str] = []
        for row in probe_rows:
            res = loop_runs[row["query_id"]]
            top_video = res["top1_video_id"]
            top_segment = res["top1_segment_id"]
            if top_video == row["video_id"] and _segment_in_range(
                top_segment, row["segment_id_range"]
            ):
                correct_ids.append(row["query_id"])
        correct_ids.sort()
        assert len(correct_ids) == 24, (
            f"Expected recall@1 == 24, got {len(correct_ids)}: {correct_ids}"
        )
        assert_golden_json(correct_ids, "bright_probes_correct_ids.json")

    def test_per_reasoning_type_recall_at_1(self, probe_rows, loop_runs):
        """Per-type recall distribution byte-equal to the locked spec."""
        per_type: Dict[str, int] = {}
        for row in probe_rows:
            res = loop_runs[row["query_id"]]
            hit = res["top1_video_id"] == row["video_id"] and _segment_in_range(
                res["top1_segment_id"], row["segment_id_range"]
            )
            if hit:
                per_type[row["reasoning_type"]] = (
                    per_type.get(row["reasoning_type"], 0) + 1
                )
        # Ensure all five categories present even if zero (so a
        # regression to "0 lateral" still surfaces as a dict-equality
        # failure rather than a missing key).
        for cat in ("causal", "contradiction", "counterfactual", "lateral", "temporal"):
            per_type.setdefault(cat, 0)
        per_type = {k: int(v) for k, v in sorted(per_type.items())}
        expected = {
            "causal": 5,
            "contradiction": 3,
            "counterfactual": 4,
            "lateral": 5,
            "temporal": 7,
        }
        assert per_type == expected, f"Per-type recall mismatch: {per_type}"

    def test_baseline_lock(self, probe_rows, loop_runs):
        """Delta against the pinned baseline equals the locked delta.

        Reads ``goldens/bright_probes_baseline.json`` (created by the
        previous iterative-loop run that established the baseline);
        asserts ``current - K_baseline == delta_locked``. Runs under
        both normal replay and ``RECORD_GOLDEN=1`` — when recording is
        on, also rewrites the golden so the locked delta stays
        consistent with the freshly-measured ``current_correct`` value.
        """
        baseline_path = GOLDEN_DIR / "bright_probes_baseline.json"
        current_correct = 0
        for row in probe_rows:
            res = loop_runs[row["query_id"]]
            if res["top1_video_id"] == row["video_id"] and _segment_in_range(
                res["top1_segment_id"], row["segment_id_range"]
            ):
                current_correct += 1

        if RECORD_GOLDEN:
            # First-time record OR explicit re-record: lock the current
            # recall@1 as the baseline and a zero delta against itself.
            # Subsequent replay runs assert ``current - baseline == 0``.
            payload = {
                "correct_at_1": current_correct,
                "delta_to_target": 0,
                "note": (
                    "Auto-recorded baseline from a full BRIGHT-corpus run. "
                    "delta_to_target=0 means subsequent runs must reproduce "
                    "this exact recall@1 value byte-equal."
                ),
            }
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
            return

        if not baseline_path.exists():
            pytest.fail(
                f"Baseline golden missing: {baseline_path}. Re-run with "
                "RECORD_GOLDEN=1 to record it from the current corpus."
            )
        baseline = json.loads(baseline_path.read_text())
        k_baseline = int(baseline["correct_at_1"])
        delta_locked = int(baseline["delta_to_target"])
        assert current_correct - k_baseline == delta_locked, (
            f"Baseline delta mismatch: current={current_correct} "
            f"baseline={k_baseline} expected_delta={delta_locked} "
            f"actual_delta={current_correct - k_baseline}"
        )

    @pytest.mark.parametrize(
        "query_id", ["bright_q1", "bright_q5", "bright_q12", "bright_q24"]
    )
    def test_per_query_trajectory(self, query_id, probe_rows, loop_runs):
        """Trajectory dict byte-equal to per-query goldens for the 4 sentinels."""
        if query_id not in loop_runs:
            pytest.fail(
                f"Probe {query_id} not present in loop_runs — corpus "
                f"sort/filter changed unexpectedly."
            )
        res = loop_runs[query_id]
        trajectory = {
            "query_id": query_id,
            "iterations_executed": res["iterations_executed"],
            "missing_aspects": res["missing_aspects"],
            "final_answer_id": res["final_answer_id"],
            "top1_video_id": res["top1_video_id"],
            "top1_segment_id": res["top1_segment_id"],
        }
        # Golden filename matches the per-query trajectory contract in
        # the spec: ``goldens/bright_qN.json`` (the query_id already
        # starts with ``bright_``).
        assert_golden_json(trajectory, f"{query_id}.json")
