# A2A Plan 1: Foundation Agents — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the 5 independent foundation agents/components that unblock the full A2A restructuring: GatewayAgent, enhanced EntityExtractionAgent, enhanced QueryEnhancementAgent, wired ProfileSelectionAgent, simplified WorkflowIntelligence.

**Architecture:** Each agent is a standalone A2A service using the existing `A2AAgent` base class. Agents emit telemetry spans for batch optimization. No inline optimization loops. Each agent loads optimized DSPy modules from `ArtifactManager` at startup.

**Tech Stack:** Python 3.12, DSPy 3.0, GLiNER (`urchade/gliner_large-v2.1`), SpaCy, Phoenix/OpenTelemetry, Pydantic v2, pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-04-08-a2a-architecture-restructuring-design.md`

---

## File Structure

### New files
| File | Responsibility |
|------|---------------|
| `libs/agents/cogniverse_agents/gateway_agent.py` | GLiNER-based triage: simple vs complex classification |
| `tests/agents/unit/test_gateway_agent.py` | Unit tests for GatewayAgent |

### Modified files
| File | What changes |
|------|-------------|
| `libs/agents/cogniverse_agents/entity_extraction_agent.py` | Add GLiNER + SpaCy + LLM relationship extraction, tiered fast/slow path, span emission |
| `libs/agents/cogniverse_agents/query_enhancement_agent.py` | Add artifact-based DSPy module loading, RRF query variants, span emission, accept entities/relationships input |
| `libs/agents/cogniverse_agents/profile_selection_agent.py` | Add span emission |
| `libs/agents/cogniverse_agents/workflow/intelligence.py` | Remove inline DSPy optimization, become read-only template/profile loader |
| `libs/runtime/cogniverse_runtime/config_loader.py` | Add 5 new agent entries to AGENT_CLASSES |
| `configs/config.json` | Add 5 new agent config entries |
| `tests/agents/unit/test_entity_extraction_agent.py` | Add tests for GLiNER+SpaCy enhancement |
| `tests/agents/unit/test_query_enhancement.py` | Add tests for artifact-based enhancement |
| `tests/agents/unit/test_workflow_intelligence.py` | Update tests for simplified API |

---

### Task 1: Create GatewayAgent — Type Definitions and Failing Tests

**Files:**
- Create: `libs/agents/cogniverse_agents/gateway_agent.py`
- Create: `tests/agents/unit/test_gateway_agent.py`

- [ ] **Step 1: Write the GatewayAgent type definitions**

Create `libs/agents/cogniverse_agents/gateway_agent.py` with type stubs:

```python
"""
GatewayAgent — A2A agent for fast query triage.

Classifies queries as simple (direct to execution agent) or complex
(forward to OrchestratorAgent) using GLiNER entity classification.
No LLM call. Target latency: <100ms.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import Field

from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput

logger = logging.getLogger(__name__)


# ── Type-Safe Input / Output / Dependencies ──────────────────────────

class GatewayInput(AgentInput):
    """Input for gateway triage."""

    query: str = Field(..., description="User query to classify")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class GatewayOutput(AgentOutput):
    """Output from gateway triage."""

    query: str = Field(..., description="Original query")
    complexity: str = Field(..., description="simple or complex")
    modality: str = Field("video", description="Detected modality: video, text, audio, image, document, both")
    generation_type: str = Field("raw_results", description="raw_results, summary, or detailed_report")
    routed_to: str = Field(..., description="Target agent name or 'orchestrator_agent'")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Classification confidence")
    reasoning: str = Field("", description="Why this classification was made")


class GatewayDeps(AgentDeps):
    """Dependencies for GatewayAgent — no memory, pure classifier."""

    gliner_model_name: str = Field(
        default="urchade/gliner_large-v2.1",
        description="GLiNER model for entity classification",
    )
    gliner_threshold: float = Field(
        default=0.3,
        description="Minimum GLiNER entity confidence",
    )
    fast_path_confidence_threshold: float = Field(
        default=0.7,
        description="Confidence threshold for simple routing",
    )


# ── Modality / Generation Type Constants ─────────────────────────────

MODALITY_LABELS = {
    "video": ["video_content", "visual_content", "media_content"],
    "text": ["document_content", "text_information", "written_content"],
    "audio": ["audio_content", "sound_content", "music_content", "podcast_content"],
    "image": ["image_content", "photo_content", "picture_content", "diagram_content", "chart_content"],
    "document": ["pdf_content", "spreadsheet_content", "presentation_content"],
}

GENERATION_LABELS = {
    "summary": ["summary_request"],
    "detailed_report": ["detailed_analysis", "report_request"],
}

ALL_LABELS = [label for labels in MODALITY_LABELS.values() for label in labels] + \
             [label for labels in GENERATION_LABELS.values() for label in labels]

# Simple routing: modality + generation_type → agent name
SIMPLE_ROUTE_MAP = {
    ("video", "raw_results"): "search_agent",
    ("text", "raw_results"): "search_agent",
    ("audio", "raw_results"): "audio_analysis_agent",
    ("image", "raw_results"): "image_search_agent",
    ("document", "raw_results"): "document_agent",
    ("video", "summary"): "summarizer_agent",
    ("text", "summary"): "summarizer_agent",
    ("audio", "summary"): "summarizer_agent",
    ("image", "summary"): "summarizer_agent",
    ("document", "summary"): "summarizer_agent",
    ("video", "detailed_report"): "detailed_report_agent",
    ("text", "detailed_report"): "detailed_report_agent",
    ("audio", "detailed_report"): "detailed_report_agent",
    ("image", "detailed_report"): "detailed_report_agent",
    ("document", "detailed_report"): "detailed_report_agent",
}


class GatewayAgent(A2AAgent[GatewayInput, GatewayOutput, GatewayDeps]):
    """Fast triage agent using GLiNER classification."""

    def __init__(self, deps: GatewayDeps, port: int = 8014):
        config = A2AAgentConfig(
            agent_name="gateway_agent",
            agent_description="Fast query triage: simple vs complex classification",
            capabilities=["gateway", "classification"],
            port=port,
            version="1.0.0",
        )
        super().__init__(deps=deps, config=config)
        self._gliner_model = None

    async def _process_impl(self, input: GatewayInput) -> GatewayOutput:
        raise NotImplementedError("Step 3 implements this")
```

- [ ] **Step 2: Write failing tests for GatewayAgent**

Create `tests/agents/unit/test_gateway_agent.py`:

```python
"""Unit tests for GatewayAgent."""

from unittest.mock import MagicMock, patch

import pytest

from cogniverse_agents.gateway_agent import (
    ALL_LABELS,
    SIMPLE_ROUTE_MAP,
    GatewayAgent,
    GatewayDeps,
    GatewayInput,
    GatewayOutput,
)


def _make_gateway(**kwargs) -> GatewayAgent:
    """Create GatewayAgent with defaults."""
    defaults = dict()
    defaults.update(kwargs)
    deps = GatewayDeps(**defaults)
    return GatewayAgent(deps=deps)


@pytest.mark.unit
class TestGatewayAgentInit:
    """Test GatewayAgent initialization."""

    def test_gateway_agent_initializes(self):
        agent = _make_gateway()
        assert agent.agent_name == "gateway_agent"
        assert "gateway" in agent.capabilities

    def test_gateway_input_requires_query(self):
        with pytest.raises(Exception):
            GatewayInput()

    def test_gateway_output_fields(self):
        output = GatewayOutput(
            query="test",
            complexity="simple",
            modality="video",
            generation_type="raw_results",
            routed_to="search_agent",
            confidence=0.9,
        )
        assert output.complexity == "simple"
        assert output.routed_to == "search_agent"

    def test_simple_route_map_covers_all_modalities(self):
        modalities = {"video", "text", "audio", "image", "document"}
        gen_types = {"raw_results", "summary", "detailed_report"}
        for mod in modalities:
            for gt in gen_types:
                assert (mod, gt) in SIMPLE_ROUTE_MAP, f"Missing ({mod}, {gt})"

    def test_all_labels_not_empty(self):
        assert len(ALL_LABELS) > 0


@pytest.mark.unit
class TestGatewayClassification:
    """Test the core classification logic."""

    @pytest.mark.asyncio
    async def test_simple_video_query(self):
        """A clear video search query should route to search_agent."""
        agent = _make_gateway()
        # Mock GLiNER to return video_content with high confidence
        agent._gliner_model = MagicMock()
        agent._gliner_model.predict_entities.return_value = [
            {"text": "video", "label": "video_content", "score": 0.85},
        ]

        input_data = GatewayInput(query="find videos about machine learning")
        result = await agent._process_impl(input_data)

        assert result.complexity == "simple"
        assert result.modality == "video"
        assert result.routed_to == "search_agent"
        assert result.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_complex_multi_modal_query(self):
        """A query matching multiple modalities should be complex."""
        agent = _make_gateway()
        agent._gliner_model = MagicMock()
        agent._gliner_model.predict_entities.return_value = [
            {"text": "video", "label": "video_content", "score": 0.6},
            {"text": "document", "label": "document_content", "score": 0.5},
        ]

        input_data = GatewayInput(query="find videos and documents about AI")
        result = await agent._process_impl(input_data)

        assert result.complexity == "complex"
        assert result.routed_to == "orchestrator_agent"

    @pytest.mark.asyncio
    async def test_low_confidence_is_complex(self):
        """Low-confidence classification should route to orchestrator."""
        agent = _make_gateway()
        agent._gliner_model = MagicMock()
        agent._gliner_model.predict_entities.return_value = [
            {"text": "stuff", "label": "video_content", "score": 0.3},
        ]

        input_data = GatewayInput(query="what happened last Tuesday")
        result = await agent._process_impl(input_data)

        assert result.complexity == "complex"
        assert result.routed_to == "orchestrator_agent"

    @pytest.mark.asyncio
    async def test_summary_request_routes_to_summarizer(self):
        """A summary request should route to summarizer_agent."""
        agent = _make_gateway()
        agent._gliner_model = MagicMock()
        agent._gliner_model.predict_entities.return_value = [
            {"text": "video", "label": "video_content", "score": 0.8},
            {"text": "summarize", "label": "summary_request", "score": 0.75},
        ]

        input_data = GatewayInput(query="summarize the machine learning videos")
        result = await agent._process_impl(input_data)

        assert result.complexity == "simple"
        assert result.generation_type == "summary"
        assert result.routed_to == "summarizer_agent"

    @pytest.mark.asyncio
    async def test_no_entities_is_complex(self):
        """Query with no GLiNER entities should be complex."""
        agent = _make_gateway()
        agent._gliner_model = MagicMock()
        agent._gliner_model.predict_entities.return_value = []

        input_data = GatewayInput(query="hello")
        result = await agent._process_impl(input_data)

        assert result.complexity == "complex"
        assert result.routed_to == "orchestrator_agent"

    @pytest.mark.asyncio
    async def test_span_emission(self):
        """GatewayAgent should emit a cogniverse.gateway span."""
        agent = _make_gateway()
        agent._gliner_model = MagicMock()
        agent._gliner_model.predict_entities.return_value = [
            {"text": "video", "label": "video_content", "score": 0.9},
        ]

        mock_tm = MagicMock()
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_tm.span.return_value = mock_span
        agent.telemetry_manager = mock_tm

        input_data = GatewayInput(query="find ML videos", tenant_id="test")
        await agent._process_impl(input_data)

        mock_tm.span.assert_called_once()
        call_kwargs = mock_tm.span.call_args
        assert "cogniverse.gateway" in str(call_kwargs)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/agents/unit/test_gateway_agent.py --tb=long -q`
Expected: Tests in `TestGatewayClassification` FAIL with `NotImplementedError`; `TestGatewayAgentInit` tests PASS.

- [ ] **Step 4: Commit type definitions and failing tests**

```bash
git add libs/agents/cogniverse_agents/gateway_agent.py tests/agents/unit/test_gateway_agent.py
git commit -m "Add GatewayAgent type definitions and failing tests"
```

---

### Task 2: Implement GatewayAgent Classification Logic

**Files:**
- Modify: `libs/agents/cogniverse_agents/gateway_agent.py`

- [ ] **Step 1: Implement GLiNER model loading**

Add to `GatewayAgent` class in `gateway_agent.py`, after `self._gliner_model = None`:

```python
    def _ensure_model_loaded(self) -> None:
        """Load GLiNER model on first use (lazy initialization)."""
        if self._gliner_model is not None:
            return

        try:
            from gliner import GLiNER

            from cogniverse_core.common.models import model_load_lock

            with model_load_lock:
                self._gliner_model = GLiNER.from_pretrained(
                    self.deps.gliner_model_name
                )
            logger.info("GatewayAgent: loaded GLiNER model %s", self.deps.gliner_model_name)
        except Exception as e:
            logger.error("GatewayAgent: failed to load GLiNER model: %s", e)
            self._gliner_model = None
```

- [ ] **Step 2: Implement the core classification method**

Replace the `_process_impl` stub with:

```python
    async def _process_impl(self, input: GatewayInput) -> GatewayOutput:
        self.emit_progress("classify", "Classifying query complexity...")

        self._ensure_model_loaded()

        # Extract entities with GLiNER
        entities = self._extract_entities(input.query)

        # Classify modality and generation type from entities
        modality, modality_confidence = self._classify_modality(entities)
        generation_type, gen_confidence = self._classify_generation_type(entities)

        # Determine complexity
        overall_confidence = max(modality_confidence, gen_confidence)
        is_complex = self._is_complex(modality, entities, overall_confidence)

        if is_complex:
            routed_to = "orchestrator_agent"
            complexity = "complex"
        else:
            complexity = "simple"
            routed_to = SIMPLE_ROUTE_MAP.get(
                (modality, generation_type), "search_agent"
            )

        reasoning = (
            f"GLiNER classified modality={modality} (conf={modality_confidence:.2f}), "
            f"generation={generation_type} (conf={gen_confidence:.2f}), "
            f"entities={len(entities)}"
        )

        # Emit telemetry span
        self._emit_gateway_span(
            input.query, input.tenant_id, complexity, modality,
            generation_type, routed_to, overall_confidence,
        )

        return GatewayOutput(
            query=input.query,
            complexity=complexity,
            modality=modality,
            generation_type=generation_type,
            routed_to=routed_to,
            confidence=round(overall_confidence, 3),
            reasoning=reasoning,
        )

    def _extract_entities(self, query: str) -> list[dict[str, Any]]:
        """Extract entities using GLiNER. Returns empty list on failure."""
        if self._gliner_model is None:
            return []
        try:
            return self._gliner_model.predict_entities(
                query, ALL_LABELS, threshold=self.deps.gliner_threshold
            )
        except Exception as e:
            logger.error("GLiNER prediction failed: %s", e)
            return []

    def _classify_modality(
        self, entities: list[dict[str, Any]]
    ) -> tuple[str, float]:
        """Determine search modality from GLiNER entities."""
        modality_scores: Dict[str, float] = {}

        for entity in entities:
            label = entity.get("label", "")
            score = entity.get("score", 0.0)

            for modality, labels in MODALITY_LABELS.items():
                if label in labels:
                    modality_scores[modality] = max(
                        modality_scores.get(modality, 0.0), score
                    )

        if not modality_scores:
            return "video", 0.0  # default modality

        best_modality = max(modality_scores, key=modality_scores.get)
        return best_modality, modality_scores[best_modality]

    def _classify_generation_type(
        self, entities: list[dict[str, Any]]
    ) -> tuple[str, float]:
        """Determine generation type from GLiNER entities."""
        for entity in entities:
            label = entity.get("label", "")
            score = entity.get("score", 0.0)

            for gen_type, labels in GENERATION_LABELS.items():
                if label in labels and score >= self.deps.gliner_threshold:
                    return gen_type, score

        return "raw_results", 0.0

    def _is_complex(
        self,
        modality: str,
        entities: list[dict[str, Any]],
        confidence: float,
    ) -> bool:
        """Determine if query requires full orchestration."""
        # No entities → can't classify → complex
        if not entities:
            return True

        # Low confidence → uncertain → complex
        if confidence < self.deps.fast_path_confidence_threshold:
            return True

        # Multiple modalities detected → needs orchestration
        detected_modalities = set()
        for entity in entities:
            label = entity.get("label", "")
            for mod, labels in MODALITY_LABELS.items():
                if label in labels:
                    detected_modalities.add(mod)
        if len(detected_modalities) > 1:
            return True

        # "both" modality → complex
        if modality == "both":
            return True

        return False

    def _emit_gateway_span(
        self,
        query: str,
        tenant_id: Optional[str],
        complexity: str,
        modality: str,
        generation_type: str,
        routed_to: str,
        confidence: float,
    ) -> None:
        """Emit cogniverse.gateway telemetry span."""
        if not self.telemetry_manager or not tenant_id:
            return
        try:
            with self.telemetry_manager.span(
                name="cogniverse.gateway",
                tenant_id=tenant_id,
                attributes={
                    "gateway.query": query[:200],
                    "gateway.complexity": complexity,
                    "gateway.modality": modality,
                    "gateway.generation_type": generation_type,
                    "gateway.routed_to": routed_to,
                    "gateway.confidence": confidence,
                },
            ):
                pass
        except Exception as e:
            logger.debug("Failed to emit gateway span: %s", e)
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/agents/unit/test_gateway_agent.py --tb=long -q`
Expected: All tests PASS.

- [ ] **Step 4: Commit implementation**

```bash
git add libs/agents/cogniverse_agents/gateway_agent.py
git commit -m "Implement GatewayAgent GLiNER classification logic"
```

---

### Task 3: Register GatewayAgent in Config and Dispatcher

**Files:**
- Modify: `libs/runtime/cogniverse_runtime/config_loader.py:34-45`
- Modify: `configs/config.json`

- [ ] **Step 1: Add GatewayAgent to AGENT_CLASSES**

In `libs/runtime/cogniverse_runtime/config_loader.py`, add to the `AGENT_CLASSES` dict:

```python
    AGENT_CLASSES = {
        "gateway_agent": "cogniverse_agents.gateway_agent:GatewayAgent",
        "routing_agent": "cogniverse_agents.routing_agent:RoutingAgent",
        # ... rest unchanged
    }
```

- [ ] **Step 2: Add GatewayAgent config entry**

In `configs/config.json`, add to the `agents` section:

```json
    "gateway_agent": {
      "enabled": true,
      "url": "http://localhost:8000",
      "capabilities": ["gateway", "classification"],
      "timeout": 10
    }
```

- [ ] **Step 3: Verify import works**

Run: `uv run python -c "from cogniverse_agents.gateway_agent import GatewayAgent; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit registration**

```bash
git add libs/runtime/cogniverse_runtime/config_loader.py configs/config.json
git commit -m "Register GatewayAgent in config loader and config.json"
```

---

### Task 4: Enhance EntityExtractionAgent — GLiNER + SpaCy Integration

**Files:**
- Modify: `libs/agents/cogniverse_agents/entity_extraction_agent.py`
- Modify: `tests/agents/unit/test_entity_extraction_agent.py`

- [ ] **Step 1: Write failing tests for GLiNER+SpaCy entity extraction**

Add to `tests/agents/unit/test_entity_extraction_agent.py`:

```python
"""Tests for enhanced EntityExtractionAgent with GLiNER + SpaCy."""

from unittest.mock import MagicMock, patch

import pytest

from cogniverse_agents.entity_extraction_agent import (
    EntityExtractionAgent,
    EntityExtractionDeps,
    EntityExtractionInput,
    EntityExtractionOutput,
    Relationship,
)


def _make_extraction_agent(**kwargs) -> EntityExtractionAgent:
    defaults = dict()
    defaults.update(kwargs)
    deps = EntityExtractionDeps(**defaults)
    return EntityExtractionAgent(deps=deps)


@pytest.mark.unit
class TestEnhancedEntityExtraction:
    """Test GLiNER + SpaCy enhanced extraction."""

    @pytest.mark.asyncio
    async def test_gliner_fast_path_extracts_entities(self):
        """GLiNER fast path should extract typed entities."""
        agent = _make_extraction_agent()
        # Mock GLiNER extractor
        agent._gliner_extractor = MagicMock()
        agent._gliner_extractor.extract_entities.return_value = [
            {"text": "TensorFlow", "label": "TECHNOLOGY", "score": 0.92},
            {"text": "Google", "label": "ORG", "score": 0.88},
        ]
        # Mock SpaCy analyzer
        agent._spacy_analyzer = MagicMock()
        agent._spacy_analyzer.extract_semantic_relationships.return_value = [
            {"subject": "TensorFlow", "relation": "developed_by", "object": "Google"},
        ]

        input_data = EntityExtractionInput(query="TensorFlow by Google tutorials")
        result = await agent._process_impl(input_data)

        assert result.entity_count == 2
        assert result.has_entities is True
        assert any(e.text == "TensorFlow" for e in result.entities)
        assert len(result.relationships) >= 1

    @pytest.mark.asyncio
    async def test_output_includes_relationships(self):
        """Enhanced output should include Relationship objects."""
        agent = _make_extraction_agent()
        agent._gliner_extractor = MagicMock()
        agent._gliner_extractor.extract_entities.return_value = [
            {"text": "Python", "label": "TECHNOLOGY", "score": 0.9},
            {"text": "machine learning", "label": "CONCEPT", "score": 0.85},
        ]
        agent._spacy_analyzer = MagicMock()
        agent._spacy_analyzer.extract_semantic_relationships.return_value = [
            {"subject": "Python", "relation": "used_for", "object": "machine learning"},
        ]

        input_data = EntityExtractionInput(query="Python for machine learning")
        result = await agent._process_impl(input_data)

        assert hasattr(result, "relationships")
        assert len(result.relationships) >= 1
        rel = result.relationships[0]
        assert rel.subject == "Python"
        assert rel.relation == "used_for"
        assert rel.object == "machine learning"

    @pytest.mark.asyncio
    async def test_fallback_when_gliner_unavailable(self):
        """Should fall back to DSPy when GLiNER is not loaded."""
        agent = _make_extraction_agent()
        agent._gliner_extractor = None
        agent._spacy_analyzer = None

        # Mock the DSPy module
        mock_result = MagicMock()
        mock_result.entities = "Python|TECHNOLOGY|0.8"
        mock_result.entity_types = "TECHNOLOGY"
        agent.dspy_module = MagicMock()
        agent.dspy_module.forward.return_value = mock_result

        input_data = EntityExtractionInput(query="Python programming")
        result = await agent._process_impl(input_data)

        assert result.entity_count >= 1
        assert result.has_entities is True

    @pytest.mark.asyncio
    async def test_span_emission(self):
        """Should emit cogniverse.entity_extraction span."""
        agent = _make_extraction_agent()
        agent._gliner_extractor = MagicMock()
        agent._gliner_extractor.extract_entities.return_value = []
        agent._spacy_analyzer = MagicMock()
        agent._spacy_analyzer.extract_semantic_relationships.return_value = []

        mock_tm = MagicMock()
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_tm.span.return_value = mock_span
        agent.telemetry_manager = mock_tm

        input_data = EntityExtractionInput(query="test", tenant_id="t1")
        await agent._process_impl(input_data)

        mock_tm.span.assert_called_once()
        call_args = mock_tm.span.call_args
        assert "cogniverse.entity_extraction" in str(call_args)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agents/unit/test_entity_extraction_agent.py::TestEnhancedEntityExtraction --tb=long -q`
Expected: FAIL — `Relationship` not defined, `_gliner_extractor` not found.

- [ ] **Step 3: Implement enhanced EntityExtractionAgent**

Modify `libs/agents/cogniverse_agents/entity_extraction_agent.py`. Add the `Relationship` model, GLiNER/SpaCy initialization, and tiered extraction:

```python
# Add after Entity class definition:

class Relationship(BaseModel):
    """Extracted relationship between entities."""

    subject: str = Field(description="Source entity")
    relation: str = Field(description="Relationship type")
    object: str = Field(description="Target entity")
    confidence: float = Field(default=0.5, description="Confidence score 0-1")


# Update EntityExtractionOutput to include relationships:

class EntityExtractionOutput(AgentOutput):
    """Type-safe output from entity extraction."""

    query: str = Field(..., description="Original query")
    entities: List[Entity] = Field(default_factory=list, description="Extracted entities")
    relationships: List[Relationship] = Field(default_factory=list, description="Extracted relationships")
    entity_count: int = Field(0, description="Number of entities found")
    has_entities: bool = Field(False, description="Whether entities were found")
    dominant_types: List[str] = Field(default_factory=list, description="Most common entity types")
    path_used: str = Field("dspy", description="Extraction path: fast (GLiNER+SpaCy) or dspy")
```

Update the `__init__` to initialize GLiNER and SpaCy:

```python
class EntityExtractionAgent(
    MemoryAwareMixin,
    A2AAgent[EntityExtractionInput, EntityExtractionOutput, EntityExtractionDeps],
):
    def __init__(self, deps: EntityExtractionDeps, port: int = 8010):
        extraction_module = EntityExtractionModule()
        config = A2AAgentConfig(
            agent_name="entity_extraction_agent",
            agent_description="Entity extraction with GLiNER + SpaCy + DSPy fallback",
            capabilities=["entity_extraction", "named_entity_recognition", "relationship_extraction"],
            port=port,
            version="2.0.0",
        )
        super().__init__(deps=deps, config=config, dspy_module=extraction_module)
        self._gliner_extractor = None
        self._spacy_analyzer = None
        self._initialize_extractors()

    def _initialize_extractors(self) -> None:
        """Initialize GLiNER and SpaCy extractors (lazy, non-fatal)."""
        try:
            from cogniverse_agents.routing.dspy_relationship_router import (
                GLiNERRelationshipExtractor,
                SpaCyDependencyAnalyzer,
            )

            self._gliner_extractor = GLiNERRelationshipExtractor()
            self._spacy_analyzer = SpaCyDependencyAnalyzer()
            logger.info("EntityExtractionAgent: GLiNER + SpaCy initialized")
        except Exception as e:
            logger.warning("EntityExtractionAgent: GLiNER/SpaCy unavailable, using DSPy fallback: %s", e)
            self._gliner_extractor = None
            self._spacy_analyzer = None
```

Replace `_process_impl` with tiered extraction:

```python
    async def _process_impl(self, input: EntityExtractionInput) -> EntityExtractionOutput:
        self.emit_progress("extract", "Extracting entities...")

        # Inject memory context if available
        if input.tenant_id is not None:
            self.set_tenant_for_context(input.tenant_id)

        # Try fast path (GLiNER + SpaCy) first
        if self._gliner_extractor is not None:
            entities, relationships, path = self._fast_path_extract(input.query)
        else:
            entities, relationships, path = await self._dspy_fallback_extract(input.query)

        # Build output
        entity_types = [e.type for e in entities]
        type_counts = {}
        for t in entity_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        dominant = sorted(type_counts, key=type_counts.get, reverse=True)[:3] if type_counts else []

        # Emit span
        self._emit_extraction_span(input.query, input.tenant_id, entities, relationships, path)

        return EntityExtractionOutput(
            query=input.query,
            entities=entities,
            relationships=relationships,
            entity_count=len(entities),
            has_entities=len(entities) > 0,
            dominant_types=dominant,
            path_used=path,
        )

    def _fast_path_extract(self, query: str) -> tuple[List[Entity], List[Relationship], str]:
        """Fast path: GLiNER entities + SpaCy relationships. No LLM call."""
        entities = []
        relationships = []

        # GLiNER entity extraction
        try:
            raw_entities = self._gliner_extractor.extract_entities(query)
            for e in raw_entities:
                entities.append(Entity(
                    text=e.get("text", ""),
                    type=e.get("label", "CONCEPT"),
                    confidence=e.get("score", 0.5),
                ))
        except Exception as e:
            logger.error("GLiNER extraction failed: %s", e)

        # SpaCy relationship extraction
        if self._spacy_analyzer and len(entities) >= 2:
            try:
                raw_rels = self._spacy_analyzer.extract_semantic_relationships(query)
                for r in raw_rels:
                    if all(k in r for k in ("subject", "relation", "object")):
                        relationships.append(Relationship(
                            subject=r["subject"],
                            relation=r["relation"],
                            object=r["object"],
                            confidence=r.get("confidence", 0.5),
                        ))
            except Exception as e:
                logger.error("SpaCy relationship extraction failed: %s", e)

        return entities, relationships, "fast"

    async def _dspy_fallback_extract(self, query: str) -> tuple[List[Entity], List[Relationship], str]:
        """DSPy fallback when GLiNER/SpaCy unavailable."""
        entities = []
        try:
            result = await self.call_dspy(
                self.dspy_module, output_field="entities", query=query
            )
            raw = getattr(result, "entities", "")
            entities = self._parse_entities(raw, query)
        except Exception as e:
            logger.error("DSPy entity extraction failed: %s", e)

        return entities, [], "dspy"

    def _emit_extraction_span(
        self, query: str, tenant_id: Optional[str],
        entities: List[Entity], relationships: List[Relationship], path: str,
    ) -> None:
        """Emit cogniverse.entity_extraction telemetry span."""
        if not self.telemetry_manager or not tenant_id:
            return
        try:
            import json
            entities_json = json.dumps([{"text": e.text, "type": e.type, "confidence": e.confidence} for e in entities])
            with self.telemetry_manager.span(
                name="cogniverse.entity_extraction",
                tenant_id=tenant_id,
                attributes={
                    "entity_extraction.query": query[:200],
                    "entity_extraction.entity_count": len(entities),
                    "entity_extraction.relationship_count": len(relationships),
                    "entity_extraction.entities": entities_json[:1000],
                    "entity_extraction.path_used": path,
                },
            ):
                pass
        except Exception as e:
            logger.debug("Failed to emit entity_extraction span: %s", e)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/unit/test_entity_extraction_agent.py --tb=long -q`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/agents/cogniverse_agents/entity_extraction_agent.py tests/agents/unit/test_entity_extraction_agent.py
git commit -m "Enhance EntityExtractionAgent with GLiNER + SpaCy tiered extraction"
```

---

### Task 5: Enhance QueryEnhancementAgent — Artifact Loading + RRF Variants

**Files:**
- Modify: `libs/agents/cogniverse_agents/query_enhancement_agent.py`
- Modify: `tests/agents/unit/test_query_enhancement.py`

- [ ] **Step 1: Write failing tests for enhanced query enhancement**

Add to `tests/agents/unit/test_query_enhancement.py`:

```python
@pytest.mark.unit
class TestEnhancedQueryEnhancementAgent:
    """Test artifact-based query enhancement with entity input."""

    @pytest.mark.asyncio
    async def test_accepts_entities_and_relationships(self):
        """Enhanced input should accept entities and relationships."""
        from cogniverse_agents.query_enhancement_agent import (
            QueryEnhancementAgent,
            QueryEnhancementDeps,
            QueryEnhancementInput,
        )

        deps = QueryEnhancementDeps()
        agent = QueryEnhancementAgent(deps=deps)

        # Mock DSPy module
        mock_result = MagicMock()
        mock_result.enhanced_query = "find TensorFlow deep learning tutorials"
        mock_result.expansion_terms = "deep learning, neural networks"
        mock_result.synonyms = "ML, AI"
        mock_result.context = "technology education"
        mock_result.confidence = "0.85"
        mock_result.reasoning = "Added related terms"
        agent.dspy_module = MagicMock()
        agent.dspy_module.forward.return_value = mock_result

        input_data = QueryEnhancementInput(
            query="find TensorFlow tutorials",
            entities=[{"text": "TensorFlow", "type": "TECHNOLOGY", "confidence": 0.9}],
            relationships=[{"subject": "TensorFlow", "relation": "is_a", "object": "framework"}],
        )
        result = await agent._process_impl(input_data)

        assert result.enhanced_query != ""
        assert result.original_query == "find TensorFlow tutorials"

    @pytest.mark.asyncio
    async def test_generates_query_variants(self):
        """Should generate RRF query variants."""
        from cogniverse_agents.query_enhancement_agent import (
            QueryEnhancementAgent,
            QueryEnhancementDeps,
            QueryEnhancementInput,
        )

        deps = QueryEnhancementDeps()
        agent = QueryEnhancementAgent(deps=deps)

        mock_result = MagicMock()
        mock_result.enhanced_query = "TensorFlow deep learning tutorial videos"
        mock_result.expansion_terms = "neural networks, deep learning"
        mock_result.synonyms = "ML, AI"
        mock_result.context = ""
        mock_result.confidence = "0.8"
        mock_result.reasoning = "Expanded terms"
        agent.dspy_module = MagicMock()
        agent.dspy_module.forward.return_value = mock_result

        input_data = QueryEnhancementInput(query="TensorFlow tutorials")
        result = await agent._process_impl(input_data)

        assert hasattr(result, "query_variants")
        assert isinstance(result.query_variants, list)

    @pytest.mark.asyncio
    async def test_span_emission(self):
        """Should emit cogniverse.query_enhancement span."""
        from cogniverse_agents.query_enhancement_agent import (
            QueryEnhancementAgent,
            QueryEnhancementDeps,
            QueryEnhancementInput,
        )

        deps = QueryEnhancementDeps()
        agent = QueryEnhancementAgent(deps=deps)

        mock_result = MagicMock()
        mock_result.enhanced_query = "enhanced"
        mock_result.expansion_terms = ""
        mock_result.synonyms = ""
        mock_result.context = ""
        mock_result.confidence = "0.7"
        mock_result.reasoning = ""
        agent.dspy_module = MagicMock()
        agent.dspy_module.forward.return_value = mock_result

        mock_tm = MagicMock()
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_tm.span.return_value = mock_span
        agent.telemetry_manager = mock_tm

        input_data = QueryEnhancementInput(query="test", tenant_id="t1")
        await agent._process_impl(input_data)

        mock_tm.span.assert_called_once()
        call_args = mock_tm.span.call_args
        assert "cogniverse.query_enhancement" in str(call_args)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agents/unit/test_query_enhancement.py::TestEnhancedQueryEnhancementAgent --tb=long -q`
Expected: FAIL — `entities` field not in `QueryEnhancementInput`, `query_variants` not in output.

- [ ] **Step 3: Implement enhanced QueryEnhancementAgent**

Update `libs/agents/cogniverse_agents/query_enhancement_agent.py`:

Add `entities` and `relationships` to input, `query_variants` to output:

```python
class QueryEnhancementInput(AgentInput):
    """Type-safe input for query enhancement."""

    query: str = Field(..., description="Query to enhance")
    entities: Optional[List[Dict[str, Any]]] = Field(None, description="Entities from EntityExtractionAgent")
    relationships: Optional[List[Dict[str, Any]]] = Field(None, description="Relationships from EntityExtractionAgent")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class QueryEnhancementOutput(AgentOutput):
    """Type-safe output from query enhancement."""

    original_query: str = Field(..., description="Original query")
    enhanced_query: str = Field(..., description="Enhanced query")
    expansion_terms: List[str] = Field(default_factory=list, description="Additional search terms")
    synonyms: List[str] = Field(default_factory=list, description="Synonym replacements")
    context_additions: List[str] = Field(default_factory=list, description="Contextual additions")
    query_variants: List[str] = Field(default_factory=list, description="RRF query variants for fusion search")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Enhancement confidence")
    reasoning: str = Field("", description="Explanation of enhancements")
```

Update `_process_impl` to use entities, generate variants, and emit spans:

```python
    async def _process_impl(self, input: QueryEnhancementInput) -> QueryEnhancementOutput:
        self.emit_progress("enhance", "Enhancing query...")

        if input.tenant_id is not None:
            self.set_tenant_for_context(input.tenant_id)

        # Build entity context for the DSPy module
        entity_context = ""
        if input.entities:
            entity_texts = [e.get("text", "") for e in input.entities[:5]]
            entity_context = f"Entities: {', '.join(entity_texts)}"
        if input.relationships:
            rel_texts = [
                f"{r.get('subject', '')} {r.get('relation', '')} {r.get('object', '')}"
                for r in input.relationships[:3]
            ]
            entity_context += f" Relationships: {'; '.join(rel_texts)}"

        # Inject memory strategies if available
        enriched_query = input.query
        if entity_context:
            enriched_query = f"{input.query} [{entity_context}]"

        # Call DSPy enhancement module
        try:
            result = await self.call_dspy(
                self.dspy_module,
                output_field="enhanced_query",
                query=enriched_query,
            )

            enhanced_query = getattr(result, "enhanced_query", input.query) or input.query
            expansion_terms = self._parse_list(getattr(result, "expansion_terms", ""))
            synonyms = self._parse_list(getattr(result, "synonyms", ""))
            context_additions = self._parse_list(getattr(result, "context", ""))
            confidence = self._parse_float(getattr(result, "confidence", "0.5"))
            reasoning = getattr(result, "reasoning", "")

        except Exception as e:
            logger.error("DSPy enhancement failed: %s", e)
            enhanced_query = input.query
            expansion_terms, synonyms, context_additions = [], [], []
            confidence, reasoning = 0.1, f"Fallback: {e}"

        # Generate RRF query variants
        query_variants = self._generate_variants(input.query, enhanced_query, expansion_terms)

        # Emit span
        self._emit_enhancement_span(
            input.query, input.tenant_id, enhanced_query, len(query_variants), confidence,
        )

        return QueryEnhancementOutput(
            original_query=input.query,
            enhanced_query=enhanced_query,
            expansion_terms=expansion_terms,
            synonyms=synonyms,
            context_additions=context_additions,
            query_variants=query_variants,
            confidence=round(confidence, 3),
            reasoning=reasoning,
        )

    def _generate_variants(
        self, original: str, enhanced: str, expansion_terms: List[str]
    ) -> List[str]:
        """Generate query variants for Reciprocal Rank Fusion search."""
        variants = []
        if enhanced != original:
            variants.append(enhanced)
        # Add expansion-augmented variant
        if expansion_terms:
            expanded = f"{original} {' '.join(expansion_terms[:3])}"
            if expanded not in variants:
                variants.append(expanded)
        return variants

    def _emit_enhancement_span(
        self, query: str, tenant_id: Optional[str],
        enhanced_query: str, variant_count: int, confidence: float,
    ) -> None:
        """Emit cogniverse.query_enhancement telemetry span."""
        if not self.telemetry_manager or not tenant_id:
            return
        try:
            with self.telemetry_manager.span(
                name="cogniverse.query_enhancement",
                tenant_id=tenant_id,
                attributes={
                    "query_enhancement.original_query": query[:200],
                    "query_enhancement.enhanced_query": enhanced_query[:200],
                    "query_enhancement.variant_count": variant_count,
                    "query_enhancement.confidence": confidence,
                },
            ):
                pass
        except Exception as e:
            logger.debug("Failed to emit query_enhancement span: %s", e)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/unit/test_query_enhancement.py::TestEnhancedQueryEnhancementAgent --tb=long -q`
Expected: All tests PASS.

- [ ] **Step 5: Run all existing query enhancement tests to verify no regressions**

Run: `uv run pytest tests/agents/unit/test_query_enhancement.py --tb=long -q`
Expected: All tests PASS (existing + new).

- [ ] **Step 6: Commit**

```bash
git add libs/agents/cogniverse_agents/query_enhancement_agent.py tests/agents/unit/test_query_enhancement.py
git commit -m "Enhance QueryEnhancementAgent with entity input, RRF variants, span emission"
```

---

### Task 6: Wire ProfileSelectionAgent — Span Emission + Registration

**Files:**
- Modify: `libs/agents/cogniverse_agents/profile_selection_agent.py`
- Modify: `libs/runtime/cogniverse_runtime/config_loader.py`
- Modify: `configs/config.json`

- [ ] **Step 1: Add span emission to ProfileSelectionAgent**

Add to `libs/agents/cogniverse_agents/profile_selection_agent.py`, at the end of the `_process_impl` method (before the final return):

```python
        # Emit telemetry span
        self._emit_profile_span(
            input.query, input.tenant_id, selected_profile,
            query_intent, modality, complexity, confidence,
        )
```

And add the span method to the class:

```python
    def _emit_profile_span(
        self, query: str, tenant_id: Optional[str],
        selected_profile: str, intent: str, modality: str,
        complexity: str, confidence: float,
    ) -> None:
        """Emit cogniverse.profile_selection telemetry span."""
        if not self.telemetry_manager or not tenant_id:
            return
        try:
            with self.telemetry_manager.span(
                name="cogniverse.profile_selection",
                tenant_id=tenant_id,
                attributes={
                    "profile_selection.query": query[:200],
                    "profile_selection.selected_profile": selected_profile,
                    "profile_selection.modality": modality,
                    "profile_selection.complexity": complexity,
                    "profile_selection.intent": intent,
                    "profile_selection.confidence": confidence,
                },
            ):
                pass
        except Exception as e:
            logger.debug("Failed to emit profile_selection span: %s", e)
```

- [ ] **Step 2: Register in config_loader.py and config.json**

Add to `config_loader.py` AGENT_CLASSES:
```python
        "profile_selection_agent": "cogniverse_agents.profile_selection_agent:ProfileSelectionAgent",
```

Add to `configs/config.json` agents section:
```json
    "profile_selection_agent": {
      "enabled": true,
      "url": "http://localhost:8000",
      "capabilities": ["profile_selection"],
      "timeout": 15
    }
```

- [ ] **Step 3: Register EntityExtractionAgent and QueryEnhancementAgent too**

Add to `config_loader.py` AGENT_CLASSES:
```python
        "entity_extraction_agent": "cogniverse_agents.entity_extraction_agent:EntityExtractionAgent",
        "query_enhancement_agent": "cogniverse_agents.query_enhancement_agent:QueryEnhancementAgent",
```

Add to `configs/config.json` agents section:
```json
    "entity_extraction_agent": {
      "enabled": true,
      "url": "http://localhost:8000",
      "capabilities": ["entity_extraction", "ner"],
      "timeout": 15
    },
    "query_enhancement_agent": {
      "enabled": true,
      "url": "http://localhost:8000",
      "capabilities": ["query_enhancement", "expansion"],
      "timeout": 15
    }
```

- [ ] **Step 4: Verify all imports work**

Run:
```bash
uv run python -c "
from cogniverse_agents.gateway_agent import GatewayAgent
from cogniverse_agents.entity_extraction_agent import EntityExtractionAgent
from cogniverse_agents.query_enhancement_agent import QueryEnhancementAgent
from cogniverse_agents.profile_selection_agent import ProfileSelectionAgent
print('All agents importable')
"
```
Expected: `All agents importable`

- [ ] **Step 5: Commit**

```bash
git add libs/agents/cogniverse_agents/profile_selection_agent.py libs/runtime/cogniverse_runtime/config_loader.py configs/config.json
git commit -m "Wire ProfileSelectionAgent, register all preprocessing agents"
```

---

### Task 7: Simplify WorkflowIntelligence — Remove Inline Optimization

**Files:**
- Modify: `libs/agents/cogniverse_agents/workflow/intelligence.py`
- Modify: `tests/agents/unit/test_workflow_intelligence.py`

- [ ] **Step 1: Write tests for simplified WorkflowIntelligence**

Add to `tests/agents/unit/test_workflow_intelligence.py`:

```python
@pytest.mark.unit
class TestSimplifiedWorkflowIntelligence:
    """Test WorkflowIntelligence as a read-only template/profile loader."""

    def test_initialization_without_optimization(self):
        """WorkflowIntelligence should initialize without DSPy optimizer."""
        intelligence = _make_intelligence()
        assert intelligence is not None
        # Should NOT have an optimizer that runs inline
        assert not hasattr(intelligence, "simba_optimizer") or intelligence.simba_optimizer is None

    @pytest.mark.asyncio
    async def test_load_templates_from_artifacts(self):
        """Should load workflow templates from ArtifactManager."""
        intelligence = _make_intelligence()
        # Templates should be loadable (empty list if none stored)
        templates = intelligence.get_workflow_templates()
        assert isinstance(templates, list)

    @pytest.mark.asyncio
    async def test_find_matching_template(self):
        """Should find matching templates for similar queries."""
        from cogniverse_agents.workflow.intelligence import WorkflowTemplate

        intelligence = _make_intelligence()
        template = WorkflowTemplate(
            template_id="t1",
            name="multi_modal_search",
            description="Multi-modal search workflow",
            query_patterns=["find videos and documents"],
            task_sequence=[{"agent": "search_agent"}, {"agent": "document_agent"}],
            expected_execution_time=3.0,
            success_rate=0.9,
        )
        intelligence.workflow_templates.append(template)

        match = intelligence._find_matching_template("find videos and documents about AI")
        assert match is not None
        assert match.template_id == "t1"

    def test_get_agent_performance_report(self):
        """Agent performance should be retrievable."""
        intelligence = _make_intelligence()
        report = intelligence.get_agent_performance_report()
        assert isinstance(report, dict)
```

- [ ] **Step 2: Run tests to verify current state**

Run: `uv run pytest tests/agents/unit/test_workflow_intelligence.py --tb=long -q`
Expected: Existing tests pass. New `TestSimplifiedWorkflowIntelligence` tests may fail if `get_workflow_templates()` doesn't exist yet.

- [ ] **Step 3: Simplify WorkflowIntelligence**

In `libs/agents/cogniverse_agents/workflow/intelligence.py`, make these changes:

1. Add `get_workflow_templates()` method:
```python
    def get_workflow_templates(self) -> List[WorkflowTemplate]:
        """Return loaded workflow templates."""
        return list(self.workflow_templates)
```

2. Remove inline DSPy optimization from `optimize_workflow_plan()` — replace with template-only optimization:
```python
    async def optimize_workflow_plan(
        self,
        query: str,
        initial_plan: WorkflowPlan,
        optimization_context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowPlan:
        """Optimize workflow plan using templates (no inline DSPy optimization)."""
        self.optimization_stats["total_optimizations"] += 1

        try:
            # Check for existing templates
            template_match = self._find_matching_template(query)
            if template_match:
                optimized_plan = self._apply_template(initial_plan, template_match)
                self.optimization_stats["templates_used"] += 1
                self.logger.info(f"Applied template '{template_match.name}'")
                return optimized_plan

            # Apply optimization strategy (no DSPy call — just heuristics)
            final_plan = self._apply_optimization_strategy(initial_plan)
            self.optimization_stats["successful_optimizations"] += 1
            return final_plan

        except Exception as e:
            self.logger.error(f"Workflow optimization failed: {e}")
            return initial_plan
```

3. Remove `_dspy_optimize_workflow()` method entirely.

4. Remove inline `record_workflow_execution()` recording to history — keep the method signature but make it a no-op that just logs (spans are the record now):
```python
    async def record_workflow_execution(self, workflow_plan: WorkflowPlan) -> None:
        """Record workflow execution via telemetry span (no inline storage)."""
        self.logger.info(
            "Workflow %s completed (recording via telemetry spans, not inline)",
            workflow_plan.workflow_id,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/unit/test_workflow_intelligence.py --tb=long -q`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/agents/cogniverse_agents/workflow/intelligence.py tests/agents/unit/test_workflow_intelligence.py
git commit -m "Simplify WorkflowIntelligence to read-only template loader"
```

---

### Task 8: Run Full Test Suite for All Foundation Agent Changes

**Files:** None (verification only)

- [ ] **Step 1: Run all affected unit tests**

```bash
uv run pytest tests/agents/unit/test_gateway_agent.py \
    tests/agents/unit/test_entity_extraction_agent.py \
    tests/agents/unit/test_query_enhancement.py \
    tests/agents/unit/test_workflow_intelligence.py \
    --tb=long -q
```
Expected: All tests PASS, 0 failed, 0 skipped.

- [ ] **Step 2: Run ruff lint on changed files**

```bash
uv run ruff check \
    libs/agents/cogniverse_agents/gateway_agent.py \
    libs/agents/cogniverse_agents/entity_extraction_agent.py \
    libs/agents/cogniverse_agents/query_enhancement_agent.py \
    libs/agents/cogniverse_agents/profile_selection_agent.py \
    libs/agents/cogniverse_agents/workflow/intelligence.py \
    libs/runtime/cogniverse_runtime/config_loader.py
```
Expected: 0 errors.

- [ ] **Step 3: Run routing unit tests to verify no regressions**

```bash
uv run pytest tests/routing/unit/ --tb=long -q
```
Expected: All tests PASS.

- [ ] **Step 4: Commit any lint fixes**

```bash
git add -A && git diff --cached --stat
# If changes: git commit -m "Fix lint errors in foundation agents"
# If clean: echo "No lint fixes needed"
```
