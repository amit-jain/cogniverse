# Cogniverse Glossary

A reference guide to terms and concepts used throughout the Cogniverse codebase.

---

## A

### A2A (Agent-to-Agent)
Google's protocol for inter-agent communication. Defines a standard message format for agents to communicate, enabling agents from different systems to work together. Cogniverse implements A2A through the `A2AAgent` base class.

### AgentBase
The abstract generic base class for all agents: `AgentBase[InputT, OutputT, DepsT]`. Provides type safety for inputs, outputs, and dependencies.

### AgentDispatcher
Class (`cogniverse_runtime/agent_dispatcher.py`) that routes agent tasks to the correct in-process agent implementation using `AgentRegistry`, `ConfigManager`, and `SchemaLoader`. Most agents are instantiated fresh, stateless, per request; `GatewayAgent`, `SearchAgent`, and the generic A2A agents (`entity_extraction`, `query_enhancement`, `profile_selection`, etc.) are instead cached per tenant (or per `(tenant, agent_name)`) in a TTL-gated `TenantLRUCache`, with cache hits re-reading only the artifact. Invokes `OrchestratorAgent` when `GatewayAgent` classifies a query as complex.

### AgentDeps
Pydantic model for agent dependencies (LLMs, backends, other agents). Injected at runtime to enable testing and configuration.

### AgentInput
Pydantic model that all agent inputs must extend. Provides validation and serialization for agent requests.

### AgentOutput
Pydantic model that all agent outputs must extend. Provides validation and serialization for agent responses.

### AgentRegistry
Central registry for discovering and accessing agents. Agents register themselves with capabilities for routing.

### AudioAnalysisAgent
A2A agent (`cogniverse_agents/audio_analysis_agent.py`) that transcribes audio with Whisper and searches it in Vespa via transcript (BM25), acoustic (CLAP nearest-neighbor), or hybrid modes. One of the 23 agents registered in `configs/config.json` under `agents.audio_analysis_agent` (enabled by default).

### AuditExplanationAgent
A2A agent (`cogniverse_agents/audit_explanation_agent.py`) that explains why a given answer memory was produced — its derivation chain, per-source trust, and any active contradictions. Registered under `agents.audit_explanation_agent` and enabled by default, unlike the 8 knowledge-layer agents that ship disabled in `configs/config.json`.

---

## B

### Backend
Abstract interface for vector database operations (search, feed, delete). Implemented by `VespaBackend`.

### BackendProfileConfig
Configuration for a specific embedding/search strategy. Defines embedding model, chunk strategy, top_k, etc. Read and written through `ConfigManager`'s backend-profile methods (`get_backend_profile`, `add_backend_profile`, `update_backend_profile`, `list_backend_profiles`), which default to `service="backend"` — the same config service used by both the runtime admin API and the dashboard's backend-profile editor.

### BackendRegistry
Central registry for vector database backends. Enables switching backends without code changes.

---

## C

### ConflictSet
Group of memories that make conflicting claims about the same `metadata.subject_key`. Produced by `ContradictionDetector.detect()`. Each `ConflictSet` carries `subject_key` and the list of `conflicting_memory_ids`. Persisted under sentinel `agent_name="_conflict_store"`.

### ContradictionDetector
Class (`cogniverse_core/memory/contradiction.py`) that groups memories by `metadata.subject_key` and emits a `ConflictSet` for each subject that has more than one distinct content signature. Memories without a `subject_key` pass through unchanged. The same module also exposes a top-level `reconcile(memories, policy)` function that applies the schema's `ContradictionPolicy` (`latest_wins`, `trust_ranked`, `preserve_both`) to resolve a conflict's members.

### ContradictionReconciliationAgent
Read-only A2A agent (`cogniverse_agents/contradiction_reconciliation_agent.py`) that resolves `ConflictSet`s by applying a `KnowledgeSchema`'s `ContradictionPolicy` over the conflicting memories. Registered under `agents.contradiction_reconciliation_agent`, disabled by default.

### CitationTracingAgent
Read-only A2A agent (`cogniverse_agents/citation_tracing_agent.py`) that walks a memory's `Provenance.derived_from` chain back to its primary sources via `ProvenanceWalker`. Registered under `agents.citation_tracing_agent`, disabled by default.

### CodingAgent
A2A agent (`cogniverse_agents/coding_agent.py`) that searches code semantically, plans and generates code changes with DSPy, and runs them inside an OpenShell `SandboxManager` sandbox, looping on execution failures. Registered under `agents.coding_agent`, enabled by default.

### CrossTenantComparisonAgent
Read-only A2A agent (`cogniverse_agents/cross_tenant_comparison_agent.py`) that compares per-tenant views of one subject across all tenants in an org via `FederationService.federated_get_all`, enforcing that the caller is `tenant_admin`/`org_admin` and that every requested tenant shares the caller's org. Registered under `agents.cross_tenant_comparison_agent`, disabled by default.

### CancellationToken
Thread-safe signal for graceful task cancellation. Used by EventQueue to coordinate workflow/ingestion abort at phase boundaries.

### Checkpoint
Saved state of a long-running optimization / auto-eval workflow, persisted so a restarted run resumes from the last completed stage instead of re-running expensive work. Implemented by `PipelineCheckpoint` in `cogniverse_core/durable` — not per-query orchestration.

### ColPali
Multi-vector embedding model for images/documents. Creates patch-level embeddings for fine-grained similarity search.

### ColQwen
Vision-language model that processes video chunks. Generates multi-vector embeddings from video chunks with text understanding.

### ConfigManager
Central API for multi-tenant configuration. Manages system, agent, backend, routing, and telemetry configs.

### ConfigScope
Category of configuration: `SYSTEM`, `AGENT`, `ROUTING`, `TELEMETRY`, `SCHEMA`, `BACKEND`.

---

## D

### DeepResearchAgent
A2A agent (`cogniverse_agents/deep_research_agent.py`) that decomposes a query, iteratively gathers evidence via parallel searches, and synthesizes a cited report. Registered under `agents.deep_research_agent`, enabled by default.

### DetailedReportAgent
A2A agent (`cogniverse_agents/detailed_report_agent.py`) that generates comprehensive reports — executive summary, findings, technical and visual analysis, recommendations — with optional RLM synthesis for large result sets. Registered under `agents.detailed_report_agent`, enabled by default.

### DocumentAgent
A2A agent (`cogniverse_agents/document_agent.py`) offering dual-strategy document search: ColPali visual (page-as-image), ColBERT/BM25 text, or hybrid, with keyword-based auto strategy selection. Registered under `agents.document_agent`, enabled by default.

### DPO (Direct Preference Optimization)
Training technique using preference pairs (chosen/rejected). Cogniverse extracts DPO pairs from annotated traces.

### DSPy
Framework for programming language models with signatures and modules. Enables prompt optimization and composable LLM programs.

### Durable Execution
Checkpoint + resume for long-running optimization / auto-eval workflows in `cogniverse_core/durable`. A restarted run continues from the last completed stage instead of re-running expensive work. See `PipelineCheckpoint` / `PipelineCheckpointStorage`.

---

## E

### Embedding
Vector representation of content for similarity search. Different models produce different dimensions (128, 768, 1024).

### EmbeddingGenerator
Embedding subsystem that generates embeddings and feeds them to backends, handling multi-vector and single-vector strategies. The concrete implementation is `EmbeddingGeneratorImpl`, constructed via `EmbeddingGeneratorFactory` / `create_embedding_generator`.

### EntityExtractionAgent
Tiered A2A NER agent (`cogniverse_agents/entity_extraction_agent.py`) with a fast GLiNER + SpaCy path (no LLM) and a DSPy `ChainOfThought` fallback. Registered under `agents.entity_extraction_agent`, enabled by default.

### EventQueue
A2A-compatible real-time notification system for streaming task progress to multiple subscribers. Supports pub/sub pattern, reconnection with replay, and graceful cancellation. Used by orchestrator and ingestion pipeline.

### EventType
Enum discriminator for event types: `STATUS`, `PROGRESS`, `ARTIFACT`, `ERROR`, `COMPLETE`. Corresponding event classes are `StatusEvent` (state transitions), `ProgressEvent` (incremental progress), `ArtifactEvent` (intermediate results), `ErrorEvent` (errors), `CompleteEvent` (task completion).

### ExperimentMetrics
Typed dataclass (`cogniverse_agents/optimizer/artifact_manager.py`) recording one optimization run. Fields: `tenant_id`, `agent_type`, `run_id`, `timestamp`, `optimizer`, `baseline_score`, `candidate_score`, `improvement`, `promoted`, `train_examples`, `extra_metrics`. Stored as one row in the per-tenant per-agent experiments dataset; queryable via `ArtifactManager.load_experiments()`.

---

## F

### FederatedQueryAgent
Read-only A2A agent (`cogniverse_agents/federated_query_agent.py`) that answers a free-text query by aggregating federated reads across multiple tenants in the same org, with an optional RLM summariser. Registered under `agents.federated_query_agent`, disabled by default.

### FederationService
Service (`cogniverse_core/memory/federation.py`) that merges an org's shared trunk of knowledge with per-tenant overlays. `federated_get_all` returns tenant rows + org-trunk rows, deduplicating by `metadata.subject_key` (tenant overlay wins). Promotion (`promote_to_org_trunk`) copies a tenant memory to `{org}:_org_trunk`. Schema sensitivity (`tenant_private` / `org_shared` / `global_shared`) gates promotion.

### Frame-based Processing
Video processing strategy that extracts individual frames. Used with ColPali for image-level search.

---

## G

### GatewayAgent
LLM-free A2A entry point (`cogniverse_agents/gateway_agent.py`) that classifies queries via GLiNER zero-shot NER (target latency <100ms, no LLM call) and routes simple queries directly to an execution agent, complex ones to `OrchestratorAgent`. Returns a `GatewayOutput` with a categorical `complexity` (`simple`/`complex`), detected `modality`, and a numeric `confidence`. Registered under `agents.gateway_agent`, enabled by default.

### GEPA (Generalized Embedding-based Prompt Adaptation)
DSPy optimizer for adapting prompts based on embeddings. Used for routing and modality optimization.

### GLiNER
Named entity recognition model used for keyword extraction. Alternative to LLM-based entity extraction.

---

## H

### HITL (Human-in-the-Loop)
Workflow pattern involving human approval or feedback. Implemented via `HumanApprovalAgent` and approval queues.

---

## I

### ImageSearchAgent
A2A agent (`cogniverse_agents/image_search_agent.py`) that performs ColPali multi-vector image similarity search against Vespa, with semantic and hybrid (BM25+ColPali) modes and image-to-image lookup. Registered under `agents.image_search_agent`, enabled by default.

---

## K

### KnowledgeGraphTraversalAgent
A2A agent (`cogniverse_agents/kg_traversal_agent.py`) that structurally walks `kg_node`/`entity_fact` and `kg_edge` memories from a seed entity into a node+edge graph view. Registered under `agents.kg_traversal_agent`, disabled by default.

### KnowledgeRegistry
In-memory registry (`cogniverse_core/memory/schema.py`) mapping memory kind strings to `KnowledgeSchema` objects. Built via `build_default_registry()` which seeds `conversation_turn`, `learned_strategy`, `tenant_instruction`, `external_doc`, `entity_fact`, `kg_node`, `kg_edge`. Unregistered kinds fall back to `permanent + tenant_private + provenance_required=True`.

### KnowledgeSummarizationAgent
A2A agent (`cogniverse_agents/knowledge_summarization_agent.py`) that distills a knowledge subgraph into a structured, citation-aware summary, with optional admin-gated promotion to the org trunk. Registered under `agents.knowledge_summarization_agent`, disabled by default.

### KnowledgeSchema
Dataclass (`cogniverse_core/memory/schema.py`) describing one kind of memory. Fields: `kind`, `retention` (PERMANENT / EPHEMERAL_SESSION / EPHEMERAL_DAYS / SCHEMA_DRIVEN), `sensitivity` (tenant_private / org_shared / global_shared), `pinnable_by`, `provenance_required`, `contradiction_policy`, `default_trust`. Gating: `validate_write` raises `SchemaViolationError` when provenance rules or pin authority are violated.

---

## L

### LifecycleScheduler
Background scheduler (`cogniverse_core/memory/lifecycle_scheduler.py`) that runs schema-driven cleanup on each warm tenant. Started in the FastAPI lifespan. Each tick looks up `metadata["kind"]` in the `KnowledgeRegistry` and applies the retention policy (EPHEMERAL_DAYS soft-delete at N days, hard-delete at 2N; SCHEMA_DRIVEN calls the schema's `cleanup_hook`). Pinned memory ids are excluded before any deletion. Configurable via `COGNIVERSE_MEMORY_LIFECYCLE_INTERVAL` (default 3600 s).

### LLM Judge
Evaluator that uses LLMs to score outputs. Supports reference-free, reference-based, and hybrid modes.

---

## M

### MCP (Model Context Protocol)
Protocol for connecting LLMs to external tools and data. Enables agents to access databases, APIs, etc.

### Mixin
Composable class that adds specific functionality to agents. Examples: `MemoryAwareMixin`, `TenantAwareAgentMixin`, `HealthCheckMixin`, `DynamicDSPyMixin`.

### Modality
Type of content defined by `ContentType` enum: `video`, `audio`, `image`, `text`, `dataframe`, `document`. Used for routing queries to appropriate agents.

### MRR (Mean Reciprocal Rank)
Evaluation metric: average of 1/rank for first relevant result. Higher is better (max 1.0).

### Multi-Agent Orchestration
Coordinating multiple agents to complete complex tasks. Managed by `OrchestratorAgent`, which is invoked by `AgentDispatcher` when `GatewayAgent` classifies a query as complex.

### MultiDocumentSynthesisAgent
A2A agent (`cogniverse_agents/multi_document_synthesis_agent.py`) that synthesises a coherent answer across N source documents while preserving the citation graph. Registered under `agents.multi_document_synthesis_agent`, disabled by default.

### Multi-Vector
Embedding strategy that produces multiple vectors per document. Used by ColPali for patch-level embeddings.

---

## N

### NDCG (Normalized Discounted Cumulative Gain)
Evaluation metric that considers position and relevance. NDCG@K evaluates top-K results.

---

## O

### OpenTelemetry
Standard for distributed tracing and metrics. Cogniverse uses OpenTelemetry for all telemetry.

### Optimizer
DSPy component that improves module performance via training. Examples: GEPA, MIPROv2.

### OrchestratorAgent
A2A agent (`cogniverse_agents/orchestrator_agent.py`) with its own DSPy planning via `OrchestrationModule`. Plans and executes multi-agent workflows independently using `AgentRegistry` for agent discovery and direct A2A HTTP calls. Invoked by `AgentDispatcher` when `GatewayAgent` classifies a query as complex. Emits `cogniverse.orchestration` telemetry spans consumed by the dashboard's Orchestration tab.

---

## P

### Phoenix
Arize's observability platform for AI applications. Stores traces, spans, experiments, and enables visualization.

### PinService
Service (`cogniverse_core/memory/pinning.py`) that pins memories so they survive lifecycle cleanup, trust decay, and curator passes. Per-role quota enforced via `PinQuotas` (defaults: user 50, tenant_admin 500, org_admin unlimited). Pin records live under sentinel `agent_name="_pinning"`. Hierarchy: org_admin can unpin any pin; tenant_admin can unpin user/tenant pins; user can only unpin their own.

### Pipeline
Sequence of processing steps for video ingestion. Includes segmentation, transcription, description, embedding.

### Provenance
Record attached to every memory write describing its origin. Fields: `written_by`, `written_at` (ISO-8601 UTC), `derivation_kind` (direct_ingest / user_assert / extraction / summarization / synthesis / agent_inference), `confidence`, `derived_from` (list of `CitationRef`), `trace_id`. Stored under `metadata["provenance"]`. `ProvenanceWalker` traverses citation chains back to primary sources.

### Precision@K
Evaluation metric: fraction of top-K results that are relevant.

### Processor
Pluggable component in ingestion pipeline. Examples: `KeyframeProcessor`, `AudioTranscriber`, `VLMDescriptor`.

### Profile
Named configuration for backend, embedding model, and search strategy. Examples: `video_colpali_mv_frame`.

### ProfileSelectionAgent
A2A agent (`cogniverse_agents/profile_selection_agent.py`) that uses DSPy LLM reasoning to pick the optimal backend search `Profile` for a query, with a heuristic fallback when the LLM is unavailable. Registered under `agents.profile_selection_agent`, enabled by default.

---

## Q

### QueryEnhancementAgent
A2A agent (`cogniverse_agents/query_enhancement_agent.py`) that expands and rewrites queries with synonyms, context, and RRF variants using DSPy. Registered under `agents.query_enhancement_agent`, enabled by default.

### QueueManager
Lifecycle manager for EventQueues. Creates, retrieves, closes, and cleans up expired queues. Supports multi-tenant isolation via `tenant_id`.

---

## R

### RAGAS
Evaluation framework for RAG systems. Cogniverse implements RAGAS-like metrics.

### Recall@K
Evaluation metric: fraction of relevant items found in top-K.

### Reranker
Component that reorders search results for better relevance. Types: learned, hybrid, multi-modal.

### RRF (Reciprocal Rank Fusion)
Algorithm for combining multiple ranked lists. Used in hybrid search to merge semantic and BM25 results.

### RLMOptions
Pydantic model (`cogniverse_core/agents/rlm_options.py`) for per-query RLM configuration. Key fields: `enabled` (explicit), `auto_detect` + `context_threshold` (size-based), `max_iterations`, `include_trajectory`, `trajectory_max_entries`. Passed as `rlm` field on eligible agent inputs.

### RLMResult
Dataclass from `RLMInference.process()`. Fields: `answer`, `depth_reached`, `total_calls`, `tokens_used` (total tokens across all LLM sub-calls), `latency_ms`, `was_fallback` (True when max_iterations exhausted without SUBMIT()), `trajectory` (populated when `include_trajectory=True`), `metadata` (always includes `trajectory_summary` and `trajectory_length`).

### Routing
Directing queries to appropriate agents based on modality and content. Entry point is `GatewayAgent` (GLiNER classification, <100ms). Simple queries route directly to a single execution agent; complex queries hand off to `OrchestratorAgent` for multi-step A2A execution.

### RoutingTier
Conceptual classification of query-routing decisions within `GatewayAgent`: GLiNER entity detection (<100ms, primary path), a deterministic keyword-based fallback when GLiNER yields no confident entity, and complex-query handoff to `OrchestratorAgent`'s DSPy planning. Not a named enum in the codebase; routing outcomes are expressed as the categorical `complexity` (`simple`/`complex`) and numeric `confidence` fields on `GatewayOutput`.

---

## S

### SandboxManager
Component (`cogniverse_runtime/sandbox_manager.py`) that wraps the OpenShell SDK to create and manage per-agent execution sandboxes. Policy controlled by `SandboxPolicy` enum. Used by `CodingAgent` (code execution) and `OrchestratorAgent` (egress-restricted A2A sub-agent calls). Emits OpenTelemetry spans for every lifecycle event.

### SandboxPolicy
Enum in `cogniverse_runtime/sandbox_manager.py` with three values: `REQUIRED` (refuse to start without gateway), `OPTIONAL` (warn and continue, default for dev), `DISABLED` (skip entirely). Resolved from `COGNIVERSE_SANDBOX_POLICY` env var → config → default `optional`.

### Schema
Vespa document schema defining fields and indexing. Tenant-specific schemas use `_tenant_id` suffix.

### SearchAgent
A2A agent (`cogniverse_agents/search_agent.py`) that performs multi-modal retrieval against Vespa across video/image/text/audio/document modalities, with DSPy query rewriting and RRF ensemble fusion. Registered under `agents.search_agent`, enabled by default.

### SignatureVariant
Named variant of an agent's DSPy input/output signature (`cogniverse_agents/optimizer/signature_variants.py`). Registered in `SignatureVariantRegistry`; tenant selects via `TenantConfig.metadata["signature_variants"][agent_type]`. Artifact manager keys per `(tenant, agent, variant)`. Default variant id is `"default"` (`DEFAULT_VARIANT_ID`).

### Session
Multi-turn conversation context. Tracked via `session_id` in telemetry for conversation-aware evaluation.

### SFT (Supervised Fine-Tuning)
Training technique using instruction-response pairs. Cogniverse extracts SFT data from traces.

### Single-Vector
Embedding strategy that produces one vector per document. Used by VideoPrism for video-level embeddings.

### Span
OpenTelemetry trace unit representing an operation. Has attributes, events, and parent/child relationships.

### Strategy
Pluggable algorithm for processing steps. Types: `FrameSegmentationStrategy`, `ChunkSegmentationStrategy`, etc.

### SummarizerAgent
A2A agent (`cogniverse_agents/summarizer_agent.py`) that turns search results into structured summaries with a thinking phase and VLM visual analysis. Registered under `agents.summarizer_agent`, enabled by default.

---

## T

### TaskEvent
Union type for all A2A-compatible events (StatusEvent, ProgressEvent, ArtifactEvent, ErrorEvent, CompleteEvent). All events extend BaseEvent which contains `event_id`, `task_id`, `tenant_id`, `timestamp`, and type-specific data.

### TaskState
A2A-compatible workflow state: `pending`, `working`, `input-required`, `completed`, `failed`, `cancelled`.

### Telemetry
Observability data: traces, spans, metrics. Managed by `TelemetryManager` with tenant isolation.

### TemporalReasoningAgent
Read-only A2A agent (`cogniverse_agents/temporal_reasoning_agent.py`) that compares a subject's knowledge across explicit time windows using `Provenance.written_at`. Registered under `agents.temporal_reasoning_agent`, disabled by default.

### TextAnalysisAgent
Runtime-configurable A2A agent (`cogniverse_agents/text_analysis_agent.py`) that performs DSPy text analysis (sentiment/summary/entities) with per-tenant persisted config. Registered under `agents.text_analysis_agent`, enabled by default.

### TrustRecord
Struct derived at memory-write time from `KnowledgeSchema.default_trust` and `DerivationKind`. Carries initial score, decay floor, and endorsement history. Trust score is on a 0.0–1.0 scale and decays 0.005/day (`_DAILY_DECAY`) above its schema default, never below it. At retrieval, `rank_with_trust` composes relevance × trust × confidence. See `cogniverse_core/memory/trust.py`.

### Tenant
Isolated organization in multi-tenant system. All data and config is tenant-scoped via `tenant_id`.

### Trace
End-to-end record of a request through the system. Contains multiple spans in a tree structure.

### Trajectory
Multi-turn conversation sequence for fine-tuning. Extracted from session-aware traces.

---

## V

### Vespa
Vector database used for content storage and search. Supports multi-vector, hybrid search, tenant isolation.

### VideoPrism
Google's video embedding model. Produces global embeddings from video chunks/segments.

### VLM (Vision Language Model)
Model that understands both images and text. Used for generating frame descriptions.

---

## W

### Workflow
Multi-step agent execution plan. Managed by `OrchestratorAgent`.

### WorkflowIntelligence
System for learning and adapting workflows based on performance. Optimizes agent selection and parameters.
