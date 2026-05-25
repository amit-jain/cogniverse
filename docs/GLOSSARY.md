# Cogniverse Glossary

A reference guide to terms and concepts used throughout the Cogniverse codebase.

---

## A

### A2A (Agent-to-Agent)
Google's protocol for inter-agent communication. Defines a standard message format for agents to communicate, enabling agents from different systems to work together. Cogniverse implements A2A through the `A2AAgent` base class.

### AgentBase
The abstract generic base class for all agents: `AgentBase[InputT, OutputT, DepsT]`. Provides type safety for inputs, outputs, and dependencies.

### AgentDeps
Pydantic model for agent dependencies (LLMs, backends, other agents). Injected at runtime to enable testing and configuration.

### AgentInput
Pydantic model that all agent inputs must extend. Provides validation and serialization for agent requests.

### AgentOutput
Pydantic model that all agent outputs must extend. Provides validation and serialization for agent responses.

### AgentRegistry
Central registry for discovering and accessing agents. Agents register themselves with capabilities for routing.

---

## B

### Backend
Abstract interface for vector database operations (search, feed, delete). Implemented by `VespaBackend`.

### BackendProfileConfig
Configuration for a specific embedding/search strategy. Defines embedding model, chunk strategy, top_k, etc.

### BackendRegistry
Central registry for vector database backends. Enables switching backends without code changes.

---

## C

### ConflictSet
Group of memories that make conflicting claims about the same `metadata.subject_key`. Produced by `ContradictionDetector.detect()`. Each `ConflictSet` carries `subject_key` and the list of `conflicting_memory_ids`. Persisted under sentinel `agent_name="_conflict_store"`.

### ContradictionDetector
Class (`cogniverse_core/memory/contradiction.py`) that groups memories by `metadata.subject_key` and emits a `ConflictSet` for each subject that has more than one distinct content signature. Memories without a `subject_key` pass through unchanged. `reconcile(memories, policy)` applies the schema's `ContradictionPolicy` (`latest_wins`, `trust_ranked`, `preserve_both`).

### CancellationToken
Thread-safe signal for graceful task cancellation. Used by EventQueue to coordinate workflow/ingestion abort at phase boundaries.

### Checkpoint
Saved state during workflow execution for durability. Allows resuming workflows after failures without re-executing completed tasks.

### CheckpointLevel
Granularity of checkpointing: `PHASE`, `TASK`, or `PHASE_AND_TASK`.

### CheckpointStatus
State of a checkpoint: `ACTIVE`, `SUPERSEDED`, `FAILED`, `COMPLETED`.

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

### DPO (Direct Preference Optimization)
Training technique using preference pairs (chosen/rejected). Cogniverse extracts DPO pairs from annotated traces.

### DSPy
Framework for programming language models with signatures and modules. Enables prompt optimization and composable LLM programs.

### Durable Execution
Workflow execution that survives failures via checkpointing. Completed tasks are replayed from cache on resume.

---

## E

### Embedding
Vector representation of content for similarity search. Different models produce different dimensions (128, 768, 1024).

### EmbeddingGenerator
Embedding subsystem that generates embeddings and feeds them to backends, handling multi-vector and single-vector strategies. The concrete implementation is `EmbeddingGeneratorImpl`, constructed via `EmbeddingGeneratorFactory` / `create_embedding_generator`.

### EventQueue
A2A-compatible real-time notification system for streaming task progress to multiple subscribers. Supports pub/sub pattern, reconnection with replay, and graceful cancellation. Used by orchestrator and ingestion pipeline.

### EventType
Enum discriminator for event types: `STATUS`, `PROGRESS`, `ARTIFACT`, `ERROR`, `COMPLETE`. Corresponding event classes are `StatusEvent` (state transitions), `ProgressEvent` (incremental progress), `ArtifactEvent` (intermediate results), `ErrorEvent` (errors), `CompleteEvent` (task completion).

### ExperimentMetrics
Typed dataclass (`cogniverse_agents/optimizer/artifact_manager.py`) recording one optimization run. Fields: `tenant_id`, `agent_type`, `run_id`, `timestamp`, `optimizer`, `baseline_score`, `candidate_score`, `improvement`, `promoted`, `train_examples`, `extra_metrics`. Stored as one row in the per-tenant per-agent experiments dataset; queryable via `ArtifactManager.load_experiments()`.

---

## F

### FederationService
Service (`cogniverse_core/memory/federation.py`) that merges an org's shared trunk of knowledge with per-tenant overlays. `federated_get_all` returns tenant rows + org-trunk rows, deduplicating by `metadata.subject_key` (tenant overlay wins). Promotion (`promote_to_org_trunk`) copies a tenant memory to `{org}:_org_trunk`. Schema sensitivity (`tenant_private` / `org_shared` / `global_shared`) gates promotion.

### Frame-based Processing
Video processing strategy that extracts individual frames. Used with ColPali for image-level search.

---

## G

### GEPA (Generalized Embedding-based Prompt Adaptation)
DSPy optimizer for adapting prompts based on embeddings. Used for routing and modality optimization.

### GLiNER
Named entity recognition model used for keyword extraction. Alternative to LLM-based entity extraction.

---

## H

### HITL (Human-in-the-Loop)
Workflow pattern involving human approval or feedback. Implemented via `HumanApprovalAgent` and approval queues.

---

## K

### KnowledgeRegistry
In-memory registry (`cogniverse_core/memory/schema.py`) mapping memory kind strings to `KnowledgeSchema` objects. Built via `build_default_registry()` which seeds `conversation_turn`, `learned_strategy`, `tenant_instruction`, `external_doc`, `entity_fact`, `kg_node`, `kg_edge`. Unregistered kinds fall back to `permanent + tenant_private + provenance_required=True`.

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
Record attached to every memory write describing its origin. Fields: `written_by`, `derivation_kind` (direct_ingest / user_assert / extraction / summarization / synthesis / agent_inference), `confidence`, `derived_from` (list of `CitationRef`), `trace_id`. Stored under `metadata["provenance"]`. `ProvenanceWalker` traverses citation chains back to primary sources.

### Precision@K
Evaluation metric: fraction of top-K results that are relevant.

### Processor
Pluggable component in ingestion pipeline. Examples: `KeyframeProcessor`, `AudioTranscriber`, `VLMDescriptor`.

### Profile
Named configuration for backend, embedding model, and search strategy. Examples: `video_colpali_mv_frame`.

---

## Q

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

---

## S

### SandboxManager
Component (`cogniverse_runtime/sandbox_manager.py`) that wraps the OpenShell SDK to create and manage per-agent execution sandboxes. Policy controlled by `SandboxPolicy` enum. Used by `CodingAgent` (code execution) and `OrchestratorAgent` (egress-restricted A2A sub-agent calls). Emits OpenTelemetry spans for every lifecycle event.

### SandboxPolicy
Enum in `cogniverse_runtime/sandbox_manager.py` with three values: `REQUIRED` (refuse to start without gateway), `OPTIONAL` (warn and continue, default for dev), `DISABLED` (skip entirely). Resolved from `COGNIVERSE_SANDBOX_POLICY` env var → config → default `optional`.

### Schema
Vespa document schema defining fields and indexing. Tenant-specific schemas use `_tenant_id` suffix.

### SignatureVariant
Named variant of an agent's DSPy input/output signature (`cogniverse_agents/optimizer/signature_variants.py`). Registered in `SignatureVariantRegistry`; tenant selects via `TenantConfig.metadata["signature_variants"][agent_type]`. Artifact manager keys per `(tenant, agent, variant)`. Default variant uses the bare agent_type key for back-compat.

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

---

## T

### TaskEvent
Union type for all A2A-compatible events (StatusEvent, ProgressEvent, ArtifactEvent, ErrorEvent, CompleteEvent). All events extend BaseEvent which contains `event_id`, `task_id`, `tenant_id`, `timestamp`, and type-specific data.

### TaskState
A2A-compatible workflow state: `pending`, `working`, `input-required`, `completed`, `failed`, `cancelled`.

### Telemetry
Observability data: traces, spans, metrics. Managed by `TelemetryManager` with tenant isolation.

### TrustRecord
Struct derived at memory-write time from `KnowledgeSchema.default_trust` and `DerivationKind`. Carries initial score, decay floor, and endorsement history. Trust ages at ≈0.5 pt/day above baseline. At retrieval, `rank_with_trust` composes relevance × trust × confidence. See `cogniverse_core/memory/trust.py`.

### Tenant
Isolated organization in multi-tenant system. All data and config is tenant-scoped via `tenant_id`.

### RoutingTier
Routing tier levels within `GatewayAgent`: `FAST_PATH` (GLiNER for common patterns, <100ms), `SLOW_PATH` (LLM + DSPy for complex queries), `LANGEXTRACT` (structured extraction), `FALLBACK` (keyword-based). Balances latency vs quality.

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
Multi-step agent execution plan. Managed by `OrchestratorAgent` with checkpointing support.

### WorkflowIntelligence
System for learning and adapting workflows based on performance. Optimizes agent selection and parameters.
