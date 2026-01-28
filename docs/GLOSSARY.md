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
Pydantic model that all agent outputs must extend. Includes optional `error` field for error reporting.

### AgentRegistry
Central registry for discovering and accessing agents. Agents register themselves with capabilities for routing.

---

## B

### Backend
Abstract interface for vector database operations (search, feed, delete). Implemented by `VespaBackend`, `PineconeBackend`, etc.

### BackendProfile
Configuration for a specific embedding/search strategy. Defines embedding model, chunk strategy, top_k, etc.

### BackendRegistry
Central registry for vector database backends. Enables switching backends without code changes.

---

## C

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
Vision-language model that processes video chunks. Generates embeddings from video frames with text understanding.

### ConfigManager
Central API for multi-tenant configuration. Manages system, agent, backend, routing, and telemetry configs.

### ConfigScope
Category of configuration: `SYSTEM`, `AGENT`, `ROUTING`, `TELEMETRY`, `BACKEND`.

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
Vector representation of content for similarity search. Different models produce different dimensions (768, 1024, 1152).

### EmbeddingGenerator
Component that generates embeddings and feeds them to backends. Handles multi-vector and single-vector strategies.

### EventQueue
A2A-compatible real-time notification system for streaming task progress to multiple subscribers. Supports pub/sub pattern, reconnection with replay, and graceful cancellation. Used by orchestrator and ingestion pipeline.

### EventType
Standard A2A event categories: `StatusEvent` (state transitions), `ProgressEvent` (incremental progress), `ArtifactEvent` (intermediate results), `ErrorEvent` (errors), `CompleteEvent` (task completion).

---

## F

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

## L

### LLM Judge
Evaluator that uses LLMs to score outputs. Supports reference-free, reference-based, and hybrid modes.

---

## M

### MCP (Model Context Protocol)
Protocol for connecting LLMs to external tools and data. Enables agents to access databases, APIs, etc.

### Mixin
Composable class that adds specific functionality to agents. Examples: `TelemetryMixin`, `MemoryMixin`, `CheckpointingMixin`.

### Modality
Type of content: `video`, `image`, `pdf`, `audio`, `document`. Used for routing queries to appropriate agents.

### MRR (Mean Reciprocal Rank)
Evaluation metric: average of 1/rank for first relevant result. Higher is better (max 1.0).

### Multi-Agent Orchestration
Coordinating multiple agents to complete complex tasks. Managed by `MultiAgentOrchestrator`.

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

---

## P

### Phoenix
Arize's observability platform for AI applications. Stores traces, spans, experiments, and enables visualization.

### Pipeline
Sequence of processing steps for video ingestion. Includes segmentation, transcription, description, embedding.

### Precision@K
Evaluation metric: fraction of top-K results that are relevant.

### Processor
Pluggable component in ingestion pipeline. Examples: `KeyframeExtractor`, `AudioTranscriber`, `VLMDescriptor`.

### Profile
Named configuration for backend, embedding model, and search strategy. Examples: `video_colpali_mv_frame`.

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

### Routing
Directing queries to appropriate agents based on modality and content. Implemented by `RoutingAgent`.

---

## S

### Schema
Vespa document schema defining fields and indexing. Tenant-specific schemas use `_tenant_id` suffix.

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

## Q

### QueueManager
Lifecycle manager for EventQueues. Creates, retrieves, closes, and cleans up expired queues. Supports multi-tenant isolation via `tenant_id`.

---

## T

### TaskEvent
Base type for all A2A-compatible events. Contains `event_id`, `task_id`, `tenant_id`, `timestamp`, and type-specific data.

### TaskState
A2A-compatible workflow state: `pending`, `working`, `input-required`, `completed`, `failed`, `cancelled`.

### Telemetry
Observability data: traces, spans, metrics. Managed by `TelemetryManager` with tenant isolation.

### Tenant
Isolated organization in multi-tenant system. All data and config is tenant-scoped via `tenant_id`.

### Tier
Routing strategy category: `fast`, `accurate`, `comprehensive`. Balances latency vs quality.

### Trace
End-to-end record of a request through the system. Contains multiple spans in a tree structure.

### Trajectory
Multi-turn conversation sequence for fine-tuning. Extracted from session-aware traces.

---

## V

### Vespa
Vector database used for content storage and search. Supports multi-vector, hybrid search, tenant isolation.

### VideoPrism
Google's video embedding model. Produces global embeddings from video frames.

### VLM (Vision Language Model)
Model that understands both images and text. Used for generating frame descriptions.

---

## W

### Workflow
Multi-step agent execution plan. Managed by `MultiAgentOrchestrator` with checkpointing support.

### WorkflowIntelligence
System for learning and adapting workflows based on performance. Optimizes agent selection and parameters.
