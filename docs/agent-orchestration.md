# Agent Orchestration & A2A Protocol

## Overview

Cogniverse uses a **multi-agent orchestration system** where specialized agents coordinate through the **Agent-to-Agent (A2A) communication protocol**. Agents can work sequentially, in parallel, or hierarchically to handle complex queries.

## Multi-Agent Coordination

```
┌─────────────────────────────────────────────────────────────┐
│                    Composing Agent                          │
│              (Central Orchestrator)                         │
│  ┌────────────────────────────────────────────────┐        │
│  │ Receives user query                            │        │
│  │ Analyzes requirements                          │        │
│  │ Coordinates specialist agents                  │        │
│  │ Aggregates results                             │        │
│  └────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
                           ↓
        ┌──────────────────┴──────────────────┐
        ↓                  ↓                   ↓
┌───────────────┐  ┌──────────────┐  ┌──────────────────┐
│ Routing       │  │ Video Search │  │ Summarizer/      │
│ Agent         │  │ Agent        │  │ Report Agent     │
│               │  │              │  │                  │
│ Routes queries│  │ Executes     │  │ Generates        │
│ to specialist │  │ video search │  │ summaries/       │
│ agents        │  │              │  │ reports          │
└───────────────┘  └──────────────┘  └──────────────────┘
```

## Agent Types

### 1. Routing Agent

**Purpose**: Analyze queries and route to appropriate specialist agents

**Capabilities**:
- Query intent classification
- Complexity assessment
- Entity extraction
- DSPy-optimized routing decisions

**Example Usage**:
```python
from src.app.agents.routing_agent import EnhancedRoutingAgent

routing_agent = EnhancedRoutingAgent.from_config()

routing_result = routing_agent.route_query(
    query="Find videos about machine learning from last week"
)

# Result:
# {
#   "recommended_agent": "video_search_agent",
#   "confidence": 0.92,
#   "reasoning": "Video search needed for visual ML content",
#   "workflow": "direct_search"
# }
```

### 2. Video Search Agent

**Purpose**: Execute video searches using ColPali/VideoPrism models

**Capabilities**:
- Semantic video search
- Temporal filtering
- Multi-modal queries
- Ranking strategy selection

**Example Usage**:
```python
from src.app.agents.enhanced_video_search_agent import EnhancedVideoSearchAgent

video_agent = EnhancedVideoSearchAgent.from_config()

search_result = video_agent.search(
    query="machine learning tutorial",
    temporal_filter={"start": "2025-09-26", "end": "2025-10-03"},
    ranking_strategy="hybrid_float_bm25",
    top_k=10
)
```

### 3. Summarizer Agent

**Purpose**: Generate summaries from search results or documents

**Capabilities**:
- Multi-document summarization
- Extractive + abstractive
- Customizable length/style

**Example Usage**:
```python
from src.app.agents.summarizer_agent import SummarizerAgent

summarizer = SummarizerAgent.from_config()

summary = summarizer.summarize(
    documents=search_results,
    max_length=200,
    style="concise"
)
```

### 4. Report Agent

**Purpose**: Create detailed reports combining multiple data sources

**Capabilities**:
- Multi-source aggregation
- Structured output formatting
- Citation tracking

## A2A Protocol Specification

### Message Format

```python
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class A2AMessage:
    """Agent-to-Agent message format"""

    # Message metadata
    message_id: str              # Unique message ID
    sender_agent: str            # Source agent name
    recipient_agent: str         # Target agent name
    timestamp: datetime          # Creation timestamp

    # Message content
    message_type: str            # "task", "result", "error", "status"
    data: Dict[str, Any]        # Payload data
    context: Optional[Dict]      # Optional context

    # Workflow tracking
    workflow_id: str             # Workflow session ID
    parent_message_id: Optional[str]  # For threading
```

### Message Types

**1. Task Message** - Request agent to perform work

```python
task_message = A2AMessage(
    message_id="msg_001",
    sender_agent="routing_agent",
    recipient_agent="video_search_agent",
    timestamp=datetime.now(),
    message_type="task",
    data={
        "action": "search",
        "query": "machine learning",
        "parameters": {
            "temporal_filter": {"start": "2025-09-26", "end": "2025-10-03"},
            "ranking_strategy": "hybrid_float_bm25",
            "top_k": 10
        }
    },
    workflow_id="wf_12345"
)
```

**2. Result Message** - Return task results

```python
result_message = A2AMessage(
    message_id="msg_002",
    sender_agent="video_search_agent",
    recipient_agent="routing_agent",
    timestamp=datetime.now(),
    message_type="result",
    data={
        "status": "success",
        "results": [...],
        "metadata": {
            "total_results": 10,
            "processing_time_ms": 120
        }
    },
    workflow_id="wf_12345",
    parent_message_id="msg_001"
)
```

**3. Error Message** - Report failures

```python
error_message = A2AMessage(
    message_id="msg_003",
    sender_agent="video_search_agent",
    recipient_agent="routing_agent",
    timestamp=datetime.now(),
    message_type="error",
    data={
        "error_type": "VespaConnectionError",
        "error_message": "Failed to connect to Vespa",
        "recoverable": True,
        "retry_strategy": "exponential_backoff"
    },
    workflow_id="wf_12345",
    parent_message_id="msg_001"
)
```

## Orchestration Patterns

### 1. Sequential Orchestration

One agent after another:

```python
from src.app.orchestration.patterns import SequentialOrchestrator

orchestrator = SequentialOrchestrator()

# Define workflow
workflow = orchestrator.create_workflow([
    {"agent": "routing_agent", "action": "analyze"},
    {"agent": "video_search_agent", "action": "search"},
    {"agent": "summarizer_agent", "action": "summarize"}
])

# Execute
result = await orchestrator.execute(
    workflow=workflow,
    initial_input={"query": "machine learning tutorial"}
)
```

**Flow**:
```
User Query
    ↓
Routing Agent (analyze) → routing decision
    ↓
Video Search Agent (search) → search results
    ↓
Summarizer Agent (summarize) → summary
    ↓
Final Result
```

### 2. Parallel Orchestration

Multiple agents simultaneously:

```python
from src.app.orchestration.patterns import ParallelOrchestrator

orchestrator = ParallelOrchestrator()

# Define parallel tasks
tasks = [
    {"agent": "video_search_agent", "action": "search", "query": "ML basics"},
    {"agent": "video_search_agent", "action": "search", "query": "advanced ML"},
]

# Execute in parallel
results = await orchestrator.execute_parallel(tasks)

# Aggregate results
aggregated = orchestrator.aggregate(results, strategy="merge_and_rank")
```

**Flow**:
```
                User Query
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
Video Search (ML basics)   Video Search (advanced ML)
        ↓                       ↓
        └───────────┬───────────┘
                    ↓
            Aggregate Results
                    ↓
             Final Result
```

### 3. Hierarchical Orchestration

Supervisor delegates to workers:

```python
from src.app.orchestration.patterns import HierarchicalOrchestrator

orchestrator = HierarchicalOrchestrator()

# Define hierarchy
hierarchy = {
    "supervisor": "composing_agent",
    "workers": [
        {"agent": "routing_agent", "role": "router"},
        {"agent": "video_search_agent", "role": "searcher"},
        {"agent": "summarizer_agent", "role": "summarizer"}
    ]
}

# Execute
result = await orchestrator.execute(
    hierarchy=hierarchy,
    query="Find and summarize ML videos from last week"
)
```

**Flow**:
```
         Composing Agent (Supervisor)
                    ↓
        ┌───────────┴──────────────┐
        ↓           ↓              ↓
  Routing     Video Search    Summarizer
    Agent         Agent          Agent
        ↓           ↓              ↓
        └───────────┬──────────────┘
                    ↓
         Composing Agent (aggregates)
                    ↓
             Final Result
```

## Workflow Examples

### Example 1: Video Search → Summarization

```python
async def search_and_summarize_workflow(query: str):
    """Search videos and generate summary"""

    # Step 1: Routing decision
    routing_msg = A2AMessage(
        message_type="task",
        sender_agent="orchestrator",
        recipient_agent="routing_agent",
        data={"action": "analyze", "query": query}
    )
    routing_result = await send_message(routing_msg)

    # Step 2: Video search
    if routing_result.data["recommended_agent"] == "video_search_agent":
        search_msg = A2AMessage(
            message_type="task",
            sender_agent="routing_agent",
            recipient_agent="video_search_agent",
            data={
                "action": "search",
                "query": query,
                "ranking_strategy": "hybrid_float_bm25"
            },
            parent_message_id=routing_msg.message_id
        )
        search_result = await send_message(search_msg)

    # Step 3: Summarization
    summary_msg = A2AMessage(
        message_type="task",
        sender_agent="video_search_agent",
        recipient_agent="summarizer_agent",
        data={
            "action": "summarize",
            "documents": search_result.data["results"]
        },
        parent_message_id=search_msg.message_id
    )
    summary_result = await send_message(summary_msg)

    return summary_result.data["summary"]
```

### Example 2: Multi-Modal Fusion

```python
async def multimodal_fusion_workflow(query: str):
    """Parallel search across modalities, then fuse"""

    # Parallel search
    tasks = [
        {
            "agent": "video_search_agent",
            "action": "visual_search",
            "query": query
        },
        {
            "agent": "video_search_agent",
            "action": "audio_search",
            "query": query
        },
        {
            "agent": "text_search_agent",
            "action": "transcript_search",
            "query": query
        }
    ]

    # Execute in parallel
    results = await execute_parallel(tasks)

    # Fusion
    fusion_msg = A2AMessage(
        message_type="task",
        sender_agent="orchestrator",
        recipient_agent="fusion_agent",
        data={
            "action": "fuse",
            "visual_results": results[0],
            "audio_results": results[1],
            "text_results": results[2],
            "fusion_strategy": "late_fusion"
        }
    )

    fused_result = await send_message(fusion_msg)
    return fused_result.data["fused_results"]
```

## Error Handling

### Retry Logic

```python
async def send_with_retry(
    message: A2AMessage,
    max_retries: int = 3,
    backoff: str = "exponential"
):
    """Send message with retry logic"""

    for attempt in range(max_retries):
        try:
            result = await send_message(message)

            if result.message_type == "error":
                if not result.data.get("recoverable"):
                    raise UnrecoverableError(result.data["error_message"])

                # Retry with backoff
                await asyncio.sleep(2 ** attempt)
                continue

            return result

        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
```

### Timeout Management

```python
async def send_with_timeout(
    message: A2AMessage,
    timeout_seconds: int = 30
):
    """Send message with timeout"""

    try:
        result = await asyncio.wait_for(
            send_message(message),
            timeout=timeout_seconds
        )
        return result

    except asyncio.TimeoutError:
        # Return timeout error message
        return A2AMessage(
            message_type="error",
            sender_agent=message.recipient_agent,
            recipient_agent=message.sender_agent,
            data={
                "error_type": "TimeoutError",
                "error_message": f"Agent {message.recipient_agent} timeout after {timeout_seconds}s",
                "recoverable": True
            }
        )
```

## Monitoring & Debugging

### Workflow Tracing

All agent interactions are traced in Phoenix:

```python
from src.app.telemetry.manager import TelemetryManager

telemetry = TelemetryManager()

# Create workflow span
with telemetry.span("workflow.search_and_summarize") as workflow_span:
    workflow_span.set_attribute("workflow.id", workflow_id)

    # Routing span
    with telemetry.span("agent.routing") as routing_span:
        routing_result = await routing_agent.analyze(query)
        routing_span.set_attribute("routing.decision", routing_result.recommended_agent)

    # Search span
    with telemetry.span("agent.video_search") as search_span:
        search_result = await video_agent.search(query)
        search_span.set_attribute("search.results_count", len(search_result))

    # Summarization span
    with telemetry.span("agent.summarizer") as summary_span:
        summary = await summarizer.summarize(search_result)
```

### Agent Performance Metrics

Track agent performance:

```python
from src.app.orchestration.metrics import AgentMetrics

metrics = AgentMetrics()

# Record agent execution
metrics.record_execution(
    agent_name="video_search_agent",
    duration_ms=120,
    success=True,
    result_quality=0.85
)

# Get agent stats
stats = metrics.get_agent_stats("video_search_agent")
# {
#   "total_executions": 1523,
#   "avg_duration_ms": 115,
#   "success_rate": 0.98,
#   "avg_quality": 0.83
# }
```

## Best Practices

### Agent Design

1. **Single Responsibility**: Each agent should do one thing well
2. **Stateless**: Agents should not maintain state between calls
3. **Idempotent**: Same input should produce same output
4. **Error Resilient**: Graceful degradation on failures

### Message Design

1. **Clear Types**: Use specific message types (not generic "data")
2. **Complete Context**: Include all necessary information in message
3. **Workflow Tracking**: Always include workflow_id for tracing
4. **Parent References**: Link related messages with parent_message_id

### Orchestration

1. **Timeout All Operations**: Never wait indefinitely
2. **Retry Transient Errors**: Use exponential backoff
3. **Circuit Breakers**: Stop calling failing agents
4. **Fallback Strategies**: Have backup plans for failures

## Related Documentation

- [Architecture Overview](architecture.md) - System architecture
- [Optimization System](optimization-system.md) - GEPA/MIPRO/SIMBA
- [Phoenix Integration](phoenix-integration.md) - Tracing and monitoring

**Last Updated**: 2025-10-04
