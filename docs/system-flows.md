# System Flows and Architecture Diagrams

## Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Query Routing Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   GLiNER     │→ │     LLM      │→ │   Keyword    │         │
│  │  (Fast Path) │  │ (Slow Path)  │  │  (Fallback)  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Search Execution                              │
│  ┌──────────────────────────────────────────────────┐          │
│  │            Vespa Vector Database                  │          │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │          │
│  │  │  BM25    │  │  Float   │  │  Binary  │      │          │
│  │  │  Text    │  │ Embeddings│ │ Embeddings│     │          │
│  │  └──────────┘  └──────────┘  └──────────┘      │          │
│  └──────────────────────────────────────────────────┘          │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Result Generation                             │
│         Ranking → Reranking → Response Synthesis                │
└─────────────────────────────────────────────────────────────────┘
```


## Module-Specific Diagrams

Detailed flow diagrams for individual modules have been moved to their respective README files:

- **Query Routing Flow**: See `src/app/routing/README.md`
- **Video Ingestion Pipeline**: See `src/app/ingestion/README.md`
- **Vespa Search Strategies**: See `src/backends/vespa/README.md`
- **Evaluation Framework**: See `src/evaluation/README.md`

## Data Processing Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     Input Layer                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │  Videos  │  │  Images  │  │   PDFs   │  │  Audio   │     │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘     │
└────────┼─────────────┼─────────────┼─────────────┼────────────┘
         ↓             ↓             ↓             ↓
┌────────────────────────────────────────────────────────────────┐
│                  Processing Strategy Layer                      │
│  ┌──────────────────────────────────────────────────────┐     │
│  │         ProcessingStrategySet (Orchestrator)         │     │
│  └──────────────────────────────────────────────────────┘     │
│         ↓                    ↓                    ↓            │
│  ┌──────────┐        ┌──────────┐        ┌──────────┐        │
│  │Segmentation│      │Transcription│     │Description│        │
│  │ Strategy  │       │  Strategy   │     │ Strategy  │        │
│  └──────────┘        └──────────┘        └──────────┘        │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                    Embedding Generation                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ ColPali  │  │ ColQwen  │  │VideoPrism│  │  Custom  │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                    Document System                              │
│            Universal Document Format (Backend-agnostic)         │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  media_type | embeddings | metadata | temporal_info  │     │
│  └──────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                    Storage Backend                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │  Vespa   │  │  Elastic │  │  Pinecone│  │  Custom  │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
└────────────────────────────────────────────────────────────────┘
```

## Cache System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    Cache Manager                                │
│              (Unified Interface & Orchestration)                │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                    Cache Backends                               │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │   Filesystem     │  │  Object Storage  │                   │
│  │  ┌──────────┐   │  │  ┌──────────┐   │                   │
│  │  │  Local   │   │  │  │    S3    │   │                   │
│  │  └──────────┘   │  │  └──────────┘   │                   │
│  │  ┌──────────┐   │  │  ┌──────────┐   │                   │
│  │  │Structured│   │  │  │   GCS    │   │                   │
│  │  └──────────┘   │  │  └──────────┘   │                   │
│  └──────────────────┘  └──────────────────┘                   │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                    Cache Operations                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │   Get    │  │   Set    │  │  Delete  │  │   TTL    │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
└────────────────────────────────────────────────────────────────┘
```

## Component Interaction Sequence

```mermaid
sequenceDiagram
    participant U as User
    participant R as Router
    participant V as Vespa
    participant E as Evaluator
    participant P as Phoenix
    
    U->>R: Submit Query
    R->>R: Analyze Query (GLiNER/LLM)
    R->>V: Search Request
    V->>V: Execute Strategy
    V-->>R: Search Results
    R-->>U: Response
    
    Note over E: Evaluation Process
    E->>V: Test Queries
    V-->>E: Results
    E->>E: Calculate Metrics
    E->>P: Send Traces
    P->>P: Store & Visualize
    P-->>U: Dashboard
```

## Deployment Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                    Production Environment                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Load Balancer                          │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Application Instances (2-10)                │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │  │
│  │  │  App #1  │  │  App #2  │  │  App #N  │  ...        │  │
│  │  └──────────┘  └──────────┘  └──────────┘             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 Service Layer                             │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │  │
│  │  │  Vespa   │  │  Phoenix │  │  Ollama  │             │  │
│  │  │ Cluster  │  │  Server  │  │ Inference│             │  │
│  │  │  (3+)    │  │    (1)   │  │   (2+)   │             │  │
│  │  └──────────┘  └──────────┘  └──────────┘             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 Storage Layer                             │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │  │
│  │  │  Model   │  │  Cache   │  │  Data    │             │  │
│  │  │  Store   │  │  Store   │  │  Store   │             │  │
│  │  └──────────┘  └──────────┘  └──────────┘             │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Module Dependencies

```mermaid
graph LR
    A[Main Application] --> B[Routing Module]
    A --> C[Ingestion Module]
    A --> D[Evaluation Module]
    
    B --> E[GLiNER Models]
    B --> F[LLM Services]
    
    C --> G[Video Processors]
    C --> H[Embedding Models]
    C --> I[Vespa Backend]
    
    D --> J[Phoenix Tracing]
    D --> K[Inspect AI]
    
    G --> L[FFmpeg]
    G --> M[Whisper]
    H --> N[ColPali]
    H --> O[VideoPrism]
    
    I --> P[Vespa Instance]
    J --> Q[Phoenix Server]
    F --> R[Ollama/Modal]
```

## Performance Flow

```
Request → Router → Backend → Response
   ↓        ↓         ↓         ↓
  <1ms    <50ms    <100ms    <200ms
(keyword) (GLiNER)  (Vespa)   (total)

Ingestion → Processing → Embedding → Storage
    ↓          ↓           ↓          ↓
   1s/video   2s/video   3s/video   <1s
  (extract)  (transcribe) (embed)   (feed)
```

## Error Handling Flow

```mermaid
graph TD
    A[Operation] --> B{Success?}
    B -->|Yes| C[Continue]
    B -->|No| D{Retryable?}
    
    D -->|Yes| E[Exponential Backoff]
    E --> F{Max Retries?}
    F -->|No| A
    F -->|Yes| G[Log Error]
    
    D -->|No| G
    G --> H{Critical?}
    
    H -->|Yes| I[Alert & Fail]
    H -->|No| J[Fallback Strategy]
    
    J --> K[Alternative Path]
    K --> C
```

## Notes

- All performance metrics shown are estimates and should be measured in production
- Mermaid diagrams can be rendered in GitHub, VSCode with extensions, or online tools
- ASCII diagrams are universally viewable but less interactive
- Component boundaries represent logical separation, not necessarily physical deployment