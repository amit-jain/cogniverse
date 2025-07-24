# Multi-Modal, Multi-Agent Retrieval System

### **🎯 What You Get**

A sophisticated AI system that can search through video content using natural language queries! This implementation provides a working multi-agent system engineered to perform advanced, multi-modal Retrieval-Augmented Generation (RAG). The system features autonomous agents that collaborate to understand and search video content using state-of-the-art AI models.

**🚀 Ready to start?** → See [QUICKSTART.md](QUICKSTART.md) for 3-step setup!

**📖 Documentation:** See the [docs/](docs/) directory for:
- [Detailed Setup Guide](docs/setup/detailed_setup.md) - Comprehensive installation instructions
- [Modal VLM Deployment](docs/modal/deploy_modal_vlm.md) - Vision-language model setup
- [Pipeline Documentation](docs/pipeline/README.md) - Video processing pipeline details
- [Testing Guide](docs/testing/) - Testing strategies and benchmarks

### **📁 System Architecture**

The current implementation focuses on video search capabilities with:
- **Video Search Agent**: Built with ColPali model and Vespa/Byaldi vector database
- **Composing Agent**: Central orchestrator using Google's Agent Development Kit (ADK)
- **A2A Protocol**: Agent-to-Agent communication standard for seamless integration
- **Text Search Agent**: Ready for future Elasticsearch integration (currently disabled)
- **Agentic Router Optimizer**: Advanced query routing system with automatic optimization

### **🧠 Intelligent Query Routing**

The system includes a sophisticated **Agentic Router Optimizer** (`src/optimizer/`) that automatically improves query routing decisions through teacher-student distillation using DSPy MIPROv2:

**Key Features:**
- **Provider-Agnostic Architecture**: Supports Modal, Ollama, Anthropic, and OpenAI
- **Teacher-Student Optimization**: Uses Claude/GPT-4 to teach smaller models like Gemma-3-1b
- **Production-Ready API**: Sub-100ms routing with auto-scaling via Modal or local Ollama
- **Unified Schema**: Consistent routing decisions (`search_modality` + `generation_type`)
- **Automatic Deployment**: Seamless artifact management and deployment pipeline

**Quick Start:**
```bash
# Option 1: Just run optimization (no deployment)
python scripts/run_orchestrator.py

# Option 2: Full workflow (optimization + deployment + testing)
python scripts/run_optimization.py

# Test the system
python tests/test_system.py
```

**Architecture:** `src/optimizer/` (optimization logic) + `src/inference/` (production API)

### **🔍 Advanced Search Capabilities**

The Vespa backend provides **13 different ranking strategies** for optimal search performance:
- **Text-only search** with BM25 fieldsets
- **Visual search** with ColPali embeddings (float/binary)
- **Hybrid search** combining text + visual with various phasing strategies
- **Automatic strategy selection** based on query characteristics
- **Input validation** ensuring proper query formatting for each strategy

See [CLAUDE.md](CLAUDE.md) for complete technical documentation.

---

## 🧪 Testing & Validation

The Cogniverse system includes a comprehensive testing suite to validate query routing, temporal extraction, and multi-modal search capabilities across different model configurations.

### Quick Testing
```bash
# Run comprehensive test with all models and approaches
python tests/test_comprehensive_routing.py

# Quick validation with limited models (faster)
python tests/test_comprehensive_routing.py quick
```

### Specialized Testing
```bash
# Test only LLM models (DeepSeek R1, Gemma 3, Qwen 3 variants)
python tests/test_comprehensive_routing.py llm-only

# Test only GLiNER models for entity recognition
python tests/test_comprehensive_routing.py gliner-only

# Test hybrid LLM + GLiNER approaches
python tests/test_comprehensive_routing.py hybrid-only
```

### Test Coverage
- **10 LLM Models**: DeepSeek R1 (1.5b, 7b, 8b), Gemma 3 (1b, 4b, 12b), Qwen 3 (0.6b, 1.7b, 4b, 8b)
- **GLiNER Models**: All models configured in `config.json`
- **Routing Accuracy**: Video vs Text vs Both classification testing
- **Temporal Extraction**: Date/time pattern recognition validation
- **Performance Metrics**: Response time, success rate, and accuracy analysis

### Generated Reports
- `comprehensive_test_results.csv` - Detailed per-query results with metrics
- `comprehensive_summary_report.json` - Aggregated performance rankings

### Test Data
- `data/test_queries.json` - Sample test queries for video search validation
  - Question-based queries (e.g., "What is the man doing in the video?")
  - Content-based queries (e.g., "over wearing head")
  - Expected video IDs and answers for validation

📖 **For detailed testing documentation:** See [tests/README.md](tests/README.md)

---

## Architecture Blueprint Documentation

Acknowledging the complexity of stateful agent frameworks, this revised blueprint introduces a phased implementation strategy. **Phase 1** focuses on rapid development and core functionality by building the composing agent using **Google's Agent Development Kit (ADK)**. This code-first toolkit simplifies the creation of a coordinator agent capable of intelligent routing and tool use without the initial overhead of complex memory management systems. The strategic foundation of this system rests on the Agent2Agent (A2A) protocol, which is employed as the universal communication standard, enabling seamless and secure interoperability between the ADK-based composing agent and the specialized retrieval agents.

A key refinement in this blueprint is the video retrieval mechanism. Instead of relying on explicit user cues, the composing agent will now intelligently infer the need for video search by analyzing the query for linguistic and temporal signals. The video agent itself has been redesigned for greater precision, utilizing **Vespa** as a high-performance vector database for production and offering **Byaldi** as a lightweight, easy-to-use alternative for rapid prototyping. The ingestion pipeline now includes detailed video pre-processing to extract and index frame-level, multi-vector embeddings from a model like ColQwen2, along with rich temporal metadata, enabling highly specific and context-aware video segment retrieval. This document serves as a detailed implementation guide for architects and technical leads tasked with building this next-generation Agentic RAG application.

## Section 1: Foundations of the Agentic Architecture

This section establishes the theoretical and architectural underpinnings of the entire system. It progresses from the high-level paradigm of Agentic RAG to the specific enabling technology—the A2A protocol for communication—that forms the foundation of this advanced retrieval system.

### 1.1. The Agentic RAG Paradigm in Practice

The proposed system represents a significant evolution from standard Retrieval-Augmented Generation (RAG) to a more dynamic and intelligent paradigm known as Agentic RAG (ARAG). Understanding this distinction is fundamental to appreciating the architectural choices and capabilities of the system.

#### Defining the Leap from RAG to Agentic RAG

Traditional RAG systems operate as a linear pipeline: a user query triggers a retrieval step from a single knowledge base, the retrieved context is appended to the prompt, and a Large Language Model (LLM) generates a response. While effective, this model is reactive and constrained. Agentic RAG, by contrast, introduces autonomous AI agents into the pipeline, transforming it from a static, rule-based process into an adaptive, intelligent problem-solving framework. Instead of merely retrieving and generating, an Agentic RAG system can reason, plan, decompose complex tasks, and execute multi-step workflows involving various tools and data sources.

#### Architectural Pillars of the Proposed System

The architecture described in this blueprint is a direct implementation of the core principles of Agentic RAG, designed to overcome the limitations of simpler systems.

  * **Flexibility & Multi-Source Retrieval:** The system is explicitly designed to pull information from multiple, distinct external knowledge sources: a proprietary text-based knowledge base powered by Elasticsearch and a new video-based knowledge base powered by Vespa. This ability to consult diverse sources is a hallmark of advanced Agentic RAG.
  * **Adaptability & Dynamic Routing:** The Composing Agent functions as an intelligent "Query Planner" or "Router". It analyzes the user's query to understand its intent and dynamically determines the optimal path for fulfillment—whether to consult the text agent, the video agent, or both. This moves beyond a static, hardcoded workflow, allowing the system to adapt its strategy based on the specific query.
  * **Accuracy through Collaboration:** The multi-agent structure promotes specialization. The Text Agent is an expert in hybrid lexical and semantic text search, while the Video Agent is an expert in visual content retrieval. The final synthesis by the Composing Agent, which integrates findings from both specialists, yields a more comprehensive and accurate response than a single, generalist agent could produce. This collaborative model mirrors the effectiveness of human expert teams.
  * **Multimodality:** By design, the system natively handles and synthesizes heterogeneous data types—structured text results from Elasticsearch and semantic video embeddings from Vespa. This capability to work across different data modalities is a key advantage of modern Agentic RAG implementations.

This system is best understood not as a collection of disparate tools, but as a cognitive team. The architecture mirrors a human expert team structure, with the Composing Agent acting as the "Team Lead" or "Supervisor Agent". This supervisor analyzes an incoming project (the user query), delegates sub-tasks to the appropriate specialists (the "Archivist" Text Agent and the "Visual Analyst" Video Agent), and then synthesizes their individual reports into a final, cohesive deliverable. This conceptual framing is more than an analogy; it directly informs the design of the agents' roles, prompts, and interaction patterns. The Composing Agent's logic is not about simple "tool calling" but about sophisticated "expert consultation," delegation, and synthesis, leading to a more robust and intelligent system.

### 1.2. The A2A Protocol: A Universal Language for Agent Collaboration

To enable effective collaboration within this cognitive team, a standardized communication protocol is required. Traditional REST APIs, while functional, treat services as passive endpoints. The Agent2Agent (A2A) protocol, by contrast, is designed to facilitate communication between autonomous, intelligent entities, making it the ideal choice for this architecture.

#### Core Tenets and Justification

The A2A protocol is an open standard designed to enable AI agents to discover, communicate, and collaborate securely and seamlessly, regardless of their underlying frameworks or platforms. Its adoption is justified by several key characteristics:

  * **Agent-Centric Design:** A2A treats agents as autonomous entities with specialized intelligence, not merely as functional endpoints to be called. This aligns perfectly with the Agentic RAG paradigm.
  * **Open Standard on Web Technologies:** The protocol is built upon familiar and widely adopted web standards, including HTTP(S), JSON-RPC 2.0, and Server-Sent Events (SSE). This significantly lowers the barrier to adoption and integration within existing enterprise IT stacks.
  * **Industry-Wide Collaboration:** A2A is a collaborative effort with support from over 100 technology companies, including SAP, Salesforce, and ServiceNow, and is now hosted by the Linux Foundation, ensuring its vendor neutrality and long-term viability.

#### A2A Technical Deep Dive

The protocol's functionality is defined by a set of core components and mechanisms:

  * **Agent Discovery via Agent Cards:** Agents advertise their capabilities through a standardized JSON object called an `AgentCard`. This card acts as a service manifest, detailing the agent's identity (name, description), its service endpoint URL, supported A2A protocol versions and capabilities (like streaming or push notifications), and, critically, the specific skills it offers and its authentication requirements. This allows the Composing Agent to dynamically discover and understand how to interact with the specialist agents.
  * **Task-Oriented Communication:** All substantive interactions in A2A are encapsulated within a `Task` object. A task is initiated by a client agent, assigned a unique ID by the server agent, and progresses through a well-defined lifecycle with states such as `submitted`, `working`, `input-required`, `completed`, and `failed`. This stateful, task-oriented approach is essential for managing the potentially long-running retrieval and analysis processes that this system will handle, which may take hours or even days to complete.
  * **Multi-Modal Message Structure:** Communication within a task occurs via `Message` objects. Each message is composed of one or more `Part` objects, which carry the actual content. The protocol defines several `Part` types, including `TextPart` for plain text, `FilePart` for binary data (sent either inline as base64 or via URI), and `DataPart` for structured JSON data. This native support for multi-modal payloads is fundamental to the system's ability to handle diverse information types.

While powerful, the A2A protocol as currently specified primarily facilitates direct, point-to-point connections between agents. In a large, complex system, this could lead to the "N-squared integration problem," where every agent must be aware of every other agent, resulting in brittle integrations and high operational overhead. The proposed architecture inherently mitigates this challenge. By positioning the Composing Agent as a central supervisor and router, the system adopts a hub-and-spoke model. The specialist agents only need to communicate with the Composing Agent, which abstracts the complexity of the underlying agent network. This makes the Composing Agent not just a desirable feature but an architectural necessity for building a scalable and maintainable system using the A2A protocol.

### Table 1: A2A Protocol vs. Traditional REST APIs for Agentic Systems

The following table justifies the architectural decision to use the A2A protocol by highlighting its specific advantages for the complex, stateful, and asynchronous interactions required in this project.

| Feature | Traditional REST API | A2A Protocol |
| :--- | :--- | :--- |
| **Task Management** | Inherently stateless. Each request is independent. Stateful, long-running operations require complex client-side logic and polling mechanisms. | Natively stateful and task-oriented. Defines a `Task` object with a managed lifecycle (`submitted`, `working`, `completed`, etc.), ideal for long-running processes. |
| **Interaction Model** | Primarily synchronous request/response. Asynchronous patterns require custom implementations (e.g., webhooks, polling). | Supports multiple interaction models out-of-the-box: synchronous, polling (`tasks/get`), streaming (`message/stream` via SSE), and push notifications (webhooks). |
| **Payload Structure** | Rigid, predefined schemas. Extending to support new data types (e.g., video, files) often requires versioning the API. | Flexible and multi-modal by design. Uses a `Message` and `Part` structure (`TextPart`, `DataPart`, `FilePart`) to handle heterogeneous data within a single request. |
| **Service Discovery** | Manual. Clients must be explicitly configured with the endpoint URLs and API specifications for each service. | Automated discovery via `AgentCard`. Agents publish their capabilities, skills, and connection details in a standardized JSON format, enabling dynamic discovery. |
| **Collaboration Model** | Master-slave or client-server. One system calls another as a "tool" or a passive data source. | Peer-to-peer collaboration. Treats agents as autonomous entities that can negotiate and collaborate on complex tasks, aligning with the agentic paradigm. |

## Section 2: Blueprints for the Specialist Retrieval Agents

This section provides the detailed, from-scratch implementation plans for the two specialist "worker" agents: the Hybrid Text Search Agent and the advanced Video Search Agent. These blueprints cover everything from the underlying data processing and retrieval logic to their encapsulation as interoperable A2A services.

### 2.1. The Hybrid Text Search Agent

This agent exposes an existing internal AI model built on Elasticsearch. The focus of this blueprint is on ensuring its search capabilities are robust and that it is correctly integrated into the multi-agent system via the A2A protocol.

#### 2.1.1. Implementing Advanced Hybrid Search in Elasticsearch

Hybrid search, which combines the strengths of traditional keyword-based (lexical) search and modern meaning-based (semantic) search, delivers superior relevance compared to either method alone. The implementation involves three key stages: index mapping, data ingestion, and query construction.

##### Index Mapping

A robust hybrid search requires an index mapping that can accommodate both lexical and vector data. The index should be created with at least two key fields: a `text` field for full-text search and a `dense_vector` field for semantic search embeddings. The `copy_to` parameter is a highly effective mechanism for streamlining data ingestion. By using `copy_to`, the raw text content indexed into the `text` field is automatically duplicated into a `semantic_text` field, which can then be processed by an inference pipeline to generate embeddings without requiring a separate data handling step.

##### Data Ingestion and Embedding

Data must be indexed along with its vector representation. This is best handled using an `inference` ingest processor within an ingest pipeline. This processor can call a deployed text embedding model (such as ELSER or a third-party model) to generate a vector from the `content_embedding_source` field and store it in the `content_vector` field at ingest time. For existing data, the `_reindex` API can be used to process all documents through this pipeline, populating the vector field across the entire dataset.

##### Query Construction with Reciprocal Rank Fusion (RRF)

The most effective way to combine lexical and semantic search results is with Reciprocal Rank Fusion (RRF). RRF is a ranking algorithm that merges result sets from different queries, giving higher scores to documents that rank well in any of the individual result lists, without needing complex score normalization.

Using the `elasticsearch-py` client, a hybrid query can be constructed with the `retriever` API. This involves defining an `rrf` retriever that contains two sub-retrievers: a `standard` retriever for the lexical `match` query and a `knn` retriever for the semantic vector search. Tuning `k` (the number of nearest neighbors to return) and `num_candidates` (the number of candidates to consider on each shard) is crucial for balancing performance and recall. Higher values increase accuracy at the cost of latency.

#### 2.1.2. Encapsulation as an A2A Service

To integrate this search capability into the agentic system, it must be wrapped in an A2A-compliant server.

  * **Defining the Agent Card:** An `AgentCard` must be created to advertise the agent's capabilities. This JSON file will specify its unique `name` ("HybridTextSearchAgent"), `description`, service `url`, and the `skills` it offers, such as a skill named "hybridTextSearch". The card will also define the expected input and output formats, likely using `DataPart` for structured JSON queries and results.
  * **Implementing the A2A Server:** A web server (e.g., using FastAPI) will expose an endpoint that conforms to the A2A protocol. It will listen for `tasks/send` requests. Upon receiving a task, it will parse the incoming A2A `Message`, extract the query from the `DataPart`, construct and execute the Elasticsearch hybrid query as shown above, and then package the results into a new A2A `Message` to be sent back to the client (the Composing Agent).

### 2.2. The Advanced Video Search Agent: A Detailed Implementation

This agent will be built from the ground up to provide semantic search capabilities over a corpus of videos. This revised design moves beyond simple frame retrieval to a more sophisticated multi-modal, multi-vector approach inspired by state-of-the-art models like ColPali and Video-ColBERT. The complete, runnable code for the ingestion pipeline and the agent server can be found in the **Appendix**.

#### 2.2.1. Designing the Video Index

For the video search backend, we propose two alternatives to suit different stages of development.

##### Option A: Production-Grade Backend with Vespa

For a robust, scalable, and production-ready system, **Vespa** is the recommended choice. It is a high-performance search engine and vector database known for its real-time capabilities and native support for multi-vector indexing and phased ranking.

The Vespa schema is designed to store the multi-vector visual embeddings alongside the fused textual data for each keyframe. This schema allows for a powerful two-phase hybrid search: a fast lexical search on the VLLM descriptions and audio transcripts, followed by a precise multi-vector visual search on the top candidates.

##### Option B: Simplified Prototyping with Byaldi

For rapid development and prototyping, **Byaldi** offers a much simpler, lightweight alternative. Byaldi is a Python library that acts as a user-friendly wrapper around complex multi-vector models like ColQwen2, managing an in-memory vector index automatically. This eliminates the need to set up and manage a separate database service, making it ideal for the initial stages of a project.

### Table 2: Specialist Agent A2A Interface Specification

This table serves as a concrete API contract, defining the precise data structures the Composing Agent will use to interact with the specialist agents. This is essential for enabling parallel development and ensuring seamless integration.

| Agent | A2A Method | Input `Part` (Type & Schema) | Output `Part` (Type & Schema) | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Text Search Agent** | `tasks/send` | **Type:** `DataPart` \<br\> **Schema:** `{"query": "string", "top_k": int}` | **Type:** `DataPart` \<br\> **Schema:** `{"results": [{"doc_id": "string", "content": "string", "score": float}]}` | Executes a hybrid (lexical + semantic) search on the text knowledge base and returns the top K documents. |
| **Video Search Agent** | `tasks/send` | **Type:** `DataPart` \<br\> **Schema:** `{"query": "string", "top_k": int, "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}` (dates are optional) | **Type:** `DataPart` \<br\> **Schema:** `{"results": [{"video_id": "string", "frame_id": int, "start_time": float, "score": float}]}` | Executes a hybrid semantic search for video frames using visual, audio, and descriptive text, with optional temporal filtering. |

## Section 3: The Composing Agent: A Phased Approach to Orchestration

This section details the architecture and implementation of the Composing Agent, the "brain" of the system. To address the complexity of building stateful agentic systems, we propose a phased approach, starting with a simplified yet powerful implementation.

### 3.1. Phase 1: Simplified Orchestration with Google's Agent Development Kit (ADK)

For the initial implementation phase, we will leverage **Google's Agent Development Kit (ADK)**. ADK is an open-source, code-first Python toolkit designed to build, evaluate, and deploy sophisticated AI agents with flexibility and control. It is model-agnostic and provides a modular framework that simplifies the creation of multi-agent systems, making it an ideal choice for building a robust coordinator without the initial overhead of more complex state and memory management frameworks.

#### 3.1.1. Introducing the ADK Framework

ADK allows developers to define agent logic, tools, and orchestration directly in Python, treating agent development like traditional software development. Its key strengths for this project include:

  * **Code-First Development:** Enables clear, version-controlled, and testable agent definitions.
  * **Modular Multi-Agent Systems:** ADK is designed for composing multiple specialized agents into flexible hierarchies, which perfectly fits our coordinator/specialist pattern.
  * **Simplified Orchestration:** ADK provides high-level abstractions like `SequentialAgent` and `ParallelAgent`, and supports LLM-driven delegation to manage workflows.

#### 3.1.2. Building the Coordinator Agent in ADK

The Composing Agent will be implemented as a parent `LlmAgent` in ADK. This agent will not have direct tools for searching; instead, its "tools" will be other agents (the Text and Video Search agents). ADK's LLM-driven delegation is the core mechanism for routing. The complete, runnable code for the Composing Agent can be found in the **Appendix**.

### 3.2. Intelligent Query Analysis and Routing

A critical enhancement in this design is moving away from requiring explicit user instructions for video search. The Composing Agent will now infer the user's intent.

#### 3.2.1. Detecting Video Search Intent and Temporal Cues

The system must intelligently determine when a query pertains to video content. This is achieved by analyzing the query for specific signals:

  * **Explicit Keywords:** The presence of words like "video," "clip," "scene," "recording," "show me," or "find a moment" are strong indicators.
  * **Implicit Intent:** Phrases suggesting visual action ("a person walking," "a car driving") can imply a need for video search.
  * **Temporal Filters:** The query may contain temporal information like "last week," "in yesterday's meeting," or specific dates. This information is crucial for filtering results in the Video Search Agent.

To reliably extract this intent, a lightweight classification model can be used as a preliminary check. A fine-tuned, compact BERT-based model like **`distilbert-base-uncased`** or **`all-MiniLM-L6-v2`** is ideal. These models are small and fast, making them suitable for a quick pre-processing step to classify the query as `text`, `video`, or `hybrid` before it's passed to the main Composing Agent's LLM.

#### 3.2.2. Prompting for Multi-Modal Routing and Metadata Extraction

The core routing logic will reside in the prompt for the ADK-based Composing Agent. This prompt instructs the LLM on how to analyze the query and delegate tasks.

**Sample Router Prompt for the Composing Agent:**

```
You are a master research assistant and task router. Your job is to analyze the user's query and delegate it to the correct specialist agent. You have two specialist agents available:
1.  **TextSearchAgent**: Use this for queries about documents, reports, and general text-based information.
2.  **VideoSearchAgent**: Use this for queries that ask for visual information, scenes, clips, or recordings. Keywords like 'video', 'show me', 'scene of' indicate a video search.

Your task:
1.  Analyze the user's query: "{query}".
2.  Determine if the query is best answered by the TextSearchAgent, the VideoSearchAgent, or both.
3.  **Temporal Analysis**: Scrutinize the query for any dates, date ranges, or relative time references (e.g., "yesterday", "last week"). If found, extract the start and end dates in YYYY-MM-DD format. Today's date is {current_date}.
4.  Delegate the task by calling the appropriate agent. If you extract temporal data, you MUST include `start_date` and `end_date` in your call to the VideoSearchAgent.
```

This prompt explicitly guides the LLM to perform the necessary analysis and data extraction, making the routing process robust and intelligent.

### 3.3. Synthesizing the Final Response

The final stage of the workflow is to synthesize the retrieved heterogeneous data into a single, user-friendly response.

#### 3.3.1. Prompt Engineering for Heterogeneous Data Summarization

The `composing_agent` is responsible for this final step after receiving results from the specialist agents. Its logic will:

1.  Collect the retrieved context (text snippets and video search results) from its session state.
2.  Format this context into a comprehensive prompt for a powerful generative LLM (e.g., GPT-4o, Claude 3.5 Sonnet).

The design of this prompt is critical for achieving a high-quality summary. It should not simply ask the model to "summarize the following." Instead, it should be engineered to guide the model in synthesizing insights from disparate sources. A hybrid abstractive/extractive approach is recommended, where the model is encouraged to use key phrases from the source material while generating a new, coherent narrative.

Key elements of a successful synthesis prompt include:

  * **Role Definition:** "You are an expert research analyst. Your task is to synthesize information from text documents and video analysis reports into a single, comprehensive summary."
  * **Context Provision:** The prompt will clearly delineate the sources: "The following information was retrieved from text documents: [text snippets]... The following information was retrieved from video analysis: [video descriptions]...".
  * **Task Specificity:** "Analyze all the provided information. Identify the key themes, entities, and events. Create a concise, well-structured summary that integrates insights from both the text and video sources. Where possible, correlate events mentioned in the text with visuals described in the video analysis."
  * **Output Format Specification:** "The final output should be in Markdown format."

This structured approach ensures the LLM understands its role and the nature of the heterogeneous data, leading to a more accurate and insightful summary than a generic prompt would produce.

### Table 3: ADK Agent State Management

This table provides a conceptual overview of how state will be managed within the ADK framework for the Composing Agent.

| State Key | Scope | Purpose | Example Value |
| :--- | :--- | :--- | :--- |
| **`initial_query`** | Session | Stores the original user query for the duration of the task. | `"Show me video of the new drone prototype from last week's presentation."` |
| **`routing_decision`** | Temporary | Holds the output from the routing logic. | `{"target_agent": "VideoSearchAgent", "query": "new drone prototype", "start_date": "2025-06-25", "end_date": "2025-07-01"}` |
| **`text_results`** | Session | Stores the results retrieved by the Text Search Agent. | `[{"doc_id": "doc_123", "content": "...", "score": 0.91}]` |
| **`video_results`** | Session | Stores the results retrieved by the Video Search Agent. | `[{"video_id": "vid_456", "frame_id": 78, "start_time": 123.45, "score": 0.88}]` |

## Section 4: System-Level Considerations and Future Directions

This final section addresses the practical aspects of deploying and scaling the multi-agent system. It also provides a strategic analysis of alternative technologies and outlines a roadmap for the system's future evolution.

### 4.1. Deployment and Scalability Strategy

A production-ready system requires a robust deployment and scalability plan.

  * **Deployment Architecture:** The recommended deployment strategy is to containerize each component of the system using Docker. This includes separate containers for:
    1.  The Composing Agent (running the ADK application).
    2.  The Text Search Agent (A2A server).
    3.  The Video Search Agent (A2A server).
    4.  The Elasticsearch cluster.
    5.  The Vespa cluster.
        This containerized approach ensures environmental consistency and simplifies dependency management. For production, the ADK agent can be deployed to a scalable, managed runtime like **Google Cloud Run** or **Vertex AI Agent Engine**, as recommended by the ADK documentation.
  * **Scalability Concerns and Mitigations:** The primary scalability challenge in an A2A-based system is the potential for the "N-squared" integration problem, where the number of connections grows quadratically with the number of agents. As discussed, the supervisor/router architecture of the Composing Agent effectively mitigates this by creating a hub-and-spoke topology. As the system grows, new specialist agents only need to connect to the central Composing Agent, not to every other agent. Another critical consideration is the governance of `AgentCard` discovery. To prevent the use of malicious or outdated agents, it is recommended to implement a central, trusted agent registry where `AgentCard`s are published and vetted. The Composing Agent would consult this registry for discovering and authenticating specialist agents, rather than relying on untrusted, ad-hoc discovery mechanisms.

### 4.2. Analysis of Alternative Technologies

The technologies chosen for this blueprint represent a powerful and coherent stack. However, it is prudent to be aware of alternatives to inform future architectural decisions and adaptations.

  * **Agent Framework:** Google ADK was chosen for its simplicity and code-first approach in Phase 1.
      * **Alternative 1: LangGraph:** A library for building stateful, multi-agent applications as graphs. It is more flexible for complex, cyclic workflows but has a steeper learning curve. It would be a candidate for a "Phase 2" implementation.
      * **Alternative 2: CrewAI:** This framework excels at orchestrating role-based collaboration. It would be a strong alternative if the system's logic were to evolve towards a more explicit simulation of a team with defined roles like "Researcher," "Writer," and "Editor".
  * **Video Understanding:** ColQwen2 was selected for its state-of-the-art performance in text-to-video retrieval.
      * **Alternative 1: InternVideo:** Another powerful foundation model for video that has shown competitive performance on various benchmarks. It represents a strong alternative from a different research group and could be swapped in if its performance characteristics better suit a specific domain.
      * **Alternative 2: SmolVLM2:** A family of smaller, highly efficient video language models designed to run on-device. While likely not as powerful as larger models, they represent a compelling option for future applications where edge processing or lower computational cost is a primary concern.
  * **Video Vector DB:** Vespa was chosen for its real-time performance and strong hybrid search capabilities.
      * **Alternative 1: Byaldi:** A lightweight, in-memory vector store wrapper for ColPali-style models. Excellent for rapid prototyping and development due to its ease of use, but not suitable for large-scale, persistent production deployments.
      * **Alternative 2: Qdrant:** A purpose-built, open-source vector database optimized for high-performance semantic search. It is a strong contender but would introduce another distinct database system to manage.

### 4.3. Conclusion and Recommendations

This report has detailed a revised architectural blueprint for a highly advanced Agentic RAG system. The core architectural decisions—a phased implementation starting with Google's ADK, employing the A2A protocol for interoperability, and using specialized retrieval agents with Vespa for video and Elasticsearch for text—provide a robust and scalable foundation. The intelligent query router elevates the system from a simple tool to a reasoning engine.

By following this blueprint, developers can construct a system that moves beyond simple information retrieval and into the realm of automated reasoning and synthesis. The resulting application will be capable of understanding complex, multi-modal user queries and delivering nuanced, comprehensive answers that draw from a rich set of heterogeneous data.

**Recommendations for Future Enhancements:**

  * **Phase 2 - Advanced State Management:** Evolve the Composing Agent by migrating from ADK to a more complex framework like **Letta** or **LangGraph**. This would introduce persistent, long-term memory, allowing the agent to learn from past interactions and manage more sophisticated, long-running tasks.
  * **Expand Specialist Agents:** The architecture is modular, allowing for the addition of new specialist agents. An **Audio Search Agent** that processes audio tracks from videos or other sources would be a natural extension.
  * **Fine-Tune the Summarization LLM:** To improve the quality of the final synthesis, the summarization LLM can be fine-tuned on a domain-specific dataset of high-quality summaries generated from text and video sources. This will adapt the model to the specific language and nuances of the target domain, leading to more accurate and contextually appropriate outputs.
  * **Incorporate Human-in-the-Loop Feedback:** A mechanism for users to rate or correct the final summaries can be implemented. This feedback can be collected and used to further refine the prompts for the router and synthesizer agents, creating a system that continuously learns and improves over time.

### Table 4: Technology Stack Alternatives and Trade-offs

This table provides a strategic overview of the technology landscape, validating the chosen stack while offering a clear view of viable alternatives for future evolution or adaptation.

| Component | Chosen Technology | Alternative 1: Pros & Cons | Alternative 2: Pros & Cons |
| :--- | :--- | :--- | :--- |
| **Agent Framework** | **Google ADK (Phase 1)** | **LangGraph:** \<br\> **Pros:** Highly flexible for complex, stateful, and cyclic workflows. \<br\> **Cons:** Steeper learning curve; more initial setup required. | **CrewAI:** \<br\> **Pros:** Excellent for role-based collaboration; intuitive for team-based agent design. \<br\> **Cons:** Less flexible for arbitrary graph-based logic compared to LangGraph. |
| **Video Vector DB** | **Vespa** | **Byaldi:** \<br\> **Pros:** Extremely easy to use; no setup required; great for rapid prototyping. \<br\> **Cons:** In-memory index; not designed for persistence or large-scale production use. | **Qdrant:** \<br\> **Pros:** A purpose-built, open-source vector database optimized for high-performance semantic search. \<br\> **Cons:** Adds another distinct database system to manage and maintain. |
| **Video Understanding** | **ColQwen2** | **InternVideo:** \<br\> **Pros:** Another SOTA foundation model with strong benchmark performance. \<br\> **Cons:** Different architectural philosophy; requires evaluation to see if it offers an advantage for the specific video domain. | **SmolVLM2:** \<br\> **Pros:** Highly efficient, designed for on-device or low-resource environments. \<br\> **Cons:** Lower performance than large-scale models; a trade-off of capability for efficiency. |
| **Agent Interoperability** | **A2A Protocol** | **REST/gRPC:** \<br\> **Pros:** Ubiquitous, well-understood. \<br\> **Cons:** Not designed for agentic, stateful, or multi-modal peer collaboration; requires significant custom logic. | **MCP (Model Context Protocol):** \<br\> **Pros:** Strong standard for connecting agents to tools. \<br\> **Cons:** More focused on the agent-tool interface than peer-to-peer agent collaboration; A2A is complementary. | |