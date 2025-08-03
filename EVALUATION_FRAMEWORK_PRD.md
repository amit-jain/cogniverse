# Product Requirements Document: Comprehensive Evaluation Framework for Cogniverse

## 1. Executive Summary

This PRD outlines the integration of Inspect AI and Arize Phoenix into Cogniverse to create a comprehensive evaluation framework for the multi-agent video RAG system. The framework will provide structured evaluation processes, detailed tracing, experiment tracking, and performance metrics for video retrieval tasks.

### Key Objectives
- Implement structured evaluation using Inspect AI's task-based framework
- Add comprehensive tracing and observability with Arize Phoenix
- Create reproducible experiments with versioned datasets
- Enable real-time performance monitoring and A/B testing
- Provide detailed metrics for retrieval quality assessment

## 2. System Architecture Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Cogniverse Application                       │
├─────────────────────────┬────────────────────┬──────────────────┤
│     Video Pipeline      │   Search Backend   │    Agents        │
│  (Processing & Ingest)  │   (Vespa/Others)   │  (ADK/Encoders)  │
└────────────┬────────────┴────────┬───────────┴──────────────────┘
             │                     │
             │    OpenTelemetry    │
             │    Instrumentation  │
             │                     │
┌────────────┴─────────────────────┴────────────────────────────┐
│                    Evaluation Framework Layer                   │
├─────────────────────────────┬──────────────────────────────────┤
│        Inspect AI           │         Arize Phoenix            │
│  ┌────────────────────┐    │    ┌──────────────────────┐     │
│  │  Task Definitions  │    │    │   Trace Collection   │     │
│  │  Dataset Manager   │    │    │   Span Analysis      │     │
│  │  Solver Chains     │    │    │   Metrics Dashboard  │     │
│  │  Scoring Engine    │    │    │   Experiment Tracker │     │
│  └────────────────────┘    │    └──────────────────────┘     │
└─────────────────────────────┴──────────────────────────────────┘
```

### 2.2 Component Integration Map

```python
# Core Integration Points
class EvaluationFramework:
    components = {
        "inspect_ai": {
            "purpose": "Structured evaluation tasks",
            "integrations": ["dataset_creation", "task_execution", "scoring"]
        },
        "arize_phoenix": {
            "purpose": "Observability and experimentation",
            "integrations": ["tracing", "metrics", "experiments", "datasets"]
        },
        "cogniverse": {
            "instrumented_components": [
                "video_pipeline",
                "search_backends",
                "query_encoders",
                "agents"
            ]
        }
    }
```

## 3. Detailed Component Specifications

### 3.1 Inspect AI Integration

#### 3.1.1 Task Definition Structure
```python
# src/evaluation/inspect_tasks/video_retrieval.py
from inspect_ai import Task, task, eval_model
from inspect_ai.solver import generate, chain_of_thought
from inspect_ai.scorer import includes, model_graded_fact

@task
def video_retrieval_accuracy():
    """Evaluate video retrieval accuracy across different query types"""
    return Task(
        dataset=VideoRetrievalDataset("cogniverse/eval/video_queries"),
        solver=[
            CogniverseRetrievalSolver(),
            ResultRankingAnalyzer(),
            RelevanceJudgmentCollector()
        ],
        scorer=VideoRetrievalScorer(
            metrics=["mrr", "ndcg", "precision", "recall"]
        )
    )

@task
def temporal_understanding():
    """Evaluate temporal query understanding"""
    return Task(
        dataset=TemporalQueryDataset("cogniverse/eval/temporal"),
        solver=[
            TemporalQueryProcessor(),
            TimeRangeExtractor(),
            CogniverseRetrievalSolver()
        ],
        scorer=TemporalAccuracyScorer()
    )

@task
def multimodal_alignment():
    """Evaluate cross-modal understanding"""
    return Task(
        dataset=MultimodalDataset("cogniverse/eval/multimodal"),
        solver=[
            VisualQueryEncoder(),
            TextQueryEncoder(),
            CrossModalAlignmentChecker()
        ],
        scorer=AlignmentScorer()
    )
```

#### 3.1.2 Custom Solvers for Video RAG
```python
# src/evaluation/inspect_tasks/solvers.py
class CogniverseRetrievalSolver:
    """Custom solver for Cogniverse retrieval evaluation"""
    
    def __init__(self, profiles: List[str], strategies: List[str]):
        self.profiles = profiles
        self.strategies = strategies
        self.search_service = SearchService()
    
    async def solve(self, state: SolverState) -> SolverState:
        query = state.input.text
        results = {}
        
        # Test across profiles and strategies
        for profile in self.profiles:
            for strategy in self.strategies:
                with phoenix_trace(
                    name="retrieval_solve",
                    attributes={
                        "profile": profile,
                        "strategy": strategy,
                        "query": query
                    }
                ):
                    search_results = await self.search_service.search(
                        query=query,
                        profile=profile,
                        ranking_strategy=strategy
                    )
                    results[f"{profile}_{strategy}"] = search_results
        
        state.metadata["retrieval_results"] = results
        return state

class ResultRankingAnalyzer:
    """Analyze ranking quality across different strategies"""
    
    async def solve(self, state: SolverState) -> SolverState:
        results = state.metadata["retrieval_results"]
        expected = state.target.expected_videos
        
        rankings = {}
        for config, search_results in results.items():
            rankings[config] = self._analyze_ranking(
                search_results, expected
            )
        
        state.metadata["ranking_analysis"] = rankings
        return state
```

#### 3.1.3 Dataset Management
```python
# src/evaluation/datasets/video_queries.py
class VideoRetrievalDataset:
    """Structured dataset for video retrieval evaluation"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.phoenix_dataset_id = None
    
    def load(self) -> List[Sample]:
        samples = []
        
        # Load from versioned dataset
        with phoenix.Dataset(name="video_retrieval_v1") as dataset:
            self.phoenix_dataset_id = dataset.id
            
            for item in dataset:
                sample = Sample(
                    input=QueryInput(
                        text=item["query"],
                        query_type=item["category"],
                        metadata=item.get("metadata", {})
                    ),
                    target=ExpectedResults(
                        expected_videos=item["expected_videos"],
                        relevance_scores=item.get("relevance_scores", {})
                    ),
                    id=item["id"]
                )
                samples.append(sample)
        
        return samples
    
    def add_evaluation_results(self, results: Dict[str, Any]):
        """Add evaluation results back to Phoenix dataset"""
        phoenix.log_dataset_feedback(
            dataset_id=self.phoenix_dataset_id,
            feedback=results
        )
```

#### 3.1.4 Scoring Mechanisms
```python
# src/evaluation/inspect_tasks/scorers.py
class VideoRetrievalScorer:
    """Comprehensive scorer for video retrieval tasks"""
    
    def __init__(self, metrics: List[str]):
        self.metrics = metrics
        self.metric_calculators = {
            "mrr": self._calculate_mrr,
            "ndcg": self._calculate_ndcg,
            "precision": self._calculate_precision,
            "recall": self._calculate_recall,
            "map": self._calculate_map
        }
    
    async def score(self, state: SolverState) -> Score:
        results = state.metadata["retrieval_results"]
        expected = state.target.expected_videos
        
        scores = {}
        for config, search_results in results.items():
            config_scores = {}
            
            for metric in self.metrics:
                if metric in self.metric_calculators:
                    config_scores[metric] = self.metric_calculators[metric](
                        search_results, expected
                    )
            
            scores[config] = config_scores
        
        # Log to Phoenix
        with phoenix_trace("scoring") as span:
            span.set_attributes({
                "scores": json.dumps(scores),
                "query": state.input.text
            })
        
        return Score(
            value=scores,
            explanation=self._generate_explanation(scores)
        )
```

### 3.2 Arize Phoenix Integration

#### 3.2.1 Instrumentation Layer
```python
# src/evaluation/phoenix/instrumentation.py
from opentelemetry import trace
from opentelemetry.instrumentation import BaseInstrumentor
import phoenix.trace

class CogniverseInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentation for Cogniverse components"""
    
    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = trace.get_tracer(__name__, "1.0.0", tracer_provider)
        
        # Instrument video pipeline
        self._instrument_video_pipeline(tracer)
        
        # Instrument search backends
        self._instrument_search_backends(tracer)
        
        # Instrument query encoders
        self._instrument_query_encoders(tracer)
        
        # Instrument agents
        self._instrument_agents(tracer)
    
    def _instrument_video_pipeline(self, tracer):
        """Add tracing to video processing pipeline"""
        from src.processing.unified_video_pipeline import UnifiedVideoPipeline
        
        original_process = UnifiedVideoPipeline.process_video
        
        def traced_process(self, video_path, *args, **kwargs):
            with tracer.start_as_current_span(
                "video_pipeline.process",
                attributes={
                    "video_path": str(video_path),
                    "profile": kwargs.get("profile", "default")
                }
            ) as span:
                try:
                    result = original_process(self, video_path, *args, **kwargs)
                    span.set_attribute("frames_extracted", len(result.get("frames", [])))
                    span.set_status(StatusCode.OK)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(StatusCode.ERROR)
                    raise
        
        UnifiedVideoPipeline.process_video = traced_process
    
    def _instrument_search_backends(self, tracer):
        """Add tracing to search operations"""
        from src.search.vespa_search_backend import VespaSearchBackend
        
        original_search = VespaSearchBackend.search
        
        def traced_search(self, *args, **kwargs):
            with tracer.start_as_current_span(
                "search.execute",
                attributes={
                    "backend": "vespa",
                    "query": kwargs.get("query_text", ""),
                    "strategy": kwargs.get("ranking_strategy", "default"),
                    "top_k": kwargs.get("top_k", 10)
                }
            ) as span:
                # Add query embeddings info
                if "query_embeddings" in kwargs and kwargs["query_embeddings"] is not None:
                    span.set_attribute("has_embeddings", True)
                    span.set_attribute("embedding_shape", str(kwargs["query_embeddings"].shape))
                
                result = original_search(self, *args, **kwargs)
                
                span.set_attribute("num_results", len(result))
                span.set_attribute("top_score", result[0].score if result else 0)
                
                return result
        
        VespaSearchBackend.search = traced_search
```

#### 3.2.2 Experiment Tracking
```python
# src/evaluation/phoenix/experiments.py
import phoenix as px
from phoenix.experiments import ExperimentRunner

class CogniverseExperimentRunner:
    """Manage experiments for different configurations"""
    
    def __init__(self):
        self.phoenix_client = px.Client()
    
    async def run_retrieval_experiment(
        self,
        name: str,
        profiles: List[str],
        strategies: List[str],
        dataset: VideoRetrievalDataset,
        iterations: int = 1
    ):
        """Run comprehensive retrieval experiment"""
        
        experiment = self.phoenix_client.create_experiment(
            name=name,
            description=f"Testing {len(profiles)} profiles with {len(strategies)} strategies",
            metadata={
                "profiles": profiles,
                "strategies": strategies,
                "dataset": dataset.dataset_path
            }
        )
        
        with experiment:
            for iteration in range(iterations):
                for profile in profiles:
                    for strategy in strategies:
                        run_id = f"{profile}_{strategy}_iter{iteration}"
                        
                        with experiment.run(run_id) as run:
                            # Execute evaluation
                            results = await self._evaluate_configuration(
                                profile, strategy, dataset
                            )
                            
                            # Log metrics
                            run.log_metrics({
                                "mrr": results["mrr"],
                                "ndcg@5": results["ndcg_5"],
                                "precision@10": results["precision_10"],
                                "avg_latency_ms": results["avg_latency"]
                            })
                            
                            # Log parameters
                            run.log_params({
                                "profile": profile,
                                "strategy": strategy,
                                "iteration": iteration
                            })
                            
                            # Log artifacts
                            run.log_artifact(
                                "detailed_results.json",
                                results["detailed"]
                            )
        
        return experiment.id
    
    def compare_experiments(self, experiment_ids: List[str]):
        """Compare multiple experiments"""
        comparison = self.phoenix_client.compare_experiments(
            experiment_ids,
            metrics=["mrr", "ndcg@5", "avg_latency_ms"]
        )
        
        return comparison.to_dataframe()
```

#### 3.2.3 Real-time Monitoring Dashboard
```python
# src/evaluation/phoenix/monitoring.py
class RetrievalMonitor:
    """Real-time monitoring of retrieval performance"""
    
    def __init__(self):
        self.phoenix_session = px.launch_app()
        self.metrics_buffer = []
        self.alert_thresholds = {
            "latency_p95_ms": 1000,
            "error_rate": 0.05,
            "mrr_drop": 0.1
        }
    
    def setup_monitors(self):
        """Setup real-time monitors"""
        # Latency monitor
        px.add_monitor(
            name="retrieval_latency",
            query="avg(duration) by (profile, strategy)",
            window="5m",
            alert_threshold=self.alert_thresholds["latency_p95_ms"]
        )
        
        # Error rate monitor
        px.add_monitor(
            name="retrieval_errors",
            query="rate(errors) by (profile)",
            window="5m",
            alert_threshold=self.alert_thresholds["error_rate"]
        )
        
        # Quality degradation monitor
        px.add_monitor(
            name="retrieval_quality",
            query="avg(mrr) by (profile)",
            window="1h",
            alert_on_decrease=self.alert_thresholds["mrr_drop"]
        )
    
    def log_retrieval_event(self, event: Dict[str, Any]):
        """Log individual retrieval event"""
        px.log_trace(
            name="retrieval",
            inputs={"query": event["query"]},
            outputs={"results": event["results"]},
            metadata={
                "profile": event["profile"],
                "strategy": event["strategy"],
                "latency_ms": event["latency_ms"]
            }
        )
```

### 3.3 Unified Evaluation Pipeline

#### 3.3.1 Pipeline Orchestrator
```python
# src/evaluation/pipeline/orchestrator.py
class EvaluationPipeline:
    """Main evaluation pipeline orchestrating Inspect and Phoenix"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.inspect_runner = InspectRunner()
        self.phoenix_runner = CogniverseExperimentRunner()
        self.monitor = RetrievalMonitor()
    
    async def run_comprehensive_evaluation(
        self,
        evaluation_name: str,
        profiles: List[str],
        strategies: List[str],
        tasks: List[str] = ["retrieval_accuracy", "temporal", "multimodal"]
    ):
        """Run comprehensive evaluation pipeline"""
        
        # Start Phoenix monitoring
        self.monitor.setup_monitors()
        
        # Create experiment
        experiment_id = await self.phoenix_runner.run_retrieval_experiment(
            name=evaluation_name,
            profiles=profiles,
            strategies=strategies,
            dataset=self._get_dataset()
        )
        
        # Run Inspect AI tasks
        inspect_results = {}
        for task_name in tasks:
            task = self._get_inspect_task(task_name)
            results = await self.inspect_runner.run(
                task,
                model="cogniverse",  # Custom model wrapper
                config={
                    "profiles": profiles,
                    "strategies": strategies
                }
            )
            inspect_results[task_name] = results
        
        # Generate comprehensive report
        report = self._generate_report(
            experiment_id,
            inspect_results
        )
        
        return report
    
    def _generate_report(self, experiment_id: str, inspect_results: Dict):
        """Generate comprehensive evaluation report"""
        report = EvaluationReport()
        
        # Add Phoenix experiment results
        experiment_data = self.phoenix_runner.get_experiment_results(experiment_id)
        report.add_experiment_section(experiment_data)
        
        # Add Inspect AI results
        for task, results in inspect_results.items():
            report.add_task_section(task, results)
        
        # Add performance analysis
        report.add_performance_analysis(
            self._analyze_performance(experiment_data, inspect_results)
        )
        
        # Generate visualizations
        report.generate_visualizations()
        
        return report
```

#### 3.3.2 Evaluation Configuration
```yaml
# configs/evaluation/eval_config.yaml
evaluation:
  name: "Cogniverse Video RAG Evaluation v1"
  
  datasets:
    - name: "video_queries_v1"
      path: "data/eval/video_queries.json"
      type: "retrieval"
      phoenix_dataset: true
    
    - name: "temporal_queries_v1"
      path: "data/eval/temporal_queries.json"
      type: "temporal"
    
    - name: "multimodal_alignment_v1"
      path: "data/eval/multimodal_pairs.json"
      type: "alignment"
  
  profiles:
    - frame_based_colpali
    - direct_video_global
    - direct_video_global_large
  
  strategies:
    - float_float
    - binary_binary
    - hybrid_binary_bm25
    - phased
  
  metrics:
    retrieval:
      - mrr
      - ndcg@[1, 5, 10]
      - precision@[1, 5, 10]
      - recall@[1, 5, 10]
    
    performance:
      - latency_p50
      - latency_p95
      - throughput_qps
      - memory_usage_mb
  
  inspect_tasks:
    - video_retrieval_accuracy
    - temporal_understanding
    - multimodal_alignment
    - failure_analysis
  
  phoenix:
    instrumentation:
      - video_pipeline
      - search_backends
      - query_encoders
      - agents
    
    monitors:
      - latency
      - errors
      - quality
    
    alerts:
      latency_threshold_ms: 1000
      error_rate_threshold: 0.05
      quality_drop_threshold: 0.1
```

### 3.4 Integration with Existing Cogniverse Components

#### 3.4.1 Search Service Integration
```python
# src/search/search_service.py (modifications)
class SearchService:
    def __init__(self):
        # Existing initialization
        self.backend = self._init_backend()
        
        # Add evaluation hooks
        self.evaluation_enabled = False
        self.phoenix_tracer = None
    
    def enable_evaluation(self):
        """Enable evaluation mode with tracing"""
        self.evaluation_enabled = True
        self.phoenix_tracer = phoenix.trace.tracer("search_service")
    
    async def search(self, query: str, **kwargs):
        if self.evaluation_enabled:
            with self.phoenix_tracer.start_span(
                "search_request",
                attributes={
                    "query": query,
                    **kwargs
                }
            ) as span:
                start_time = time.time()
                results = await self._execute_search(query, **kwargs)
                
                span.set_attribute("latency_ms", (time.time() - start_time) * 1000)
                span.set_attribute("num_results", len(results))
                
                # Log to Phoenix dataset if in experiment
                if phoenix.active_experiment():
                    phoenix.log_example(
                        inputs={"query": query},
                        outputs={"results": [r.to_dict() for r in results]},
                        metadata=kwargs
                    )
                
                return results
        else:
            return await self._execute_search(query, **kwargs)
```

#### 3.4.2 Agent Integration
```python
# src/agents/video_agent.py (modifications)
class VideoAgent:
    def __init__(self):
        # Existing initialization
        self.search_service = SearchService()
        
        # Add Inspect AI compatibility
        self.inspect_compatible = True
    
    async def process_query(self, query: str, context: Dict = None):
        """Process query with evaluation support"""
        
        # If called from Inspect AI
        if context and context.get("inspect_ai"):
            return await self._process_for_inspect(query, context)
        
        # Normal processing with optional tracing
        with optional_trace("video_agent.process") as span:
            # Existing processing logic
            results = await self._process_query_internal(query)
            
            if span:
                span.set_attribute("query_type", self._classify_query(query))
                span.set_attribute("num_results", len(results))
            
            return results
    
    async def _process_for_inspect(self, query: str, context: Dict):
        """Special processing for Inspect AI evaluation"""
        # Use context to determine evaluation parameters
        profile = context.get("profile", self.default_profile)
        strategy = context.get("strategy", "default")
        
        # Execute with specific configuration
        results = await self.search_service.search(
            query=query,
            profile=profile,
            ranking_strategy=strategy
        )
        
        # Return in Inspect-compatible format
        return {
            "results": results,
            "metadata": {
                "profile": profile,
                "strategy": strategy,
                "agent": "video_agent"
            }
        }
```

## 4. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
1. **Setup Infrastructure**
   - Install Inspect AI and Arize Phoenix
   - Configure OpenTelemetry providers
   - Setup Phoenix dashboard

2. **Basic Instrumentation**
   - Instrument search backends
   - Add tracing to video pipeline
   - Create initial Phoenix monitors

3. **Dataset Preparation**
   - Convert existing test queries to Inspect format
   - Create Phoenix datasets
   - Version control evaluation data

### Phase 2: Core Integration (Week 3-4)
1. **Inspect AI Tasks**
   - Implement video retrieval tasks
   - Create custom solvers
   - Build scoring mechanisms

2. **Phoenix Experiments**
   - Create experiment runner
   - Implement metric collection
   - Setup A/B testing framework

3. **Pipeline Integration**
   - Connect Inspect and Phoenix
   - Create unified evaluation API
   - Build configuration system

### Phase 3: Advanced Features (Week 5-6)
1. **Advanced Evaluation Tasks**
   - Temporal understanding evaluation
   - Multimodal alignment testing
   - Failure analysis tasks

2. **Real-time Monitoring**
   - Production monitoring setup
   - Alert configuration
   - Performance dashboards

3. **Reporting System**
   - Automated report generation
   - Comparative analysis tools
   - Visualization suite

### Phase 4: Production Deployment (Week 7-8)
1. **CI/CD Integration**
   - Automated evaluation on commits
   - Performance regression detection
   - Quality gates

2. **Documentation**
   - User guides
   - API documentation
   - Best practices

3. **Training & Rollout**
   - Team training sessions
   - Gradual rollout plan
   - Support structure

## 5. Success Metrics

### Technical Metrics
1. **Coverage**
   - 100% of search operations instrumented
   - All ranking strategies evaluated
   - Complete profile coverage

2. **Performance**
   - <5% overhead from instrumentation
   - <100ms additional latency for evaluation
   - Real-time metric updates

3. **Quality**
   - Automated detection of >90% of regressions
   - <1% false positive rate on alerts
   - Comprehensive metric tracking

### Business Metrics
1. **Development Velocity**
   - 50% reduction in debugging time
   - 75% faster performance optimization
   - 90% reduction in production issues

2. **System Reliability**
   - 99.9% uptime for evaluation system
   - <5 minute detection time for issues
   - Complete audit trail

## 6. Risk Mitigation

### Technical Risks
1. **Performance Impact**
   - Risk: Instrumentation overhead
   - Mitigation: Sampling strategies, async collection

2. **Data Volume**
   - Risk: Large trace/metric storage
   - Mitigation: Retention policies, aggregation

3. **Integration Complexity**
   - Risk: Complex tool interactions
   - Mitigation: Modular design, thorough testing

### Operational Risks
1. **Team Adoption**
   - Risk: Learning curve for new tools
   - Mitigation: Training, documentation, gradual rollout

2. **Maintenance Burden**
   - Risk: Additional system to maintain
   - Mitigation: Automation, clear ownership

## 7. Appendices

### A. Tool Comparison Matrix
| Feature | Current System | Inspect AI | Arize Phoenix | Combined |
|---------|---------------|------------|---------------|----------|
| Structured Tasks | ❌ | ✅ | ❌ | ✅ |
| Tracing | ❌ | ❌ | ✅ | ✅ |
| Experiments | ❌ | ✅ | ✅ | ✅ |
| Real-time Monitoring | ❌ | ❌ | ✅ | ✅ |
| Custom Metrics | ✅ | ✅ | ✅ | ✅ |
| Report Generation | ✅ | ✅ | ✅ | ✅✅ |

### B. Sample Evaluation Report Structure
```
Cogniverse Evaluation Report
Generated: 2024-01-15

1. Executive Summary
   - Overall MRR: 0.89
   - Best Profile: direct_video_global
   - Best Strategy: hybrid_binary_bm25

2. Detailed Results
   2.1 By Profile
   2.2 By Strategy
   2.3 By Query Type

3. Performance Analysis
   3.1 Latency Distribution
   3.2 Throughput Metrics
   3.3 Resource Usage

4. Failure Analysis
   4.1 Failed Queries
   4.2 Root Causes
   4.3 Recommendations

5. Comparative Analysis
   5.1 Profile Comparison
   5.2 Strategy Effectiveness
   5.3 Trend Analysis

6. Appendices
   6.1 Raw Data
   6.2 Methodology
   6.3 Configuration
```

### C. Configuration Examples
```python
# Example: Running evaluation
evaluator = EvaluationPipeline("configs/evaluation/eval_config.yaml")

results = await evaluator.run_comprehensive_evaluation(
    evaluation_name="Q1_2024_Video_RAG_Eval",
    profiles=["frame_based_colpali", "direct_video_global"],
    strategies=["float_float", "hybrid_binary_bm25"],
    tasks=["retrieval_accuracy", "temporal_understanding"]
)

# Generate report
report = results.generate_html_report()
report.save("reports/q1_2024_evaluation.html")
```

This PRD provides a comprehensive blueprint for integrating Inspect AI and Arize Phoenix into Cogniverse, creating a robust evaluation framework that enables systematic testing, monitoring, and optimization of the video RAG system.