# DSPy 3.0 Routing Integration Implementation Plan

## Overview

This plan transforms the multi-agent routing system to leverage DSPy 3.0's advanced capabilities while maintaining A2A protocol compatibility. The key innovation is integrating relationship extraction for query enhancement and intelligent routing.

## Architecture Vision

### Current State
```
QueryAnalysisToolV3 → tries to initialize → RoutingAgent (BROKEN)
```

### Target Architecture
```
User Query → RoutingAgent (Entry Point)
              ↓
          DSPy 3.0 Routing Module (Entity + Relationship Extraction)
              ↓ 
          Query Enhancement (relationship tuples integrated)
              ↓ 
          Enhanced Query → A2A Agents → Improved Results
```

## Implementation Phases

### **Phase 1: DSPy 3.0 Foundation (Week 1)**
**Goal**: Establish DSPy 3.0 infrastructure and A2A integration

#### **1.1 Dependencies & Infrastructure**
- [ ] Upgrade pyproject.toml to DSPy 3.0+
- [ ] Add MLflow 3.0 for observability
- [ ] Add spaCy for dependency parsing
- [ ] Test DSPy 3.0 basic functionality
- [ ] Ensure compatibility with existing codebase

#### **1.2 DSPy-A2A Bridge**
- [ ] Create `src/app/agents/dspy_a2a_agent_base.py`
- [ ] Implement A2A protocol endpoint management
- [ ] Add DSPy module integration layer
- [ ] Create conversion utilities (A2A ↔ DSPy)

#### **1.3 Core DSPy Signatures**
- [ ] Create basic routing signatures
- [ ] Implement entity extraction signatures  
- [ ] Add query analysis signatures
- [ ] Test signature compilation and execution

#### **1.4 Unit Tests - Phase 1**
- [ ] Create `tests/routing/unit/test_dspy_a2a_base.py`
- [ ] Test DSPy-A2A conversion utilities
- [ ] Test basic signature functionality
- [ ] Test A2A endpoint management
- [ ] Test error handling and fallbacks

**Deliverables**:
- DSPy 3.0 properly integrated
- DSPyA2AAgentBase class functional
- Basic signatures working with A2A protocol
- **Phase 1 unit tests passing**

---

### **Phase 2: Relationship Extraction Engine (Week 2)**
**Goal**: Build intelligent relationship extraction capabilities

#### **2.1 Relationship Extraction Signatures**
- [ ] Create `RelationshipExtractionSignature` with comprehensive outputs
- [ ] Design multi-modal relationship signatures for visual queries
- [ ] Implement confidence scoring for relationships
- [ ] Add temporal relationship extraction

#### **2.2 Tool Integration**
- [ ] Integrate GLiNER as DSPy tool (`dspy.ToolCalls`)
- [ ] Add spaCy dependency parsing tool
- [ ] Create visual relationship extraction tools
- [ ] Implement tool result processing

#### **2.3 DSPyRelationshipRouter Module**
- [ ] Create `src/app/routing/dspy_relationship_router.py`
- [ ] Implement entity extraction with GLiNER
- [ ] Add dependency parsing for relationships
- [ ] Integrate DSPy 3.0 History for context
- [ ] Add multi-modal analysis capabilities

#### **2.4 Unit Tests - Phase 2**
- [ ] Create `tests/routing/unit/test_relationship_extraction.py`
- [ ] Test RelationshipExtractionSignature with mock data
- [ ] Test GLiNER tool integration
- [ ] Test spaCy dependency parsing integration
- [ ] Test relationship tuple generation accuracy
- [ ] Test multi-modal relationship extraction
- [ ] Create `tests/routing/unit/test_dspy_relationship_router.py`
- [ ] Test DSPyRelationshipRouter end-to-end functionality

**Deliverables**:
- Relationship extraction working end-to-end
- Entity recognition enhanced with relationships
- Tool integrations functional
- **Phase 2 unit tests passing**

---

### **Phase 3: Query Enhancement System (Week 2-3)**
**Goal**: Transform extracted relationships into enhanced queries

#### **3.1 Query Enhancement Signatures**
- [ ] Create `QueryEnhancementSignature` for query rewriting
- [ ] Design relationship tuple integration patterns
- [ ] Add semantic expansion capabilities
- [ ] Implement search strategy optimization

#### **3.2 Query Enhancement Engine**
- [ ] Create `QueryEnhancer` DSPy module
- [ ] Implement relationship tuple → query enhancement logic
- [ ] Add semantic expansion using relationships
- [ ] Create query customization for different agents

#### **3.3 Relationship-Driven Rewriting**
- [ ] Transform relationships into natural language expansions
- [ ] Add domain-specific query enhancement patterns
- [ ] Implement query validation and quality scoring
- [ ] Create fallback mechanisms for enhancement failures

#### **3.4 Unit Tests - Phase 3**
- [ ] Create `tests/routing/unit/test_query_enhancement.py`
- [ ] Test QueryEnhancementSignature with various relationship patterns
- [ ] Test relationship tuple → query expansion logic
- [ ] Test agent-specific query customization
- [ ] Test semantic expansion quality
- [ ] Create `tests/routing/unit/test_query_enhancer.py`
- [ ] Test QueryEnhancer module end-to-end

#### **3.5 Integration Tests - Phases 1-3**
- [ ] Create `tests/routing/integration/test_relationship_to_enhancement_flow.py`
- [ ] Test complete flow: Raw Query → Relationships → Enhanced Query
- [ ] Test with real-world query examples
- [ ] Test error handling and fallback scenarios
- [ ] Validate query enhancement quality improvements
- [ ] Test A2A protocol compatibility throughout the flow

**Examples**:
```
Original: "robots playing soccer"
Relationships: [(robots, playing, soccer), (robots, involved_in, sport)]
Enhanced: "robots playing soccer OR robotic soccer OR autonomous robots in sports"
```

**Deliverables**:
- Query enhancement working with relationship context
- Semantic expansion improving retrieval
- Agent-specific query customization
- **Phase 3 unit tests passing**
- **Integration tests for Phases 1-3 passing**

---

### **Phase 4: Enhanced Routing Agent (Week 3-4)**
**Goal**: Create the unified RoutingAgent with full orchestration

#### **4.1 Enhanced RoutingAgent**
- [ ] Create `src/app/agents/enhanced_routing_agent.py`
- [ ] Integrate DSPyRelationshipRouter and QueryEnhancer
- [ ] Implement complete routing flow
- [ ] Add A2A agent registry management

#### **4.2 Multi-Agent Orchestration**
- [ ] Implement sequential agent execution
- [ ] Add parallel execution using DSPy 3.0 batch processing
- [ ] Create hybrid execution strategies
- [ ] Add result aggregation and coordination

#### **4.3 A2A Protocol Integration**
- [ ] Ensure seamless A2A communication
- [ ] Add enhanced query parameter passing
- [ ] Implement relationship context forwarding
- [ ] Create error handling and fallback mechanisms

#### **4.4 Workflow Intelligence**
- [ ] Add dynamic agent selection logic
- [ ] Implement workflow optimization
- [ ] Create success criteria evaluation
- [ ] Add performance tracking

#### **4.5 Unit Tests - Phase 4**
- [ ] Create `tests/agents/unit/test_enhanced_routing_agent.py`
- [ ] Test EnhancedRoutingAgent A2A protocol compliance
- [ ] Test multi-agent orchestration (sequential/parallel)
- [ ] Test agent registry management
- [ ] Test workflow intelligence and optimization
- [ ] Test error handling and fallback mechanisms

**Deliverables**:
- Complete RoutingAgent as A2A entry point
- Multi-agent orchestration functional
- Enhanced queries flowing to all agents
- **Phase 4 unit tests passing**

---

### **Phase 5: Agent Enhancement (Week 4-5)**
**Goal**: Update existing agents to leverage enhanced queries

#### **5.1 VideoSearchAgent Enhancement**
- [ ] Update to process enhanced queries with relationships
- [ ] Add relationship relevance scoring to results
- [ ] Implement visual relationship matching
- [ ] Create relationship-aware result ranking

#### **5.2 Other Agent Updates**
- [ ] Enhance SummarizerAgent with relationship context
- [ ] Update DetailedReportAgent for relationship analysis
- [ ] Add relationship information to agent responses
- [ ] Create relationship visualization in results

#### **5.3 Result Enhancement**
- [ ] Add relationship relevance scoring
- [ ] Implement cross-agent result correlation
- [ ] Create relationship-based result clustering
- [ ] Add explanation of relationship matching

#### **5.4 Unit Tests - Phase 5**
- [ ] Create `tests/agents/unit/test_enhanced_video_search_agent.py`
- [ ] Test enhanced query processing in VideoSearchAgent
- [ ] Test relationship relevance scoring
- [ ] Create `tests/agents/unit/test_enhanced_summarizer_agent.py`
- [ ] Test relationship context integration in SummarizerAgent
- [ ] Create `tests/agents/unit/test_enhanced_detailed_report_agent.py`
- [ ] Test relationship analysis in DetailedReportAgent

#### **5.5 Integration Tests - Phases 4-5**
- [ ] Create `tests/agents/integration/test_end_to_end_enhanced_flow.py`
- [ ] Test complete flow: Query → RoutingAgent → Enhanced Agents → Results
- [ ] Test with multiple agent types (video, summarizer, report)
- [ ] Test relationship context preservation across agents
- [ ] Test result aggregation and correlation
- [ ] Validate query enhancement impact on result quality

**Deliverables**:
- All agents utilizing enhanced queries
- Relationship context improving result quality
- Cross-agent result correlation
- **Phase 5 unit tests passing**
- **Integration tests for Phases 4-5 passing**

---

### **Phase 6: Advanced Optimization (Week 5-6)**
**Goal**: Implement self-learning and optimization

#### **6.1 GRPO (Reinforcement Learning) Integration**
- [ ] Create routing RL environment
- [ ] Implement reward functions for routing decisions
- [ ] Add GRPO optimizer for routing strategies
- [ ] Create continuous learning pipeline

#### **6.2 SIMBA (Feedback Learning)**
- [ ] Implement user feedback collection
- [ ] Add SIMBA optimizer for query enhancement
- [ ] Create feedback-driven improvement loop
- [ ] Add satisfaction score tracking

#### **6.3 Adaptive Threshold Learning**
- [ ] Create `AdaptiveThresholdSignature`
- [ ] Implement dynamic confidence threshold learning
- [ ] Add performance-based threshold adjustment
- [ ] Create escalation strategy optimization

#### **6.4 MLflow Integration**
- [ ] Add comprehensive observability
- [ ] Implement experiment tracking
- [ ] Create optimization metrics dashboards
- [ ] Add model versioning and deployment

#### **6.5 Unit Tests - Phase 6**
- [ ] Create `tests/routing/unit/test_grpo_optimization.py`
- [ ] Test RL environment and reward functions
- [ ] Create `tests/routing/unit/test_simba_learning.py`
- [ ] Test feedback collection and learning loops
- [ ] Create `tests/routing/unit/test_adaptive_thresholds.py`
- [ ] Test dynamic threshold adjustment
- [ ] Create `tests/routing/unit/test_mlflow_integration.py`
- [ ] Test observability and experiment tracking

**Deliverables**:
- Self-optimizing routing system
- Continuous learning from user feedback
- Advanced observability and monitoring
- **Phase 6 unit tests passing**

---

### **Phase 7: Testing & Validation (Week 6-7)**
**Goal**: Comprehensive testing and performance validation

#### **7.1 Comprehensive Integration Tests**
- [ ] Create `tests/routing/integration/test_full_system_integration.py`
- [ ] Test complete end-to-end flow with real queries
- [ ] Test all agent combinations and orchestration modes
- [ ] Test error handling and recovery scenarios
- [ ] Test performance under load

#### **7.2 Real-World Testing & Validation**
- [ ] Create test dataset with diverse query types
- [ ] Test relationship extraction accuracy with ground truth
- [ ] Evaluate query enhancement effectiveness (A/B testing)
- [ ] Benchmark routing decision quality against baseline
- [ ] Test multi-modal query processing

#### **7.3 Performance Benchmarking**
- [ ] Create `tests/routing/performance/test_performance_benchmarks.py`
- [ ] Compare latency: DSPy 3.0 system vs current system
- [ ] Measure throughput improvements
- [ ] Evaluate memory usage and resource consumption
- [ ] Create performance regression test suite

#### **7.4 Test Quality & Coverage**
- [ ] Ensure 90%+ unit test coverage for all new modules
- [ ] Validate all integration tests pass consistently
- [ ] Run full linting (make lint) on all new code
- [ ] Ensure all pytest markers are properly applied
- [ ] Validate A2A protocol compliance across all agents

#### **7.5 User Experience Validation**
- [ ] Create user testing scenarios with relationship-enhanced queries
- [ ] Measure query result relevance improvements
- [ ] Validate relationship context improves user satisfaction
- [ ] Document performance improvements and trade-offs

**Testing Standards**:
- **All tests must pass locally before any commits**
- **Use JAX_PLATFORM_NAME=cpu uv run python -m pytest for test execution**
- **Never mark tasks as completed if tests are failing**
- **All new code must pass linting (make lint) checks**
- **Integration tests must be properly mocked for CI/CD**

**Deliverables**:
- 90%+ test coverage for all new components
- Performance validation showing improvements
- User experience improvements documented
- **All Phase 7 tests passing**
- **System ready for production deployment**

---

## Key Benefits Expected

### **1. Routing Intelligence**
- **20%+ improvement** in routing accuracy through DSPy optimization
- **Context-aware routing** using conversation history and relationships
- **Dynamic strategy selection** based on query complexity

### **2. Query Enhancement**
- **30-50% improvement** in retrieval relevance through relationship integration
- **Semantic expansion** capturing related concepts
- **Agent-specific optimization** for different search modalities

### **3. System Performance**
- **Parallel execution** of agents using DSPy 3.0 batch processing
- **Intelligent caching** based on learned query patterns
- **Adaptive optimization** through continuous learning

### **4. User Experience**
- **Better search results** through relationship-aware queries
- **Faster responses** through optimized routing
- **Continuous improvement** through feedback learning

## Risk Mitigation

### **Technical Risks**
- **DSPy 3.0 Compatibility**: Thorough testing with current system
- **Performance Impact**: Benchmark at each phase
- **A2A Integration**: Maintain backward compatibility

### **Implementation Risks**
- **Complexity Management**: Phase-based implementation
- **Testing Coverage**: Comprehensive test suite development
- **User Impact**: Gradual rollout with fallback mechanisms

## Success Metrics

### **Quantitative Metrics**
- Routing accuracy improvement: Target 20%+
- Query enhancement effectiveness: Target 30%+ relevance improvement
- System latency: Maintain or improve current performance
- Test coverage: 90%+ for new components

### **Qualitative Metrics**
- User satisfaction with search results
- Relationship extraction accuracy and usefulness
- System reliability and robustness
- Developer experience with new architecture

---

This plan transforms the routing system from a static decision tree to an intelligent, learning orchestration system powered by DSPy 3.0's advanced capabilities while maintaining perfect compatibility with the A2A agent protocol.