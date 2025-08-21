#!/usr/bin/env python
"""
Comprehensive test suite for the new routing system.
Tests all tiers, strategies, and optimization capabilities.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import sys
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.routing import (
    ComprehensiveRouter,
    TieredRouter,
    RoutingConfig,
    GLiNERRoutingStrategy,
    LLMRoutingStrategy,
    KeywordRoutingStrategy,
    HybridRoutingStrategy,
    EnsembleRoutingStrategy,
    AutoTuningOptimizer,
    load_config
)
from src.routing.base import SearchModality, GenerationType, RoutingDecision
# from src.agents.query_analysis_tool_v2 import QueryAnalysisToolV2  # Commented out - requires google.adk


# Test queries with expected routing
TEST_QUERIES = [
    # Video queries
    {
        "query": "Show me the tutorial video on Python programming",
        "expected_modality": "video",
        "expected_generation": "raw_results",
        "category": "video_explicit"
    },
    {
        "query": "Find clips from yesterday's presentation",
        "expected_modality": "video",
        "expected_generation": "raw_results",
        "category": "video_temporal"
    },
    
    # Text queries
    {
        "query": "Find the research paper on quantum computing",
        "expected_modality": "text",
        "expected_generation": "raw_results",
        "category": "text_explicit"
    },
    {
        "query": "Create a detailed report on AI advancements in 2024",
        "expected_modality": "text",
        "expected_generation": "detailed_report",
        "category": "text_report"
    },
    
    # Summary queries
    {
        "query": "Summarize the main points from the TED talk",
        "expected_modality": "video",
        "expected_generation": "summary",
        "category": "video_summary"
    },
    {
        "query": "Give me a brief overview of the climate change article",
        "expected_modality": "text",
        "expected_generation": "summary",
        "category": "text_summary"
    },
    
    # Both modalities
    {
        "query": "Search for content about machine learning from last month",
        "expected_modality": "both",
        "expected_generation": "raw_results",
        "category": "both_temporal"
    },
    {
        "query": "Find information about solar panels",
        "expected_modality": "both",
        "expected_generation": "raw_results",
        "category": "both_general"
    },
    
    # Complex queries
    {
        "query": "I'm frustrated with the service, show me how to file a complaint and compare alternative providers",
        "expected_modality": "both",
        "expected_generation": "raw_results",
        "category": "complex_multi_intent"
    },
    {
        "query": "Analyze the performance metrics from Q3 and create a comprehensive report with visual charts",
        "expected_modality": "both",
        "expected_generation": "detailed_report",
        "category": "complex_analysis"
    }
]


class ComprehensiveRoutingTester:
    """Test harness for the comprehensive routing system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the test harness."""
        self.config = load_config(config_path)
        self.results = []
        self.tier_stats = {"fast_path": 0, "slow_path": 0, "fallback": 0}
    
    async def test_individual_strategies(self):
        """Test each routing strategy individually."""
        print("\n" + "="*80)
        print("TESTING INDIVIDUAL STRATEGIES")
        print("="*80)
        
        strategies = {
            "GLiNER": GLiNERRoutingStrategy(self.config.gliner_config),
            "LLM": LLMRoutingStrategy(self.config.llm_config),
            "Keyword": KeywordRoutingStrategy(self.config.keyword_config)
        }
        
        for strategy_name, strategy in strategies.items():
            print(f"\n📊 Testing {strategy_name} Strategy:")
            print("-" * 40)
            
            results = []
            total_time = 0
            
            for test_case in TEST_QUERIES[:5]:  # Test subset for speed
                start_time = time.time()
                
                try:
                    decision = await strategy.route(test_case["query"])
                    execution_time = time.time() - start_time
                    
                    # Evaluate accuracy
                    modality_correct = decision.search_modality.value == test_case["expected_modality"]
                    generation_correct = decision.generation_type.value == test_case["expected_generation"]
                    
                    results.append({
                        "query": test_case["query"][:50] + "...",
                        "modality_correct": modality_correct,
                        "generation_correct": generation_correct,
                        "confidence": decision.confidence_score,
                        "time": execution_time
                    })
                    
                    total_time += execution_time
                    
                    # Print result
                    status = "✅" if modality_correct and generation_correct else "❌"
                    print(f"  {status} Query: {test_case['query'][:40]}...")
                    print(f"     Confidence: {decision.confidence_score:.2f}, Time: {execution_time:.3f}s")
                    
                except Exception as e:
                    print(f"  ❌ Error: {str(e)[:50]}")
                    results.append({
                        "query": test_case["query"][:50] + "...",
                        "modality_correct": False,
                        "generation_correct": False,
                        "confidence": 0,
                        "time": time.time() - start_time
                    })
            
            # Calculate statistics
            if results:
                accuracy = sum(1 for r in results if r["modality_correct"]) / len(results)
                avg_confidence = sum(r["confidence"] for r in results) / len(results)
                avg_time = total_time / len(results)
                
                print(f"\n  📈 {strategy_name} Statistics:")
                print(f"     Accuracy: {accuracy:.1%}")
                print(f"     Avg Confidence: {avg_confidence:.2f}")
                print(f"     Avg Time: {avg_time:.3f}s")
    
    async def test_tiered_router(self):
        """Test the tiered routing architecture."""
        print("\n" + "="*80)
        print("TESTING TIERED ROUTER")
        print("="*80)
        
        router = TieredRouter(self.config)
        
        print("\n📊 Running queries through tiered architecture:")
        print("-" * 40)
        
        results = []
        
        for test_case in TEST_QUERIES:
            start_time = time.time()
            
            try:
                decision = await router.route(test_case["query"])
                execution_time = time.time() - start_time
                
                # Track tier usage
                tier_used = decision.metadata.get("tier", "unknown")
                if tier_used in self.tier_stats:
                    self.tier_stats[tier_used] += 1
                
                # Evaluate accuracy
                modality_correct = decision.search_modality.value == test_case["expected_modality"]
                generation_correct = decision.generation_type.value == test_case["expected_generation"]
                
                result = {
                    "query": test_case["query"],
                    "category": test_case["category"],
                    "tier": tier_used,
                    "modality_correct": modality_correct,
                    "generation_correct": generation_correct,
                    "confidence": decision.confidence_score,
                    "time": execution_time,
                    "routing_method": decision.routing_method
                }
                results.append(result)
                
                # Print result
                status = "✅" if modality_correct and generation_correct else "❌"
                tier_emoji = {"fast_path": "🚀", "slow_path": "🧠", "fallback": "🔧"}.get(tier_used, "❓")
                
                print(f"  {status} {tier_emoji} [{test_case['category']}] {test_case['query'][:40]}...")
                print(f"     Tier: {tier_used}, Confidence: {decision.confidence_score:.2f}, Time: {execution_time:.3f}s")
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                results.append({
                    "query": test_case["query"],
                    "category": test_case["category"],
                    "tier": "error",
                    "modality_correct": False,
                    "generation_correct": False,
                    "confidence": 0,
                    "time": time.time() - start_time,
                    "routing_method": "error"
                })
        
        # Print tier statistics
        print("\n📊 Tier Usage Distribution:")
        total = sum(self.tier_stats.values())
        for tier, count in self.tier_stats.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {tier}: {count} ({percentage:.1f}%)")
        
        # Print performance statistics
        if results:
            self._print_performance_stats(results)
        
        # Get router statistics
        router_stats = router.get_tier_statistics()
        print("\n📊 Router Internal Statistics:")
        for tier, stats in router_stats.items():
            print(f"  {tier}:")
            print(f"    Success Rate: {stats['success_rate']:.1f}%")
            print(f"    Usage: {stats['usage_percentage']:.1f}%")
        
        return results
    
    async def test_ensemble_router(self):
        """Test the ensemble routing strategy."""
        print("\n" + "="*80)
        print("TESTING ENSEMBLE ROUTER")
        print("="*80)
        
        ensemble_config = {
            "enabled_strategies": ["gliner", "keyword"],  # Exclude LLM for speed
            "voting_method": "weighted",
            "weights": {"gliner": 2.0, "keyword": 1.0}
        }
        
        ensemble = EnsembleRoutingStrategy(ensemble_config)
        
        print("\n📊 Testing ensemble with weighted voting:")
        print("-" * 40)
        
        for test_case in TEST_QUERIES[:5]:
            start_time = time.time()
            
            try:
                decision = await ensemble.route(test_case["query"])
                execution_time = time.time() - start_time
                
                modality_correct = decision.search_modality.value == test_case["expected_modality"]
                
                status = "✅" if modality_correct else "❌"
                print(f"  {status} {test_case['query'][:40]}...")
                print(f"     Confidence: {decision.confidence_score:.2f}, Time: {execution_time:.3f}s")
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
    
    async def test_optimization(self):
        """Test the auto-optimization capabilities."""
        print("\n" + "="*80)
        print("TESTING AUTO-OPTIMIZATION")
        print("="*80)
        
        # Create a strategy with optimizer
        strategy = KeywordRoutingStrategy(self.config.keyword_config)
        optimizer = AutoTuningOptimizer(strategy, self.config.optimization_config)
        
        print("\n📊 Simulating optimization with feedback:")
        print("-" * 40)
        
        # Simulate queries with feedback
        for i, test_case in enumerate(TEST_QUERIES[:10]):
            # Make prediction
            decision = await strategy.route(test_case["query"])
            
            # Create ground truth
            ground_truth = RoutingDecision(
                search_modality=SearchModality(test_case["expected_modality"]),
                generation_type=GenerationType(test_case["expected_generation"]),
                confidence_score=1.0,
                routing_method="ground_truth"
            )
            
            # Track performance
            optimizer.track_performance(
                query=test_case["query"],
                predicted=decision,
                actual=ground_truth,
                user_feedback={"satisfaction": 0.8}
            )
            
            print(f"  📝 Tracked query {i+1}/10")
        
        # Trigger optimization
        print("\n  🔧 Triggering optimization...")
        await optimizer.optimize()
        
        # Show optimization results
        current_metrics = optimizer._calculate_current_metrics()
        print(f"\n  📈 Optimization Results:")
        print(f"     Accuracy: {current_metrics.accuracy:.1%}")
        print(f"     F1 Score: {current_metrics.f1_score:.2f}")
        print(f"     Avg Latency: {current_metrics.avg_latency:.3f}s")
    
    async def test_query_analysis_tool(self):
        """Test the enhanced QueryAnalysisTool."""
        print("\n" + "="*80)
        print("TESTING QUERY ANALYSIS TOOL V2")
        print("="*80)
        
        analyzer = QueryAnalysisToolV2()
        
        print("\n📊 Testing QueryAnalysisToolV2:")
        print("-" * 40)
        
        for test_case in TEST_QUERIES[:5]:
            start_time = time.time()
            
            try:
                analysis = await analyzer.execute(test_case["query"])
                execution_time = time.time() - start_time
                
                # Check results
                video_search = analysis["needs_video_search"]
                text_search = analysis["needs_text_search"]
                
                # Determine actual modality
                if video_search and text_search:
                    actual_modality = "both"
                elif video_search:
                    actual_modality = "video"
                elif text_search:
                    actual_modality = "text"
                else:
                    actual_modality = "none"
                
                modality_correct = actual_modality == test_case["expected_modality"]
                
                status = "✅" if modality_correct else "❌"
                print(f"  {status} {test_case['query'][:40]}...")
                print(f"     Modality: {actual_modality}, Tier: {analysis.get('routing_tier', 'unknown')}")
                print(f"     Confidence: {analysis['confidence_score']:.2f}, Time: {execution_time:.3f}s")
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
        
        # Print statistics
        stats = analyzer.get_statistics()
        print(f"\n  📈 Analyzer Statistics:")
        print(f"     Total Queries: {stats['total_queries']}")
        print(f"     QPM: {stats['queries_per_minute']:.1f}")
    
    def _print_performance_stats(self, results: List[Dict[str, Any]]):
        """Print performance statistics."""
        total = len(results)
        correct_modality = sum(1 for r in results if r["modality_correct"])
        correct_generation = sum(1 for r in results if r["generation_correct"])
        fully_correct = sum(1 for r in results if r["modality_correct"] and r["generation_correct"])
        
        avg_confidence = sum(r["confidence"] for r in results) / total if total > 0 else 0
        avg_time = sum(r["time"] for r in results) / total if total > 0 else 0
        
        print("\n📈 Overall Performance:")
        print(f"  Modality Accuracy: {correct_modality}/{total} ({correct_modality/total*100:.1f}%)")
        print(f"  Generation Accuracy: {correct_generation}/{total} ({correct_generation/total*100:.1f}%)")
        print(f"  Full Accuracy: {fully_correct}/{total} ({fully_correct/total*100:.1f}%)")
        print(f"  Avg Confidence: {avg_confidence:.2f}")
        print(f"  Avg Time: {avg_time:.3f}s")
        
        # Per-category analysis
        categories = {}
        for r in results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "correct": 0}
            categories[cat]["total"] += 1
            if r["modality_correct"]:
                categories[cat]["correct"] += 1
        
        print("\n📊 Per-Category Accuracy:")
        for cat, stats in sorted(categories.items()):
            acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"  {cat}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    def save_results(self, results: List[Dict[str, Any]], filename: str):
        """Save test results to file."""
        output_dir = Path("outputs/routing_tests")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to {filepath}")


async def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Test the comprehensive routing system")
    parser.add_argument("--config", help="Path to routing configuration file")
    parser.add_argument("--test", choices=["all", "strategies", "tiered", "ensemble", "optimization", "tool"],
                       default="all", help="Which tests to run")
    parser.add_argument("--save", action="store_true", help="Save test results")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🧪 COMPREHENSIVE ROUTING SYSTEM TEST SUITE")
    print("="*80)
    
    tester = ComprehensiveRoutingTester(args.config)
    
    results = {}
    
    if args.test in ["all", "strategies"]:
        await tester.test_individual_strategies()
    
    if args.test in ["all", "tiered"]:
        tiered_results = await tester.test_tiered_router()
        results["tiered"] = tiered_results
    
    if args.test in ["all", "ensemble"]:
        await tester.test_ensemble_router()
    
    if args.test in ["all", "optimization"]:
        await tester.test_optimization()
    
    # if args.test in ["all", "tool"]:
    #     await tester.test_query_analysis_tool()  # Requires google.adk
    
    if args.save and results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        tester.save_results(results, f"routing_test_results_{timestamp}.json")
    
    print("\n" + "="*80)
    print("✅ TEST SUITE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())