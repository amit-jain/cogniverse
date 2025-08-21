#!/usr/bin/env python3
"""
Unified routing demonstration script.
Combines features from both demo_routing_tiers.py and demo_routing_escalation.py.
Supports both concise and verbose output modes.
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from src.app.routing import TieredRouter
from src.app.routing.router import RoutingTier


class RouterWithTracking(TieredRouter):
    """Extended router that tracks escalation path."""
    
    def __init__(self, config, verbose=False):
        super().__init__(config)
        self.verbose = verbose
        self.escalation_paths = []
        self.tier_stats = {
            "TIER_1_GLINER": 0,
            "TIER_2_LLM": 0,
            "TIER_3_LANGEXTRACT": 0,
            "TIER_4_KEYWORD": 0
        }
    
    async def route(self, query: str, context: Dict[str, Any] = None) -> Any:
        """Route with optional detailed tracking."""
        import time
        
        if not query.strip():
            if self.verbose:
                print("   [Empty query - direct to fallback]")
            return await super().route(query, context)
        
        start_time = time.time()
        context = context or {}
        
        path = {
            "query": query,
            "tiers_tried": [],
            "final_tier": None,
            "final_confidence": 0.0
        }
        
        # Check cache
        cached = self._check_cache(query)
        if cached:
            if self.verbose:
                print(f"   📦 CACHE HIT (confidence: {cached.confidence_score:.3f})")
            return cached
        
        # Get thresholds
        if isinstance(self.config, dict):
            fast_threshold = self.config.get("tier_config", {}).get("fast_path_confidence_threshold", 0.7)
            slow_threshold = self.config.get("tier_config", {}).get("slow_path_confidence_threshold", 0.6)
            langextract_threshold = self.config.get("tier_config", {}).get("langextract_confidence_threshold", 0.5)
        else:
            fast_threshold = 0.7
            slow_threshold = 0.6
            langextract_threshold = 0.5
        
        decision = None
        
        # Tier 1: GLiNER
        if RoutingTier.FAST_PATH in self.strategies:
            if self.verbose:
                print(f"\n🚀 Tier 1 (GLiNER Fast Path) - Threshold: {fast_threshold}")
            
            fast_decision = await self._try_fast_path(query, context)
            if fast_decision:
                conf = fast_decision.confidence_score
                passed = conf >= fast_threshold
                status = "✅" if passed else "❌"
                
                if self.verbose:
                    print(f"   Confidence: {conf:.3f} {status}")
                    print(f"     → Modality: {fast_decision.search_modality.value} | Generation: {fast_decision.generation_type.value}")
                    if fast_decision.reasoning:
                        print(f"     → Reasoning: {fast_decision.reasoning[:80]}")
                else:
                    print(f"   T1 GLiNER:      {conf:.3f} {status} (threshold: {fast_threshold})")
                
                if passed:
                    self._update_cache(query, fast_decision)
                    self.tier_stats["TIER_1_GLINER"] += 1
                    return fast_decision
                decision = fast_decision
        
        # Tier 2: LLM
        if RoutingTier.SLOW_PATH in self.strategies:
            if self.verbose:
                print(f"🤖 Tier 2 (LLM Slow Path) - Threshold: {slow_threshold}")
            
            slow_decision = await self._try_slow_path(query, context, decision)
            if slow_decision:
                conf = slow_decision.confidence_score
                passed = conf >= slow_threshold
                status = "✅" if passed else "❌"
                
                if self.verbose:
                    print(f"   Confidence: {conf:.3f} {status}")
                    print(f"     → Modality: {slow_decision.search_modality.value} | Generation: {slow_decision.generation_type.value}")
                    if slow_decision.reasoning:
                        print(f"     → Reasoning: {slow_decision.reasoning[:80]}")
                else:
                    print(f"   T2 LLM:         {conf:.3f} {status} (threshold: {slow_threshold})")
                
                if passed:
                    self._update_cache(query, slow_decision)
                    self.tier_stats["TIER_2_LLM"] += 1
                    return slow_decision
                decision = slow_decision
        
        # Tier 3: LangExtract
        if RoutingTier.LANGEXTRACT in self.strategies:
            if self.verbose:
                print(f"🔬 Tier 3 (LangExtract) - Threshold: {langextract_threshold}")
            
            lang_decision = await self._try_langextract(query, context)
            if lang_decision:
                conf = lang_decision.confidence_score
                passed = conf >= langextract_threshold
                status = "✅" if passed else "❌"
                
                if self.verbose:
                    print(f"   Confidence: {conf:.3f} {status}")
                    print(f"     → Modality: {lang_decision.search_modality.value} | Generation: {lang_decision.generation_type.value}")
                    if lang_decision.reasoning:
                        print(f"     → Reasoning: {lang_decision.reasoning[:80]}")
                else:
                    print(f"   T3 LangExtract: {conf:.3f} {status} (threshold: {langextract_threshold})")
                
                if passed:
                    self._update_cache(query, lang_decision)
                    self.tier_stats["TIER_3_LANGEXTRACT"] += 1
                    return lang_decision
                decision = lang_decision
        
        # Tier 4: Fallback
        if RoutingTier.FALLBACK in self.strategies:
            if self.verbose:
                print(f"🔍 Tier 4 (Keyword Fallback) - Always passes")
            
            fallback_decision = await self._try_fallback(query, context)
            if fallback_decision:
                if self.verbose:
                    print(f"   Confidence: {fallback_decision.confidence_score:.3f} ✅")
                    print(f"     → Modality: {fallback_decision.search_modality.value} | Generation: {fallback_decision.generation_type.value}")
                else:
                    print(f"   T4 Keyword:     {fallback_decision.confidence_score:.3f} ✅ (always passes)")
                
                self.tier_stats["TIER_4_KEYWORD"] += 1
                return fallback_decision
        
        raise RuntimeError("No routing strategy succeeded")


# Test queries categorized by expected behavior
TEST_QUERIES = {
    "tier1_simple": [
        "show me videos about cats",
        "find documents about machine learning",
        "search for cooking videos",
    ],
    "tier2_complex": [
        "show me videos related to the document I just read",
        "compare the video analysis with the text summary",
        "what's the connection between these two topics",
    ],
    "tier3_structured": [
        "extract specific timestamps and speaker names from the video in JSON format",
        "parse the structured data from this API response and validate against schema",
        "extract regulatory compliance information in exact legal format",
    ],
    "tier4_fallback": [
        "xyzabc123 quantum flibbertigibbet",
        "",
    ]
}


async def run_demonstration(verbose=False, category=None):
    """Run the routing demonstration."""
    
    # Load config
    config_path = Path("configs/config.json")
    if not config_path.exists():
        print("❌ Config file not found")
        return
    
    with open(config_path) as f:
        full_config = json.load(f)
        
    # Extract the routing section
    if "routing" not in full_config:
        print("❌ No 'routing' section found in config.json")
        return
    
    config_dict = full_config["routing"]
    
    # Ensure all tiers are enabled
    config_dict["tier_config"]["enable_fast_path"] = True
    config_dict["tier_config"]["enable_slow_path"] = True
    config_dict["tier_config"]["enable_langextract"] = True
    config_dict["tier_config"]["enable_fallback"] = True
    
    print("="*80)
    print("COMPREHENSIVE ROUTING DEMONSTRATION")
    print("="*80)
    print(f"\nMode: {'VERBOSE (showing all tier attempts)' if verbose else 'CONCISE (showing final decisions)'}")
    print("\nConfiguration:")
    print("  • Tier 1 (GLiNER):      ≥ 0.70 confidence")
    print("  • Tier 2 (LLM):         ≥ 0.60 confidence")
    print("  • Tier 3 (LangExtract): ≥ 0.50 confidence")
    print("  • Tier 4 (Keyword):     Always passes")
    
    # Initialize router
    print("\nInitializing router...")
    router = RouterWithTracking(config_dict, verbose=verbose)
    
    # Filter queries if category specified
    if category and category in TEST_QUERIES:
        queries_to_test = {category: TEST_QUERIES[category]}
    else:
        queries_to_test = TEST_QUERIES
    
    # Test each category
    all_decisions = []
    for cat_name, queries in queries_to_test.items():
        print(f"\n{'='*80}")
        print(f"Testing: {cat_name.replace('_', ' ').upper()}")
        print("="*80)
        
        for query in queries:
            if not query:
                print(f"\n[Empty Query]")
            else:
                print(f"\nQuery: \"{query[:60]}{'...' if len(query) > 60 else ''}\"")
            print("-" * 40)
            
            decision = await router.route(query)
            all_decisions.append((query, decision))
            
            # Show final result
            method = decision.routing_method.lower()
            if "gliner" in method:
                final_tier = "TIER_1_GLINER"
            elif "llm" in method:
                final_tier = "TIER_2_LLM"
            elif "langextract" in method or "structured" in method:
                final_tier = "TIER_3_LANGEXTRACT"
            else:
                final_tier = "TIER_4_KEYWORD"
            
            if verbose:
                print(f"\n📍 Final Decision: {final_tier.replace('_', ' ')}")
                print(f"   Confidence: {decision.confidence_score:.3f}")
            else:
                print(f"   → Final: {final_tier.replace('_', ' ')} | {decision.search_modality.value} | {decision.generation_type.value}")
    
    # Summary statistics
    if not category:  # Only show full stats when testing all categories
        print("\n" + "="*80)
        print("ROUTING DISTRIBUTION SUMMARY")
        print("="*80)
        
        total_queries = sum(router.tier_stats.values())
        if total_queries > 0:
            for tier, count in router.tier_stats.items():
                if count > 0:
                    percentage = (count / total_queries) * 100
                    tier_name = tier.replace('_', ' ')
                    bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
                    print(f"\n{tier_name}:")
                    print(f"  {bar} {count}/{total_queries} ({percentage:.1f}%)")
                    
                    # Show example queries for this tier
                    examples = []
                    for q, d in all_decisions:
                        if tier == "TIER_1_GLINER" and "gliner" in d.routing_method.lower():
                            examples.append(q[:40] if q else "[empty]")
                        elif tier == "TIER_2_LLM" and "llm" in d.routing_method.lower():
                            examples.append(q[:40] if q else "[empty]")
                        elif tier == "TIER_3_LANGEXTRACT" and "langextract" in d.routing_method.lower():
                            examples.append(q[:40] if q else "[empty]")
                        elif tier == "TIER_4_KEYWORD" and "keyword" in d.routing_method.lower():
                            examples.append(q[:40] if q else "[empty]")
                    
                    if examples:
                        print(f"  Examples: {', '.join(examples[:2])}")
        
        print(f"\n✅ Demonstration completed - tested {total_queries} queries")


def main():
    parser = argparse.ArgumentParser(description="Unified routing system demonstration")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed tier-by-tier decision process"
    )
    parser.add_argument(
        "--category", "-c",
        choices=["tier1_simple", "tier2_complex", "tier3_structured", "tier4_fallback"],
        help="Test only specific category of queries"
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_demonstration(verbose=args.verbose, category=args.category))


if __name__ == "__main__":
    main()