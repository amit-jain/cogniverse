#!/usr/bin/env python3
"""
GLiNER Label Optimization - The Actually Useful Approach

This optimizes the labels that GLiNER uses for entity extraction,
which is what really matters for routing accuracy.
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Tuple
import itertools
from collections import defaultdict

class GLiNERLabelOptimizer:
    """
    Optimize GLiNER labels for better entity extraction and routing.
    
    The key insight: If GLiNER can accurately identify the right entities,
    routing is trivial. The hard part is finding labels that GLiNER 
    recognizes well in your specific domain.
    """
    
    def __init__(self):
        # Start with a large pool of potential labels
        self.label_pool = {
            # Video-related labels to test
            "video_indicators": [
                "video", "videos", "video_content", "visual_content", 
                "footage", "clip", "clips", "recording", "media",
                "watch", "show", "play", "movie", "animation",
                "visual", "multimedia", "video_file", "video_material"
            ],
            
            # Document-related labels to test  
            "document_indicators": [
                "document", "documents", "text", "text_content",
                "paper", "papers", "article", "articles", "report",
                "doc", "docs", "file", "files", "written",
                "textual", "documentation", "text_file", "written_content"
            ],
            
            # Temporal labels to test
            "temporal_indicators": [
                "time", "date", "temporal", "when", "time_reference",
                "yesterday", "today", "tomorrow", "week", "month",
                "temporal_phrase", "time_period", "date_reference",
                "time_expression", "temporal_expression"
            ]
        }
        
        # Test queries with known correct routing
        self.test_queries = [
            # Video queries
            ("Show me videos about AI", {"video": True, "text": False, "temporal": None}),
            ("Play the presentation from yesterday", {"video": True, "text": False, "temporal": "yesterday"}),
            ("I want to watch the demo", {"video": True, "text": False, "temporal": None}),
            ("Find video clips of robots", {"video": True, "text": False, "temporal": None}),
            
            # Document queries
            ("Find documents about machine learning", {"video": False, "text": True, "temporal": None}),
            ("Show me the report from last week", {"video": False, "text": True, "temporal": "last_week"}),
            ("I need papers on neural networks", {"video": False, "text": True, "temporal": None}),
            ("Search for articles about AI", {"video": False, "text": True, "temporal": None}),
            
            # Mixed queries
            ("Find all content from yesterday", {"video": True, "text": True, "temporal": "yesterday"}),
            ("Search videos and documents about AI", {"video": True, "text": True, "temporal": None}),
            
            # Temporal queries
            ("What did we discuss yesterday", {"video": True, "text": True, "temporal": "yesterday"}),
            ("Show me everything from last month", {"video": True, "text": True, "temporal": "last_month"}),
        ]
    
    async def find_optimal_labels(self, gliner_model) -> Dict[str, Any]:
        """
        Find the optimal set of labels for GLiNER that maximize routing accuracy.
        
        This is what's actually useful - finding labels that GLiNER recognizes well
        in your specific domain and query patterns.
        """
        print("🔍 Finding Optimal GLiNER Labels")
        print("=" * 60)
        
        # Step 1: Test individual labels to see which ones GLiNER recognizes
        label_performance = await self._test_individual_labels(gliner_model)
        
        # Step 2: Find best combinations
        best_combination = await self._find_best_combination(
            gliner_model, 
            label_performance
        )
        
        # Step 3: Fine-tune thresholds
        optimal_config = await self._optimize_threshold(
            gliner_model,
            best_combination
        )
        
        return optimal_config
    
    async def _test_individual_labels(self, gliner_model) -> Dict[str, float]:
        """Test how well GLiNER recognizes each individual label."""
        print("\n📊 Testing individual labels...")
        
        label_scores = defaultdict(lambda: {"hits": 0, "attempts": 0})
        
        for category, labels in self.label_pool.items():
            print(f"\n🏷️ Testing {category}:")
            
            for label in labels:
                # Test this label on relevant queries
                for query, expected in self.test_queries:
                    # Only test on relevant queries
                    if category == "video_indicators" and not expected["video"]:
                        continue
                    if category == "document_indicators" and not expected["text"]:
                        continue
                    if category == "temporal_indicators" and not expected["temporal"]:
                        continue
                    
                    # Run GLiNER
                    entities = gliner_model.predict_entities(
                        query, 
                        [label], 
                        threshold=0.3
                    )
                    
                    # Check if label was found
                    if entities:
                        label_scores[label]["hits"] += 1
                    label_scores[label]["attempts"] += 1
                
                # Calculate recognition rate
                if label_scores[label]["attempts"] > 0:
                    score = label_scores[label]["hits"] / label_scores[label]["attempts"]
                    print(f"   {label}: {score:.1%} recognition rate")
        
        # Convert to simple scores
        return {
            label: scores["hits"] / scores["attempts"] 
            for label, scores in label_scores.items()
            if scores["attempts"] > 0
        }
    
    async def _find_best_combination(self, gliner_model, label_scores: Dict[str, float]) -> List[str]:
        """Find the best combination of labels for routing."""
        print("\n🎯 Finding optimal label combination...")
        
        # Get top performing labels from each category
        top_video_labels = sorted(
            [(label, score) for label, score in label_scores.items() 
             if label in self.label_pool["video_indicators"]],
            key=lambda x: x[1], reverse=True
        )[:3]
        
        top_doc_labels = sorted(
            [(label, score) for label, score in label_scores.items()
             if label in self.label_pool["document_indicators"]], 
            key=lambda x: x[1], reverse=True
        )[:3]
        
        top_temporal_labels = sorted(
            [(label, score) for label, score in label_scores.items()
             if label in self.label_pool["temporal_indicators"]],
            key=lambda x: x[1], reverse=True  
        )[:2]
        
        print(f"\n✅ Top video labels: {[l[0] for l in top_video_labels]}")
        print(f"✅ Top document labels: {[l[0] for l in top_doc_labels]}")
        print(f"✅ Top temporal labels: {[l[0] for l in top_temporal_labels]}")
        
        # Test combinations
        best_accuracy = 0
        best_labels = []
        
        # Try different combinations
        for n_video in [1, 2]:
            for n_doc in [1, 2]:
                for n_temporal in [1, 2]:
                    # Create combination
                    labels = (
                        [l[0] for l in top_video_labels[:n_video]] +
                        [l[0] for l in top_doc_labels[:n_doc]] +
                        [l[0] for l in top_temporal_labels[:n_temporal]]
                    )
                    
                    # Test this combination
                    accuracy = await self._test_label_combination(gliner_model, labels)
                    
                    print(f"\n📊 Labels: {labels}")
                    print(f"   Accuracy: {accuracy:.1%}")
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_labels = labels
        
        print(f"\n🏆 Best combination: {best_labels} ({best_accuracy:.1%})")
        return best_labels
    
    async def _test_label_combination(self, gliner_model, labels: List[str], threshold: float = 0.3) -> float:
        """Test how well a label combination performs on routing."""
        correct = 0
        total = 0
        
        for query, expected in self.test_queries:
            # Get GLiNER entities
            entities = gliner_model.predict_entities(query, labels, threshold=threshold)
            
            # Simple routing logic based on entities
            found_video = any(e["label"] in self.label_pool["video_indicators"] for e in entities)
            found_text = any(e["label"] in self.label_pool["document_indicators"] for e in entities)
            found_temporal = any(e["label"] in self.label_pool["temporal_indicators"] for e in entities)
            
            # Check routing accuracy
            routing_correct = True
            if found_video != expected["video"]:
                routing_correct = False
            if found_text != expected["text"]:
                routing_correct = False
            if expected["temporal"] and not found_temporal:
                routing_correct = False
                
            if routing_correct:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0
    
    async def _optimize_threshold(self, gliner_model, labels: List[str]) -> Dict[str, Any]:
        """Find optimal threshold for the best label combination."""
        print("\n🎚️ Optimizing threshold...")
        
        best_threshold = 0.3
        best_accuracy = 0
        
        for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            accuracy = await self._test_label_combination(gliner_model, labels, threshold)
            print(f"   Threshold {threshold}: {accuracy:.1%}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        return {
            "labels": labels,
            "threshold": best_threshold,
            "accuracy": best_accuracy,
            "label_count": len(labels)
        }


def demonstrate_why_this_matters():
    """Show why optimizing GLiNER labels is the key."""
    print("\n💡 Why Label Optimization is What Actually Matters:")
    print("=" * 60)
    
    print("\n❌ Bad Labels (Generic):")
    print('   Labels: ["entity", "thing", "object", "content"]')
    print("   → GLiNER finds nothing useful")
    print("   → Routing fails")
    
    print("\n✅ Good Labels (Optimized for your domain):")
    print('   Labels: ["video", "document", "yesterday"]')
    print("   → GLiNER accurately identifies entities")
    print("   → Routing is trivial:")
    print("      - Found 'video' → route to video search")
    print("      - Found 'document' → route to text search")
    print("      - Found 'yesterday' → add temporal filter")
    
    print("\n📊 The Real Challenge:")
    print("Finding labels that:")
    print("1. GLiNER recognizes well (model-specific)")
    print("2. Match your users' query patterns (domain-specific)")
    print("3. Map clearly to routing decisions (task-specific)")
    
    print("\n🎯 Once you have good labels, routing is just:")
    print("if 'video' in entities: needs_video_search = True")
    print("No complex reasoning needed!")


if __name__ == "__main__":
    print("🚀 GLiNER Label Optimization")
    print("The Actually Useful Approach")
    print("=" * 60)
    
    demonstrate_why_this_matters()
    
    print("\n\n📝 To use this optimizer:")
    print("1. Load your GLiNER model")
    print("2. Run the optimizer with your test queries")
    print("3. Get optimized labels that GLiNER recognizes well")
    print("4. Use these labels in production")
    print("\nNo DSPy needed - just find labels that work!")