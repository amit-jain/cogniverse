#!/usr/bin/env python
"""
Create a cache of training examples from COMPREHENSIVE_ROUTING.md
and implement LangExtract-style data generation using the existing cache system
"""

import json
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the base module first
from src.common.cache.base import CacheBackend

# Now import the backend
from src.common.cache.backends.structured_filesystem import StructuredFilesystemBackend, StructuredFilesystemConfig

# Training examples from COMPREHENSIVE_ROUTING.md
TRAINING_EXAMPLES = [
    # Video + Raw Results
    {
        "query": "Show me how to bake a cake",
        "search_modality": "video",
        "generation_type": "raw_results",
        "entities": ["cake", "baking"],
        "intent": "tutorial",
        "sentiment": "neutral"
    },
    {
        "query": "Find me tutorials on Python programming",
        "search_modality": "video", 
        "generation_type": "raw_results",
        "entities": ["Python", "programming", "tutorials"],
        "intent": "learn",
        "sentiment": "neutral"
    },
    {
        "query": "Show me the pasta making video",
        "search_modality": "video",
        "generation_type": "raw_results",
        "entities": ["pasta", "making", "video"],
        "intent": "watch",
        "sentiment": "neutral"
    },
    
    # Text + Detailed Report
    {
        "query": "Construct a detailed report on the economic impact of AI",
        "search_modality": "text",
        "generation_type": "detailed_report",
        "entities": ["economic impact", "AI", "report"],
        "intent": "analyze",
        "sentiment": "neutral"
    },
    {
        "query": "Create a comprehensive analysis of climate change policies",
        "search_modality": "text",
        "generation_type": "detailed_report",
        "entities": ["climate change", "policies", "analysis"],
        "intent": "analyze",
        "sentiment": "neutral"
    },
    {
        "query": "Build a detailed report on solar panel efficiency",
        "search_modality": "text",
        "generation_type": "detailed_report",
        "entities": ["solar panel", "efficiency", "report"],
        "intent": "research",
        "sentiment": "neutral"
    },
    
    # Text + Summary
    {
        "query": "What is the main point of the new climate study?",
        "search_modality": "text",
        "generation_type": "summary",
        "entities": ["climate study", "main point"],
        "intent": "summarize",
        "sentiment": "neutral"
    },
    {
        "query": "Summarize the key findings about vaccine effectiveness",
        "search_modality": "text",
        "generation_type": "summary",
        "entities": ["vaccine", "effectiveness", "findings"],
        "intent": "summarize",
        "sentiment": "neutral"
    },
    {
        "query": "Give me a quick summary of quantum supremacy",
        "search_modality": "text",
        "generation_type": "summary",
        "entities": ["quantum supremacy", "summary"],
        "intent": "summarize",
        "sentiment": "neutral"
    },
    
    # Video + Summary
    {
        "query": "Summarize the main points from the TED talk on leadership",
        "search_modality": "video",
        "generation_type": "summary",
        "entities": ["TED talk", "leadership", "main points"],
        "intent": "summarize",
        "sentiment": "neutral"
    },
    {
        "query": "What were the key takeaways from that video?",
        "search_modality": "video",
        "generation_type": "summary",
        "entities": ["video", "key takeaways"],
        "intent": "summarize",
        "sentiment": "neutral"
    },
    
    # Text + Raw Results
    {
        "query": "Find research papers on machine learning",
        "search_modality": "text",
        "generation_type": "raw_results",
        "entities": ["research papers", "machine learning"],
        "intent": "search",
        "sentiment": "neutral"
    },
    {
        "query": "Show me articles about space exploration",
        "search_modality": "text",
        "generation_type": "raw_results",
        "entities": ["articles", "space exploration"],
        "intent": "search",
        "sentiment": "neutral"
    },
    
    # Complex queries from the document
    {
        "query": "I'm frustrated with the terrible service on my flight to San Francisco last Tuesday, compare the business class options on United and Delta for a return trip in three weeks",
        "search_modality": "both",
        "generation_type": "detailed_report",
        "entities": ["San Francisco", "United", "Delta", "business class", "flight"],
        "intent": "complaint_and_compare",
        "sentiment": "negative",
        "temporal": "last Tuesday, in three weeks"
    },
    
    # Additional varied examples
    {
        "query": "Show me yesterday's presentation about quarterly results",
        "search_modality": "video",
        "generation_type": "raw_results",
        "entities": ["presentation", "quarterly results"],
        "intent": "review",
        "sentiment": "neutral",
        "temporal": "yesterday"
    },
    {
        "query": "Find all documentation about the new API from last month",
        "search_modality": "text",
        "generation_type": "raw_results",
        "entities": ["documentation", "API"],
        "intent": "search",
        "sentiment": "neutral",
        "temporal": "last month"
    },
    {
        "query": "Create a comprehensive report comparing AWS and Azure services",
        "search_modality": "text",
        "generation_type": "detailed_report",
        "entities": ["AWS", "Azure", "services", "comparison"],
        "intent": "compare",
        "sentiment": "neutral"
    },
    {
        "query": "What did the CEO say in the all-hands meeting?",
        "search_modality": "video",
        "generation_type": "summary",
        "entities": ["CEO", "all-hands meeting"],
        "intent": "summarize",
        "sentiment": "neutral"
    },
    {
        "query": "Search for videos and documents about quantum computing from 2024",
        "search_modality": "both",
        "generation_type": "raw_results",
        "entities": ["quantum computing", "videos", "documents"],
        "intent": "search",
        "sentiment": "neutral",
        "temporal": "2024"
    },
    {
        "query": "I need urgent help with the broken deployment pipeline",
        "search_modality": "both",
        "generation_type": "raw_results",
        "entities": ["deployment pipeline", "help"],
        "intent": "troubleshoot",
        "sentiment": "negative"
    }
]

async def create_training_cache():
    """Create a training cache using the existing cache system"""
    
    # Initialize the structured filesystem cache
    cache_dir = Path.home() / ".cache" / "cogniverse" / "routing_training"
    
    config = StructuredFilesystemConfig(
        base_path=str(cache_dir),
        serialization_format="json",
        enable_ttl=False,  # Training data doesn't expire
        cleanup_on_startup=False
    )
    
    cache = StructuredFilesystemBackend(config)
    
    print(f"🗄️ Using cache at: {cache_dir}")
    
    # Store each training example in the cache
    for i, example in enumerate(TRAINING_EXAMPLES):
        cache_key = f"training_example_{i:03d}"
        await cache.set(cache_key, example)
    
    print(f"✅ Cached {len(TRAINING_EXAMPLES)} training examples")
    
    # Create metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "total_examples": len(TRAINING_EXAMPLES),
        "modality_distribution": {},
        "generation_distribution": {},
        "intent_distribution": {}
    }
    
    for example in TRAINING_EXAMPLES:
        # Count modalities
        modality = example["search_modality"]
        metadata["modality_distribution"][modality] = metadata["modality_distribution"].get(modality, 0) + 1
        
        # Count generation types
        gen_type = example["generation_type"]
        metadata["generation_distribution"][gen_type] = metadata["generation_distribution"].get(gen_type, 0) + 1
        
        # Count intents
        intent = example.get("intent", "unknown")
        metadata["intent_distribution"][intent] = metadata["intent_distribution"].get(intent, 0) + 1
    
    # Store metadata in cache
    await cache.set("training_metadata", metadata)
    
    # Also create a manifest of all training keys
    manifest = {
        "example_keys": [f"training_example_{i:03d}" for i in range(len(TRAINING_EXAMPLES))],
        "metadata_key": "training_metadata",
        "created_at": datetime.now().isoformat()
    }
    await cache.set("training_manifest", manifest)
    
    print(f"📊 Training data distribution:")
    print(f"   Modalities: {metadata['modality_distribution']}")
    print(f"   Generation: {metadata['generation_distribution']}")
    print(f"   Intents: {metadata['intent_distribution']}")
    
    # Get cache stats
    stats = await cache.get_stats()
    print(f"\n📈 Cache statistics:")
    print(f"   Total entries: {stats.get('total_entries', 0)}")
    print(f"   Cache size: {stats.get('total_size_bytes', 0) / 1024:.2f} KB")
    
    return cache_dir

async def test_gliner_with_cache():
    """Test GLiNER with the training cache"""
    try:
        from gliner import GLiNER
        
        print("\n🧪 Testing GLiNER with cached training examples...")
        
        # Load training examples from cache
        cache_dir = Path.home() / ".cache" / "cogniverse" / "routing_training"
        
        config = StructuredFilesystemConfig(
            base_path=str(cache_dir),
            serialization_format="json",
            enable_ttl=False,
            cleanup_on_startup=False
        )
        
        cache = StructuredFilesystemBackend(config)
        
        # Get manifest to know which examples to load
        manifest = await cache.get("training_manifest")
        if not manifest:
            print("❌ No training cache found. Run create_training_cache first.")
            return
        
        print(f"📂 Loading from cache: {cache_dir}")
        
        # Load just one model for testing
        model_name = "urchade/gliner_multi-v2.1"
        print(f"📥 Loading {model_name}...")
        model = GLiNER.from_pretrained(model_name)
        
        # Define labels based on our training data
        labels = [
            "video_content", "text_content", "both_content",
            "tutorial", "documentation", "presentation",
            "summary_request", "detailed_report", "comparison",
            "temporal_reference", "location", "organization",
            "negative_sentiment", "positive_sentiment",
            "urgent", "complaint"
        ]
        
        # Test on cached examples
        correct = 0
        total = 0
        
        # Load first 10 examples from cache
        for i in range(min(10, len(manifest["example_keys"]))):
            key = manifest["example_keys"][i]
            example = await cache.get(key)
            
            if not example:
                continue
            
            query = example["query"]
            expected_modality = example["search_modality"]
            
            # Predict entities
            entities = model.predict_entities(query, labels, threshold=0.3)
            
            # Determine predicted modality based on entities
            predicted_modality = "both"  # default
            entity_labels = [e["label"] for e in entities]
            
            if "video_content" in entity_labels and "text_content" not in entity_labels:
                predicted_modality = "video"
            elif "text_content" in entity_labels and "video_content" not in entity_labels:
                predicted_modality = "text"
            
            is_correct = predicted_modality == expected_modality
            if is_correct:
                correct += 1
            total += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{status} Query: {query[:50]}...")
            print(f"   Expected: {expected_modality}, Predicted: {predicted_modality}")
            print(f"   Entities: {entity_labels}")
        
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"\n📈 GLiNER Accuracy: {accuracy:.1f}% ({correct}/{total})")
        
        # Show cache stats
        stats = await cache.get_stats()
        print(f"\n📊 Cache stats: {stats.get('total_entries', 0)} entries")
        
    except ImportError:
        print("❌ GLiNER not available")
    except Exception as e:
        print(f"❌ Error testing GLiNER: {e}")

async def main():
    """Main function to setup training cache"""
    # Create training cache
    cache_dir = await create_training_cache()
    
    # Test with GLiNER
    await test_gliner_with_cache()
    
    print("\n✨ Training cache setup complete!")
    print(f"📂 Cache location: {cache_dir}")

if __name__ == "__main__":
    asyncio.run(main())