#!/usr/bin/env python3
"""
Generate training data for routing using LangExtract.
This script uses LangExtract to analyze queries and generate high-quality labeled data
for training routing models.
"""

import json
import os
import asyncio
from typing import List, Dict, Any
from datetime import datetime
from langextract import LangExtract

# Sample queries from different categories
SAMPLE_QUERIES = [
    # Video-focused queries
    "Show me the presentation from yesterday's meeting",
    "Find the demo video about the new product features",
    "What did John say in the morning standup video?",
    "Show me all videos from the Q3 planning session",
    "Find clips where someone mentions the budget",
    
    # Text-focused queries
    "Search for documents about API documentation",
    "Find all mentions of 'security compliance' in our docs",
    "Show me the latest technical specifications",
    "What are the requirements in the PRD document?",
    "Find all error logs from last week",
    
    # Mixed modality queries
    "Summarize the key points from today's meetings and documents",
    "Give me a comprehensive overview of the project status",
    "What decisions were made in yesterday's review?",
    "Create a report of all customer feedback",
    "Analyze the performance metrics from all sources",
    
    # Time-based queries
    "Show me everything from last Monday",
    "What happened between 2pm and 4pm yesterday?",
    "Find all content from Q2 2024",
    "Show me this morning's activities",
    "What was discussed in meetings last week?",
    
    # Summary requests
    "Give me a brief summary of today's meetings",
    "Summarize the key decisions from this week",
    "Provide a quick overview of the project status",
    "What are the main action items?",
    "Give me the highlights from the past month"
]

def initialize_langextract():
    """Initialize LangExtract with API key."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    return LangExtract(
        model="gemini-2.0-flash-exp",
        api_key=api_key
    )

def create_extraction_prompt(query: str) -> str:
    """Create a detailed prompt for LangExtract to analyze the query."""
    return f"""
    Analyze this user query for a multi-modal RAG system and extract detailed routing information.
    
    Query: "{query}"
    
    Please analyze and provide:
    1. Primary search modality needed:
       - "video" if the query is specifically about video/visual content
       - "text" if the query is specifically about text/document content  
       - "both" if the query needs both video and text sources
    
    2. Generation type required:
       - "raw" if the user wants direct search results
       - "summary" if the user wants a condensed overview
       - "detailed" if the user wants comprehensive analysis
    
    3. Temporal context:
       - Extract any time references (dates, times, periods)
       - Identify if this is time-sensitive
    
    4. Key entities:
       - People mentioned
       - Topics/subjects
       - Document types
       - Actions/verbs
    
    5. Query intent:
       - Information retrieval
       - Summarization
       - Analysis
       - Reporting
    
    6. Confidence score (0.0-1.0) for this routing decision
    
    7. Detailed reasoning for the routing decision
    
    Output as JSON:
    {{
        "search_modality": "video" | "text" | "both",
        "generation_type": "raw" | "summary" | "detailed",
        "temporal_context": {{
            "has_temporal": boolean,
            "time_references": [list of time references],
            "is_time_sensitive": boolean
        }},
        "entities": {{
            "people": [list],
            "topics": [list],
            "document_types": [list],
            "actions": [list]
        }},
        "query_intent": "retrieval" | "summarization" | "analysis" | "reporting",
        "confidence": float,
        "reasoning": string,
        "routing_decision": {{
            "primary_strategy": "fast_path" | "slow_path" | "langextract" | "fallback",
            "recommended_tier": 1 | 2 | 3 | 4
        }}
    }}
    """

async def generate_training_data(extractor: LangExtract, queries: List[str]) -> List[Dict[str, Any]]:
    """Generate training data for all queries."""
    training_data = []
    
    for i, query in enumerate(queries, 1):
        print(f"Processing query {i}/{len(queries)}: {query[:50]}...")
        
        try:
            # Create extraction prompt
            prompt = create_extraction_prompt(query)
            
            # Extract routing information
            result = extractor.extract(prompt)
            
            # Parse result
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    # Try to extract JSON from the response
                    import re
                    json_match = re.search(r'\{.*\}', result, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group())
                    else:
                        print(f"  ⚠️  Failed to parse result for query {i}")
                        continue
            
            # Create training example
            training_example = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "langextract_analysis": result,
                "expected_routing": {
                    "search_modality": result.get("search_modality", "both"),
                    "generation_type": result.get("generation_type", "raw"),
                    "confidence": result.get("confidence", 0.5),
                    "routing_method": f"langextract_tier{result.get('routing_decision', {}).get('recommended_tier', 3)}"
                },
                "metadata": {
                    "entities": result.get("entities", {}),
                    "temporal_context": result.get("temporal_context", {}),
                    "query_intent": result.get("query_intent", "retrieval")
                }
            }
            
            training_data.append(training_example)
            print(f"  ✅ Generated training data for query {i}")
            
        except Exception as e:
            print(f"  ❌ Error processing query {i}: {e}")
            continue
        
        # Add a small delay to avoid rate limiting
        await asyncio.sleep(0.5)
    
    return training_data

async def main():
    """Main function to generate training data."""
    print("🚀 Starting LangExtract training data generation...")
    
    # Initialize LangExtract
    try:
        extractor = initialize_langextract()
        print("✅ LangExtract initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize LangExtract: {e}")
        return
    
    # Generate training data
    training_data = await generate_training_data(extractor, SAMPLE_QUERIES)
    
    # Save training data
    output_file = "data/training/langextract_training_data.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "total_examples": len(training_data),
            "model_used": "gemini-2.0-flash-exp",
            "examples": training_data
        }, f, indent=2)
    
    print(f"\n✅ Generated {len(training_data)} training examples")
    print(f"📁 Saved to {output_file}")
    
    # Print summary statistics
    modality_counts = {}
    generation_counts = {}
    intent_counts = {}
    
    for example in training_data:
        analysis = example["langextract_analysis"]
        modality_counts[analysis.get("search_modality", "unknown")] = \
            modality_counts.get(analysis.get("search_modality", "unknown"), 0) + 1
        generation_counts[analysis.get("generation_type", "unknown")] = \
            generation_counts.get(analysis.get("generation_type", "unknown"), 0) + 1
        intent_counts[analysis.get("query_intent", "unknown")] = \
            intent_counts.get(analysis.get("query_intent", "unknown"), 0) + 1
    
    print("\n📊 Training Data Statistics:")
    print(f"  Search Modalities: {modality_counts}")
    print(f"  Generation Types: {generation_counts}")
    print(f"  Query Intents: {intent_counts}")
    
    # Calculate average confidence
    avg_confidence = sum(e["expected_routing"]["confidence"] for e in training_data) / len(training_data)
    print(f"  Average Confidence: {avg_confidence:.2f}")

if __name__ == "__main__":
    asyncio.run(main())