#!/usr/bin/env python3
"""
Extract queries from Video-ChatGPT QA files that match our sample videos.
This creates a more relevant test set based on actual QA pairs for our videos.
"""

import json
from pathlib import Path
from typing import Dict, List, Set

def extract_video_ids_from_filenames(sample_videos_dir: Path) -> Set[str]:
    """Extract video IDs from sample video filenames."""
    video_ids = set()
    
    for video_file in sample_videos_dir.glob("*.mp4"):
        video_id = video_file.stem
        video_ids.add(video_id)
    
    for video_file in sample_videos_dir.glob("*.mkv"):
        video_id = video_file.stem
        video_ids.add(video_id)
    
    return video_ids

def extract_queries_for_videos(qa_file: Path, video_ids: Set[str]) -> List[Dict]:
    """Extract QA pairs that match our sample videos."""
    matching_queries = []
    
    try:
        with open(qa_file, 'r') as f:
            qa_data = json.load(f)
        
        for qa_pair in qa_data:
            video_name = qa_pair.get('video_name', '').replace('.mp4', '').replace('.mkv', '')
            if video_name in video_ids:
                matching_queries.append({
                    'video_id': video_name,
                    'question': qa_pair.get('Q', ''),
                    'answer': qa_pair.get('A', ''),
                    'source': qa_file.name
                })
    except Exception as e:
        print(f"Error processing {qa_file}: {e}")
    
    return matching_queries

def main():
    # Paths
    base_dir = Path("/Users/amjain/source/hobby/cogniverse")
    sample_videos_dir = base_dir / "data/testset/evaluation/sample_videos"
    queries_dir = base_dir / "data/testset/queries"
    output_dir = base_dir / "data/testset/evaluation"
    
    # Get sample video IDs
    video_ids = extract_video_ids_from_filenames(sample_videos_dir)
    print(f"Found {len(video_ids)} sample videos:")
    for vid in sorted(video_ids):
        print(f"  - {vid}")
    
    # Extract queries from each QA file
    all_queries = []
    qa_files = [
        "generic_qa.json",
        "temporal_qa.json", 
        "consistency_qa.json"
    ]
    
    for qa_filename in qa_files:
        qa_file = queries_dir / qa_filename
        if qa_file.exists():
            queries = extract_queries_for_videos(qa_file, video_ids)
            all_queries.extend(queries)
            print(f"\nExtracted {len(queries)} queries from {qa_filename}")
    
    # Group queries by type and video
    queries_by_type = {}
    queries_by_video = {}
    
    for query in all_queries:
        # By type
        source = query['source'].replace('.json', '').replace('_qa', '')
        if source not in queries_by_type:
            queries_by_type[source] = []
        queries_by_type[source].append(query)
        
        # By video
        video_id = query['video_id']
        if video_id not in queries_by_video:
            queries_by_video[video_id] = []
        queries_by_video[video_id].append(query)
    
    # Create retrieval test queries
    retrieval_queries = []
    
    for query in all_queries:
        # Create different types of retrieval queries from the QA
        video_id = query['video_id']
        question = query['question']
        answer = query['answer']
        
        # 1. Direct question as query
        retrieval_queries.append({
            'query': question,
            'expected_videos': [video_id],
            'ground_truth': answer,
            'query_type': 'question',
            'source': query['source']
        })
        
        # 2. Extract key phrases from answer as queries
        # Focus on visual elements, actions, objects
        if 'generic' in query['source']:
            # For generic queries, use parts of the answer
            if len(answer) > 20:
                key_phrases = extract_key_phrases_from_answer(answer)
                for phrase in key_phrases:
                    retrieval_queries.append({
                        'query': phrase,
                        'expected_videos': [video_id],
                        'ground_truth': answer,
                        'query_type': 'answer_phrase',
                        'source': query['source']
                    })
    
    # Save results
    output_file = output_dir / "sample_videos_retrieval_queries.json"
    with open(output_file, 'w') as f:
        json.dump(retrieval_queries, f, indent=2)
    
    print(f"\n✅ Created {len(retrieval_queries)} retrieval queries")
    print(f"📄 Saved to: {output_file}")
    
    # Print summary
    print("\n📊 Summary by video:")
    for video_id, queries in queries_by_video.items():
        print(f"  {video_id}: {len(queries)} QA pairs")
    
    print("\n📊 Summary by type:")
    for qtype, queries in queries_by_type.items():
        print(f"  {qtype}: {len(queries)} queries")

def extract_key_phrases_from_answer(answer: str) -> List[str]:
    """Extract key visual phrases from answer text."""
    phrases = []
    
    # Simple extraction of noun phrases and action phrases
    # This is a basic implementation - could be enhanced with NLP
    
    # Look for patterns like "person wearing X", "man doing Y", etc.
    import re
    
    # Action patterns
    action_patterns = [
        r'(person|man|woman|people) (?:is |are )?([\w\s]+ing)',
        r'(lifting|throwing|catching|jumping|running|walking|standing)',
    ]
    
    # Object patterns  
    object_patterns = [
        r'wearing ([\w\s]+)',
        r'holding ([\w\s]+)',
        r'with ([\w\s]+)',
    ]
    
    for pattern in action_patterns:
        matches = re.findall(pattern, answer.lower())
        for match in matches:
            if isinstance(match, tuple):
                phrase = ' '.join(match).strip()
            else:
                phrase = match.strip()
            if len(phrase) > 3:
                phrases.append(phrase)
    
    for pattern in object_patterns:
        matches = re.findall(pattern, answer.lower())
        for match in matches:
            if isinstance(match, tuple):
                phrase = ' '.join(match).strip()
            else:
                phrase = match.strip()
            if len(phrase) > 3 and len(phrase.split()) < 5:
                phrases.append(phrase)
    
    # Limit to unique phrases
    return list(set(phrases))[:3]

if __name__ == "__main__":
    main()