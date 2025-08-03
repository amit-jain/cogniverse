#!/usr/bin/env python3
"""
Example: Video Chunks Processing Pipeline

This example demonstrates how to process videos using the TwelveLabs-style
chunking approach with VideoPrism embeddings.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.processing.pipeline_steps.video_chunk_processor import VideoChunkProcessor
from src.processing.pipeline_steps.audio_transcriber import AudioTranscriber
from src.models.videoprism_loader import load_videoprism_model
import numpy as np


def process_video_with_chunks(video_path: Path):
    """
    Example of processing a video into searchable chunks.
    """
    print(f"Processing video: {video_path.name}")
    
    # Step 1: Transcribe audio
    print("1. Transcribing audio...")
    transcriber = AudioTranscriber(model_size="base")
    transcript_data = transcriber.transcribe_audio(video_path, Path("temp_output"))
    print(f"   Found {len(transcript_data.get('segments', []))} transcript segments")
    
    # Step 2: Process video into chunks
    print("2. Creating video chunks...")
    chunk_processor = VideoChunkProcessor(
        chunk_duration=6.0,      # 6-second chunks
        chunk_overlap=1.0,       # 1-second overlap
        sampling_fps=2.0,        # 2 FPS sampling
        max_frames_per_chunk=12  # Max 12 frames per chunk
    )
    
    video_data = chunk_processor.process_video(video_path, transcript_data)
    print(f"   Created {len(video_data['chunks'])} chunks")
    
    # Step 3: Generate embeddings for each chunk
    print("3. Generating embeddings for chunks...")
    model, processor = load_videoprism_model("videoprism_public_v1_large_hf")
    
    chunk_embeddings = []
    for i, chunk in enumerate(video_data['chunks']):
        # Process frames for this chunk
        if chunk.frames:
            # VideoPrism expects RGB frames
            frames_rgb = [frame[:, :, ::-1] for frame in chunk.frames]  # BGR to RGB
            
            # Generate embedding for chunk
            # Note: Actual implementation would batch process frames
            embedding = model.encode_frames(frames_rgb)  # This is pseudo-code
            chunk_embeddings.append(embedding)
            
            print(f"   Chunk {i}: {chunk.start_time:.1f}-{chunk.end_time:.1f}s, "
                  f"{len(chunk.frames)} frames, "
                  f"transcript: '{chunk.transcript_text[:50]}...'")
    
    # Step 4: Create Vespa document
    print("4. Creating Vespa document...")
    doc = chunk_processor.create_vespa_document(
        video_data=video_data,
        chunk_embeddings=chunk_embeddings,
        video_metadata={
            "title": "Sample Video Title",
            "keywords": "demo, example, video chunks",
            "summary": "This is a demonstration of video chunk processing"
        }
    )
    
    print("\nDocument structure:")
    print(f"- Video ID: {doc['video_id']}")
    print(f"- Duration: {doc['duration']:.1f}s")
    print(f"- Chunks: {len(doc['chunk_embeddings'])}")
    print(f"- Full transcript length: {len(doc['transcript'])} chars")
    print(f"- Chunk transcripts: {len(doc['chunk_transcripts'])} segments")
    
    return doc


def search_example(query_text: str, query_embedding: np.ndarray):
    """
    Example of how search would work with chunked videos.
    """
    print(f"\nSearching for: '{query_text}'")
    
    # In a real implementation, this would query Vespa
    # Here we show the conceptual search process:
    
    # 1. Text search across title, keywords, summary, and full transcript
    text_score = compute_bm25_score(query_text)  # Pseudo-code
    
    # 2. Semantic search across all chunks
    # For each video document:
    #   - Compare query_embedding with each chunk_embedding
    #   - Find best matching chunk
    #   - Use max similarity as video score
    
    # 3. Hybrid ranking combines both scores
    # final_score = text_score + 10 * max_chunk_similarity
    
    # 4. Return results with timestamps
    results = {
        "video_id": "sample_video",
        "title": "Sample Video Title",
        "relevance_score": 0.95,
        "matching_chunks": [
            {
                "chunk_id": 5,
                "start_time": 25.0,
                "end_time": 31.0,
                "similarity": 0.89,
                "transcript": "This segment talks about the exact topic..."
            },
            {
                "chunk_id": 12,
                "start_time": 60.0,
                "end_time": 66.0,
                "similarity": 0.76,
                "transcript": "Another relevant segment here..."
            }
        ]
    }
    
    print("\nSearch Results:")
    print(f"Found video: {results['title']} (score: {results['relevance_score']:.2f})")
    print(f"Matching segments:")
    for chunk in results['matching_chunks']:
        print(f"  - {chunk['start_time']:.1f}-{chunk['end_time']:.1f}s: "
              f"'{chunk['transcript'][:50]}...' (similarity: {chunk['similarity']:.2f})")


def compute_bm25_score(query: str) -> float:
    """Placeholder for BM25 scoring"""
    return 0.5


if __name__ == "__main__":
    # Example usage
    video_path = Path("data/videos/sample_video.mp4")
    
    if video_path.exists():
        # Process video
        doc = process_video_with_chunks(video_path)
        
        # Example search
        search_example(
            query_text="machine learning tutorial",
            query_embedding=np.random.randn(1024)  # Placeholder embedding
        )
    else:
        print(f"Video not found: {video_path}")
        print("\nThis is a demonstration of the video chunks processing pipeline.")
        print("Key features:")
        print("- 6-second overlapping chunks")
        print("- Transcript alignment with chunks")
        print("- Single document per video")
        print("- Efficient chunk-level search")