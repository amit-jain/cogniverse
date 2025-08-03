#!/usr/bin/env python3
"""
Example: Single Vector Video Processing

Demonstrates how to use the generic single-vector processing pipeline
for different strategies (chunks, windows, global) with any model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.processing.pipeline_steps.single_vector_processor import SingleVectorVideoProcessor
from src.processing.pipeline_steps.embedding_generator.single_vector_document_builder import SingleVectorDocumentBuilder
from src.processing.pipeline_steps.audio_transcriber import AudioTranscriber
import numpy as np


def demonstrate_all_strategies(video_path: Path):
    """
    Demonstrate all three single-vector strategies on the same video.
    """
    # First, transcribe audio (shared across all strategies)
    print("Transcribing audio...")
    transcriber = AudioTranscriber(model_size="base")
    transcript_data = transcriber.transcribe_audio(video_path, Path("temp_output"))
    
    strategies = [
        {
            "name": "single__video_videoprism_large_6s",
            "processor_config": {
                "strategy": "chunks",
                "segment_duration": 6.0,
                "segment_overlap": 1.0,
                "sampling_fps": 2.0,
                "max_frames_per_segment": 12,
                "store_as_single_doc": True
            },
            "description": "TwelveLabs-style 6s chunks with overlap"
        },
        {
            "name": "single__video_videoprism_large_30s",
            "processor_config": {
                "strategy": "windows",
                "segment_duration": 30.0,
                "segment_overlap": 0,
                "sampling_fps": 1.0,
                "max_frames_per_segment": 30,
                "store_as_single_doc": False
            },
            "description": "30-second windows, separate documents"
        },
        {
            "name": "single__video_videoprism_lvt_base_global",
            "processor_config": {
                "strategy": "global",
                "segment_duration": 0,  # Ignored for global
                "sampling_fps": 1.0,
                "max_frames_per_segment": 40,
                "store_as_single_doc": True
            },
            "description": "Global video embedding"
        }
    ]
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy['name']}")
        print(f"Description: {strategy['description']}")
        print('='*60)
        
        # Create processor
        processor = SingleVectorVideoProcessor(**strategy['processor_config'])
        
        # Process video
        video_data = processor.process_video(
            video_path=video_path,
            transcript_data=transcript_data,
            metadata={
                "title": f"Demo Video - {strategy['name']}",
                "keywords": "demo, single vector, " + strategy['processor_config']['strategy']
            }
        )
        
        print(f"\nProcessing Results:")
        print(f"- Segments created: {len(video_data['segments'])}")
        print(f"- Strategy: {video_data['metadata']['strategy']}")
        print(f"- Document structure: {video_data['document_structure']['type']}")
        
        # Show segment details
        for i, segment in enumerate(video_data['segments'][:3]):  # First 3 segments
            print(f"\nSegment {i}:")
            print(f"  - Time: {segment.start_time:.1f}-{segment.end_time:.1f}s")
            print(f"  - Frames: {len(segment.frames)}")
            print(f"  - Transcript: '{segment.transcript_text[:50]}...'")
        
        if len(video_data['segments']) > 3:
            print(f"\n... and {len(video_data['segments']) - 3} more segments")
        
        # Simulate embedding generation
        print("\nSimulating embedding generation...")
        mock_embeddings = [
            np.random.randn(1024) for _ in video_data['segments']
        ]
        
        # Create documents
        builder = SingleVectorDocumentBuilder(
            schema_name="video_single",
            storage_mode=video_data['document_structure']['type'].replace('_', '').replace('documents', 'doc')
        )
        
        documents = builder.build_documents(
            video_data=video_data,
            embeddings=mock_embeddings,
            additional_metadata={
                "url": "https://example.com/video.mp4"
            }
        )
        
        print(f"\nDocuments created: {len(documents)}")
        for doc in documents[:2]:  # Show first 2 docs
            print(f"\nDocument ID: {doc['put']}")
            print(f"Fields: {list(doc['fields'].keys())}")
            
            # For single doc, show tensor structure
            if 'embeddings' in doc['fields']:
                print(f"Embeddings tensor keys: {list(doc['fields']['embeddings'].keys())[:5]}...")
            if 'segment_metadata' in doc['fields']:
                print(f"Segment metadata keys: {list(doc['fields']['segment_metadata'].keys())[:5]}...")


def compare_storage_efficiency():
    """
    Compare storage efficiency between strategies.
    """
    print("\n" + "="*60)
    print("Storage Efficiency Comparison")
    print("="*60)
    
    video_duration = 300  # 5 minutes
    
    strategies = [
        ("6s chunks (single doc)", 6, 1, 1),
        ("6s chunks (multi doc)", 6, 1, 50),
        ("30s windows (multi doc)", 30, 0, 10),
        ("Global (single doc)", 300, 0, 1)
    ]
    
    for name, duration, overlap, num_docs in strategies:
        if duration < video_duration:
            num_segments = int((video_duration - overlap) / (duration - overlap))
        else:
            num_segments = 1
            
        print(f"\n{name}:")
        print(f"  - Segments: {num_segments}")
        print(f"  - Documents: {num_docs}")
        print(f"  - Efficiency: {num_segments/num_docs:.1f} segments per doc")


if __name__ == "__main__":
    # Example usage
    video_path = Path("data/videos/sample_video.mp4")
    
    if video_path.exists():
        demonstrate_all_strategies(video_path)
    else:
        print("Video not found. Running conceptual demonstration...")
        print("\nThe generic SingleVectorVideoProcessor supports:")
        print("1. Any segmentation strategy (chunks, windows, global)")
        print("2. Any embedding model (VideoPrism, custom, future)")
        print("3. Flexible document storage (single doc or multi-doc)")
        print("4. Consistent interface across all approaches")
        
    compare_storage_efficiency()