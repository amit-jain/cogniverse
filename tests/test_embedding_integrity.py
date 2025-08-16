#!/usr/bin/env python3
"""
Test script to verify embedding integrity in Vespa
Checks for cross-video contamination and validates embedding uniqueness
"""

import json
import sys
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import random
import argparse
import time

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))
from src.common.utils.output_manager import get_output_manager


class EmbeddingIntegrityTester:
    """Test embedding integrity across videos in Vespa"""
    
    def __init__(self, sample_size: Optional[int] = None):
        self.output_manager = get_output_manager()
        self.sample_size = sample_size
        self.results = {
            "test_time": datetime.now().isoformat(),
            "total_documents": 0,
            "videos_analyzed": set(),
            "tests": {
                "cross_video_uniqueness": {"status": "pending", "details": []},
                "within_video_variation": {"status": "pending", "details": []},
                "embedding_structure": {"status": "pending", "details": []},
                "statistical_analysis": {"status": "pending", "details": {}}
            },
            "duplicates_found": [],
            "warnings": []
        }
    
    
    def parse_embedding_from_doc(self, doc: dict) -> Optional[Dict[str, List[float]]]:
        """Extract embedding patches from a document"""
        fields = doc.get("fields", {})
        embedding_raw = fields.get("colpali_embedding", {})
        
        if not embedding_raw:
            return None
        
        # Handle Vespa tensor format
        blocks = embedding_raw.get("blocks", {})
        if not blocks:
            return None
        
        # Extract a subset of patches for comparison
        patches = {}
        for patch_id in ["0", "100", "500", "873"]:  # Sample patches
            if patch_id in blocks:
                patches[patch_id] = blocks[patch_id]
        
        return patches
    
    def build_embedding_index(self, export_file: Path) -> Dict:
        """Build an index of embeddings from exported data with proper sampling"""
        print("\n" + "=" * 80)
        print("BUILDING EMBEDDING INDEX")
        print("=" * 80)
        
        # First pass: collect all video IDs and frame counts
        video_frame_counts = defaultdict(int)
        total_docs = 0
        
        print("First pass: discovering all videos...")
        with open(export_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    doc = json.loads(line)
                    video_id = doc.get("fields", {}).get("video_id")
                    if video_id:
                        video_frame_counts[video_id] += 1
                        total_docs += 1
                except:
                    pass
        
        print(f"\nFound {len(video_frame_counts)} videos with {total_docs} total documents")
        for video_id, count in sorted(video_frame_counts.items()):
            print(f"  {video_id}: {count} frames")
        
        # Second pass: sample frames from each video
        index = defaultdict(lambda: defaultdict(dict))
        
        # Calculate how many frames to sample per video
        if self.sample_size and len(video_frame_counts) > 0:
            frames_per_video = max(5, self.sample_size // len(video_frame_counts))
        else:
            frames_per_video = 20  # Default sampling
        
        print(f"\nSecond pass: sampling up to {frames_per_video} frames per video...")
        
        video_samples = defaultdict(int)
        doc_count = 0
        
        with open(export_file, 'r') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                
                try:
                    doc = json.loads(line)
                    fields = doc.get("fields", {})
                    
                    video_id = fields.get("video_id")
                    frame_id = fields.get("frame_id")
                    
                    if not video_id or frame_id is None:
                        continue
                    
                    # Skip if we've sampled enough from this video
                    if video_samples[video_id] >= frames_per_video:
                        continue
                    
                    # Extract embedding samples
                    patches = self.parse_embedding_from_doc(doc)
                    if patches:
                        index[video_id][frame_id] = {
                            "patches": patches,
                            "num_patches": len(doc["fields"]["colpali_embedding"]["blocks"])
                        }
                        self.results["videos_analyzed"].add(video_id)
                        video_samples[video_id] += 1
                        doc_count += 1
                    
                    if doc_count % 100 == 0:
                        print(f"  Processed {doc_count} documents...")
                        
                except json.JSONDecodeError:
                    self.results["warnings"].append(f"Failed to parse line {line_num}")
                except Exception as e:
                    self.results["warnings"].append(f"Error processing line {line_num}: {str(e)}")
        
        self.results["total_documents"] = doc_count
        self.results["videos_analyzed"] = list(self.results["videos_analyzed"])
        
        print(f"\n✓ Indexed {doc_count} documents from {len(index)} videos")
        print("Samples per video:")
        for video_id, count in sorted(video_samples.items()):
            print(f"  {video_id}: {count} frames sampled")
        
        return dict(index)
    
    def test_cross_video_uniqueness(self, index: Dict) -> bool:
        """Test that different videos have different embeddings"""
        print("\n" + "=" * 80)
        print("TEST: CROSS-VIDEO UNIQUENESS")
        print("=" * 80)
        
        test_result = self.results["tests"]["cross_video_uniqueness"]
        videos = list(index.keys())
        
        if len(videos) < 2:
            test_result["status"] = "skipped"
            test_result["details"] = "Need at least 2 videos to test"
            return True
        
        duplicates_found = False
        comparisons = 0
        max_comparisons = 50  # Limit comparisons for performance
        
        # Compare same frame numbers across different videos
        for frame_id in range(min(10, min(len(index[v]) for v in videos))):
            for i, video1 in enumerate(videos[:-1]):
                for video2 in videos[i+1:]:
                    if frame_id not in index[video1] or frame_id not in index[video2]:
                        continue
                    
                    emb1 = index[video1][frame_id]["patches"]
                    emb2 = index[video2][frame_id]["patches"]
                    
                    # Compare multiple patches for thoroughness
                    patches_to_check = ["0", "100", "500", "873"]
                    identical_patches = 0
                    
                    for patch_id in patches_to_check:
                        if patch_id in emb1 and patch_id in emb2:
                            patch1 = np.array(emb1[patch_id])
                            patch2 = np.array(emb2[patch_id])
                            
                            if np.array_equal(patch1, patch2):
                                identical_patches += 1
                    
                    if identical_patches == len(patches_to_check):
                        duplicates_found = True
                        self.results["duplicates_found"].append({
                            "type": "cross_video",
                            "video1": video1,
                            "video2": video2,
                            "frame_id": frame_id,
                            "identical_patches": identical_patches
                        })
                        test_result["details"].append(
                            f"❌ DUPLICATE: {video1} and {video2} have identical embeddings at frame {frame_id} (all {identical_patches} patches checked)"
                        )
                    elif identical_patches > 0:
                        test_result["details"].append(
                            f"⚠️ WARNING: {video1} and {video2} have {identical_patches}/{len(patches_to_check)} identical patches at frame {frame_id}"
                        )
                    
                    comparisons += 1
                    if comparisons >= max_comparisons:
                        break
                if comparisons >= max_comparisons:
                    break
            if comparisons >= max_comparisons:
                break
        
        if not duplicates_found:
            test_result["status"] = "passed"
            test_result["details"].append(f"✓ No cross-video duplicates found in {comparisons} comparisons")
        else:
            test_result["status"] = "failed"
        
        print(f"\nTest Status: {test_result['status'].upper()}")
        for detail in test_result["details"][-5:]:  # Show last 5 details
            print(f"  {detail}")
        
        return not duplicates_found
    
    def test_within_video_variation(self, index: Dict) -> bool:
        """Test that frames within a video have different embeddings"""
        print("\n" + "=" * 80)
        print("TEST: WITHIN-VIDEO VARIATION")
        print("=" * 80)
        
        test_result = self.results["tests"]["within_video_variation"]
        all_good = True
        
        for video_id, frames in index.items():
            if len(frames) < 2:
                continue
            
            # Compare consecutive frames
            frame_ids = sorted(list(frames.keys()))[:10]  # Limit to first 10 frames
            
            for i in range(len(frame_ids) - 1):
                frame1_id = frame_ids[i]
                frame2_id = frame_ids[i + 1]
                
                emb1 = frames[frame1_id]["patches"]
                emb2 = frames[frame2_id]["patches"]
                
                # Check multiple patches
                patches_to_check = ["0", "100", "500", "873"]
                identical_patches = 0
                high_similarity_patches = 0
                
                for patch_id in patches_to_check:
                    if patch_id in emb1 and patch_id in emb2:
                        patch1 = np.array(emb1[patch_id])
                        patch2 = np.array(emb2[patch_id])
                        
                        if np.array_equal(patch1, patch2):
                            identical_patches += 1
                        else:
                            cosine_sim = np.dot(patch1, patch2) / (np.linalg.norm(patch1) * np.linalg.norm(patch2))
                            if cosine_sim > 0.95:
                                high_similarity_patches += 1
                
                if identical_patches == len(patches_to_check):
                    all_good = False
                    test_result["details"].append(
                        f"❌ IDENTICAL: {video_id} frames {frame1_id} and {frame2_id} have identical embeddings (all {identical_patches} patches)"
                    )
                elif identical_patches > 0:
                    all_good = False
                    test_result["details"].append(
                        f"⚠️ PARTIAL: {video_id} frames {frame1_id} and {frame2_id} have {identical_patches}/{len(patches_to_check)} identical patches"
                    )
        
        if all_good:
            test_result["status"] = "passed"
            test_result["details"].append("✓ All tested frame pairs show variation")
        else:
            test_result["status"] = "failed"
        
        print(f"\nTest Status: {test_result['status'].upper()}")
        for detail in test_result["details"][-5:]:  # Show last 5 details
            print(f"  {detail}")
        
        return all_good
    
    def test_embedding_structure(self, index: Dict) -> bool:
        """Test embedding structure and dimensions"""
        print("\n" + "=" * 80)
        print("TEST: EMBEDDING STRUCTURE")
        print("=" * 80)
        
        test_result = self.results["tests"]["embedding_structure"]
        all_good = True
        
        patch_counts = []
        embedding_dims = set()
        
        for video_id, frames in index.items():
            for frame_id, data in frames.items():
                patch_counts.append(data["num_patches"])
                
                # Check embedding dimensions
                for patch_id, values in data["patches"].items():
                    embedding_dims.add(len(values))
        
        # Validate structure
        if len(embedding_dims) == 1:
            expected_dim = list(embedding_dims)[0]
            test_result["details"].append(f"✓ Consistent embedding dimension: {expected_dim}")
        else:
            all_good = False
            test_result["details"].append(f"❌ Inconsistent embedding dimensions: {embedding_dims}")
        
        # Check patch count consistency
        unique_patch_counts = list(set(patch_counts))
        if len(unique_patch_counts) == 1:
            test_result["details"].append(f"✓ Consistent patch count: {unique_patch_counts[0]}")
        elif len(unique_patch_counts) <= 3:
            test_result["details"].append(f"⚠️ Variable patch counts (expected for different image sizes): {unique_patch_counts}")
        else:
            test_result["details"].append(f"❌ Too many different patch counts: {len(unique_patch_counts)}")
        
        test_result["status"] = "passed" if all_good else "failed"
        
        print(f"\nTest Status: {test_result['status'].upper()}")
        for detail in test_result["details"]:
            print(f"  {detail}")
        
        return all_good
    
    def test_statistical_analysis(self, index: Dict) -> bool:
        """Perform statistical analysis on embeddings"""
        print("\n" + "=" * 80)
        print("TEST: STATISTICAL ANALYSIS")
        print("=" * 80)
        
        test_result = self.results["tests"]["statistical_analysis"]
        stats = test_result["details"]
        
        # Collect all embedding values
        all_values = []
        for video_id, frames in index.items():
            for frame_id, data in frames.items():
                for patch_id, values in data["patches"].items():
                    all_values.extend(values)
        
        if not all_values:
            test_result["status"] = "skipped"
            return True
        
        all_values = np.array(all_values)
        
        # Calculate statistics
        stats["mean"] = float(np.mean(all_values))
        stats["std"] = float(np.std(all_values))
        stats["min"] = float(np.min(all_values))
        stats["max"] = float(np.max(all_values))
        stats["unique_values"] = len(np.unique(all_values))
        stats["total_values"] = len(all_values)
        
        # Check for reasonable distribution
        all_good = True
        if stats["std"] < 0.01:
            test_result["details"] = "❌ Very low standard deviation - embeddings may be too similar"
            all_good = False
        elif stats["unique_values"] < 100:
            test_result["details"] = "❌ Too few unique values - possible quantization issue"
            all_good = False
        else:
            test_result["details"] = "✓ Embedding distribution looks reasonable"
        
        test_result["status"] = "passed" if all_good else "failed"
        
        print(f"\nTest Status: {test_result['status'].upper()}")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std Dev: {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Unique values: {stats['unique_values']:,} / {stats['total_values']:,}")
        
        return all_good
    
    def export_vespa_data_in_slices(self, num_slices: int = 100) -> Path:
        """Export Vespa data in slices to avoid OOM errors"""
        output_dir = self.output_manager.base_dir / "exports" / "vespa_slices"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🚀 Exporting Vespa data in {num_slices} slices")
        print(f"📁 Output directory: {output_dir}")
        
        slice_files = []
        
        # Export each slice
        for slice_id in range(num_slices):
            print(f"📥 Exporting slice {slice_id}/{num_slices-1}...", end="", flush=True)
            
            slice_file = output_dir / f"slice_{slice_id}.jsonl"
            
            # Run vespa visit command
            cmd = [
                "vespa", "visit", "-t", "local",
                "--slices", str(num_slices),
                "--slice-id", str(slice_id),
                "--selection", "video_frame"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # Write output to file
                with open(slice_file, 'w') as f:
                    f.write(result.stdout)
                
                lines = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
                print(f" ✅ {lines} documents")
                slice_files.append(slice_file)
                
                # Small delay to avoid overwhelming the system
                time.sleep(0.5)
                
            except subprocess.CalledProcessError as e:
                print(f" ❌ Failed: {e.stderr}")
                raise
        
        # Combine all slices
        print("\n🔗 Combining all slices...")
        combined_file = output_dir / "all_documents.jsonl"
        
        with open(combined_file, 'w') as outfile:
            for slice_file in slice_files:
                with open(slice_file, 'r') as infile:
                    outfile.write(infile.read())
        
        # Count total documents
        with open(combined_file, 'r') as f:
            total_lines = sum(1 for _ in f)
        
        print(f"✅ Export complete: {total_lines} total documents")
        print(f"📄 Combined file: {combined_file}")
        
        return combined_file
    
    def run_all_tests(self, exported_file: Optional[str] = None, num_slices: int = 100) -> bool:
        """Run all embedding integrity tests"""
        print("\n" + "=" * 80)
        print("VESPA EMBEDDING INTEGRITY TEST")
        print("=" * 80)
        
        # Export data if no file provided
        if exported_file:
            export_file = Path(exported_file)
            print(f"Using provided export file: {export_file}")
            if not export_file.exists():
                print(f"❌ Export file not found: {export_file}")
                return False
        else:
            print("No export file provided, exporting data from Vespa...")
            export_file = self.export_vespa_data_in_slices(num_slices)
        
        # Build embedding index
        index = self.build_embedding_index(export_file)
        if not index:
            print("❌ No embeddings found in exported data")
            return False
        
        # Run tests
        test_results = []
        test_results.append(self.test_cross_video_uniqueness(index))
        test_results.append(self.test_within_video_variation(index))
        test_results.append(self.test_embedding_structure(index))
        test_results.append(self.test_statistical_analysis(index))
        
        # Save results
        report_file = self.output_manager.get_test_results_dir() / f"embedding_integrity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        all_passed = all(test_results)
        failed_tests = [name for name, result in self.results["tests"].items() 
                       if result["status"] == "failed"]
        
        print(f"\nTotal documents analyzed: {self.results['total_documents']}")
        print(f"Videos analyzed: {len(self.results['videos_analyzed'])}")
        
        if all_passed:
            print("\n✅ ALL TESTS PASSED")
        else:
            print(f"\n❌ {len(failed_tests)} TESTS FAILED:")
            for test in failed_tests:
                print(f"  - {test}")
        
        if self.results["duplicates_found"]:
            print(f"\n⚠️ Found {len(self.results['duplicates_found'])} duplicate embeddings")
        
        print(f"\nDetailed report saved to: {report_file}")
        
        # Don't delete the combined export file - it's in exports directory
        # and may be useful for debugging
        
        return all_passed


def main():
    parser = argparse.ArgumentParser(description="Test embedding integrity in Vespa")
    parser.add_argument("--sample-size", type=int, help="Limit number of documents to analyze")
    parser.add_argument("--exported-file", type=str, help="Path to exported Vespa data file (optional - will export if not provided)")
    parser.add_argument("--num-slices", type=int, default=100, help="Number of slices for data export (default: 100)")
    
    args = parser.parse_args()
    
    tester = EmbeddingIntegrityTester(sample_size=args.sample_size)
    success = tester.run_all_tests(exported_file=args.exported_file, num_slices=args.num_slices)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()