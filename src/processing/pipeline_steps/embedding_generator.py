#!/usr/bin/env python3
"""
Embedding Generation Step

Generates vector embeddings for search backends (Byaldi/Vespa).
"""

import json
import time
import torch
import requests
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image
from binascii import hexlify


class EmbeddingGenerator:
    """Handles embedding generation for different search backends"""
    
    def __init__(self, backend: str = "byaldi"):
        self.backend = backend
        self.supported_backends = ["byaldi", "vespa"]
        self.col_model = None
        self.col_processor = None
        
        # Load configuration
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent.parent))
        from src.tools.config import get_config
        self.config = get_config()
        self.vespa_schema = self.config.get("vespa_schema", "video_frame")
        
        # Setup logging - use the root logger to inherit the file handler from main pipeline
        self.logger = logging.getLogger("VideoIngestionPipeline.EmbeddingGenerator")
        
        if backend not in self.supported_backends:
            raise ValueError(f"Unsupported backend: {backend}. Supported: {self.supported_backends}")
        
        self.logger.info(f"Initialized EmbeddingGenerator with backend: {backend}")
        self.logger.info(f"Using Vespa schema: {self.vespa_schema}")
        
        # Load ColPali model during initialization for Vespa backend
        if backend == "vespa":
            self.logger.info("Loading ColPali model during initialization...")
            self._load_colpali_model()
            if self.col_model and self.col_processor:
                self.logger.info("ColPali model loaded successfully during initialization")
            else:
                self.logger.error("Failed to load ColPali model during initialization")
    
    def generate_embeddings(self, video_data: Dict[str, Any], output_dir: Path = None) -> Dict[str, Any]:
        """Generate embeddings using queue-based processing to avoid large dataset deadlocks"""
        video_id = video_data.get('video_id', 'unknown')
        self.logger.info(f"Starting embedding generation for video: {video_id}")
        self.logger.info(f"Backend: {self.backend}")
        self.logger.info(f"Generating {self.backend} embeddings for: {video_id}")
        
        # Use OutputManager for consistent directory structure
        if output_dir is None:
            from src.utils.output_manager import get_output_manager
            output_manager = get_output_manager()
            output_dir = output_manager.get_processing_dir()
        
        start_time = time.time()
        
        if self.backend == "byaldi":
            self.logger.info(f"Running {self.backend} embeddings for: {video_id}")
            result = self._generate_byaldi_embeddings_queue(video_data, output_dir)
        elif self.backend == "vespa":
            self.logger.info(f"Running {self.backend} embeddings for: {video_id}")
            result = self._generate_vespa_embeddings_queue(video_data, output_dir)
        else:
            result = {"error": f"Unsupported backend: {self.backend}"}
        
        elapsed_time = time.time() - start_time
        
        if "error" in result:
            self.logger.error(f"Embedding generation failed for {video_id}: {result['error']}")
        else:
            doc_count = result.get("total_documents", 0)
            self.logger.info(f"Embedding generation completed for {video_id} in {elapsed_time:.2f}s - {doc_count} documents")
        
        return result
    
    def _generate_vespa_embeddings_queue(self, video_data: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Generate embeddings for Vespa using queue-based processing"""
        video_id = video_data.get('video_id', 'unknown')
        self.logger.info(f"Starting queue-based processing for video: {video_id}")
        
        # Load data from files incrementally to avoid large object issues
        output_dir = Path(video_data.get("output_dir", output_dir))
        
        try:
            # Load keyframes metadata
            keyframes_file = output_dir / "metadata" / f"{video_id}_keyframes.json"
            if not keyframes_file.exists():
                return {"error": "Keyframes file not found"}
                
            with open(keyframes_file, 'r') as f:
                metadata = json.load(f)
            
            keyframes = metadata.get("keyframes", [])
            total_frames = len(keyframes)
            
            self.logger.info(f"Found {total_frames} keyframes to process")
            
            # Check progress from progress file
            progress_file = output_dir / "embeddings" / f"{video_id}_vespa_progress.json"
            if progress_file.exists():
                try:
                    with open(progress_file, 'r') as f:
                        progress_data = json.load(f)
                    if progress_data and progress_data.get("total_documents", 0) > 0:
                        doc_count = progress_data["total_documents"]
                        self.logger.info(f"Found existing progress: {doc_count} documents processed")
                        # Only return early if ALL frames have been processed
                        if doc_count >= total_frames:
                            self.logger.info(f"All {total_frames} frames already processed, skipping")
                            return {
                                "video_id": video_id,
                                "backend": "vespa",
                                "total_documents": doc_count,
                                "status": "completed"
                            }
                        else:
                            self.logger.info(f"Only {doc_count}/{total_frames} frames processed, continuing from where left off")
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(f"Invalid progress file: {e}")
            
            # Check if ColPali model is loaded (should be loaded during initialization)
            if not self.col_model or not self.col_processor:
                return {"error": "ColPali model not loaded - this should not happen"}
            
            # Load descriptions and transcript
            descriptions_file = output_dir / "descriptions" / f"{video_id}.json"
            if descriptions_file.exists():
                with open(descriptions_file, 'r') as f:
                    descriptions_raw = json.load(f)
            else:
                descriptions_raw = {}
            
            transcript_file = output_dir / "transcripts" / f"{video_id}.json"
            if transcript_file.exists():
                with open(transcript_file, 'r') as f:
                    transcript_data = json.load(f)
            else:
                transcript_data = {}
            
            # Process frames in larger queues for better throughput
            queue_size = 50  # Process 50 frames at a time (5x increase for better GPU utilization)
            vespa_docs = []
            
            # Determine starting point based on existing progress
            existing_doc_count = 0
            if progress_file.exists():
                try:
                    with open(progress_file, 'r') as f:
                        progress_data = json.load(f)
                    existing_doc_count = progress_data.get("total_documents", 0)
                    if existing_doc_count > 0:
                        # Just track the count, don't load documents
                        self.logger.info(f"Resuming from frame {existing_doc_count + 1}")
                except (json.JSONDecodeError, KeyError):
                    existing_doc_count = 0
            
            total_processed = existing_doc_count
            
            # Handle transcript data - support both old (dict) and new (list) formats
            if transcript_data:
                if isinstance(transcript_data, list):
                    # New format: list of segments
                    audio_transcript = " ".join([segment.get("text", "") for segment in transcript_data])
                elif isinstance(transcript_data, dict) and "segments" in transcript_data:
                    # Old format: dict with segments key
                    audio_transcript = " ".join([segment.get("text", "") for segment in transcript_data["segments"]])
                else:
                    audio_transcript = ""
            else:
                audio_transcript = ""
            
            creation_timestamp = int(time.time())
            
            self.logger.info(f"Processing {total_frames} frames in queues of {queue_size}")
            
            for queue_start in range(existing_doc_count, total_frames, queue_size):
                queue_end = min(queue_start + queue_size, total_frames)
                frame_queue = keyframes[queue_start:queue_end]
                
                queue_num = queue_start // queue_size + 1
                total_queues = (total_frames + queue_size - 1) // queue_size
                
                self.logger.info(f"Processing queue {queue_num}/{total_queues}: frames {queue_start+1}-{queue_end}")
                
                # Track documents created in this queue
                queue_doc_count = 0
                
                # Process this queue
                for i, keyframe in enumerate(frame_queue):
                    frame_idx = queue_start + i
                    frame_id = str(keyframe["frame_id"])
                    frame_path = Path(keyframe["path"])
                    description = descriptions_raw.get(frame_id, "")
                    timestamp = keyframe.get("timestamp", 0)
                    
                    if frame_idx < 5 or (frame_idx + 1) % 50 == 0:
                        self.logger.info(f"Processing frame {frame_idx+1}/{total_frames}: {frame_id}")
                    
                    if not frame_path.exists():
                        self.logger.warning(f"Frame not found: {frame_path}")
                        continue
                    
                    # Generate ColPali embeddings
                    frame_start = time.time()
                    self.logger.debug(f"About to generate embeddings for frame {frame_id}")
                    colpali_embeddings = self._get_frame_embeddings(frame_path)
                    frame_time = time.time() - frame_start
                    self.logger.debug(f"Finished generating embeddings for frame {frame_id} in {frame_time:.2f}s")
                    
                    if not colpali_embeddings:
                        self.logger.warning(f"Failed to generate embeddings for frame {frame_id}")
                        continue
                    
                    total_processed += 1
                    
                    if frame_idx < 5 or (frame_idx + 1) % 50 == 0:
                        self.logger.info(f"Frame {frame_id}: {frame_time:.2f}s ({total_processed}/{total_frames} completed)")
                    
                    # Generate binary embeddings
                    binary_embeddings = self._generate_binary_embeddings(colpali_embeddings)
                    
                    # Generate float embeddings in correct Vespa format
                    float_embeddings = self._generate_float_embeddings(colpali_embeddings)
                    
                    colpali_binary_cells = []
                    for patch_key, hex_string in binary_embeddings.items():
                        patch_idx = patch_key.replace("patch", "")
                        binary_bytes = bytes.fromhex(hex_string)
                        for v_idx, byte_val in enumerate(binary_bytes[:16]):
                            colpali_binary_cells.append({
                                "address": {"patch": patch_idx, "v": str(v_idx)},
                                "value": int(byte_val) if byte_val < 128 else int(byte_val) - 256
                            })
                    
                    doc_id = f"{video_id}_frame_{frame_id}"
                    
                    vespa_doc = {
                        "put": f"id:video:{self.vespa_schema}::{doc_id}",
                        "fields": {
                            "video_id": video_id,
                            "video_title": video_id,
                            "creation_timestamp": creation_timestamp,
                            "frame_id": int(frame_id),
                            "start_time": float(timestamp),
                            "end_time": float(timestamp + 1.0),
                            "frame_description": description,
                            "audio_transcript": audio_transcript,
                            "colpali_embedding": float_embeddings,
                            "colpali_binary": {"cells": colpali_binary_cells}
                        }
                    }
                    vespa_docs.append(vespa_doc)
                    queue_doc_count += 1  # Track actual documents created
                
                # Feed current queue to Vespa
                # Get only the documents from this current queue (use actual count, not frame count)
                if queue_doc_count > 0:
                    queue_docs = vespa_docs[-queue_doc_count:]  # Get actual docs created in this queue
                    self.logger.info(f"Feeding {len(queue_docs)} documents to Vespa...")
                    self._feed_to_vespa(queue_docs, video_id)
                else:
                    self.logger.warning(f"No documents created in queue {queue_num}, skipping Vespa feed")
                
                # Save progress metadata only (not the actual embeddings)
                progress_data = {
                    "video_id": video_id,
                    "backend": "vespa",
                    "total_documents": total_processed,  # Use actual processed count, not accumulated docs
                    "queues_completed": queue_num,
                    "total_queues": total_queues,
                    "last_processed_frame": queue_end - 1,
                    "created_at": time.time()
                }
                
                # Save to a progress file instead of embeddings file
                progress_file = output_dir / "embeddings" / f"{video_id}_vespa_progress.json"
                progress_file.parent.mkdir(parents=True, exist_ok=True)
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=2)
                
                self.logger.info(f"Queue {queue_num}/{total_queues} completed. Progress saved.")
            
            # All documents have been fed during queue processing
            # No need for final batch feeding
            
            # Final result (without documents to avoid memory issues)
            final_result = {
                "video_id": video_id,
                "backend": "vespa",
                "total_documents": total_processed,  # Use actual processed count
                "total_processed": total_processed,
                "status": "completed",
                "created_at": time.time()
            }
            
            # Save final progress
            with open(progress_file, 'w') as f:
                json.dump(final_result, f, indent=2)
            
            print(f"🎉 Completed! Processed {total_processed}/{total_frames} frames")
            self.logger.info(f"Completed! Processed {total_processed}/{total_frames} frames")
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Queue-based processing failed: {e}")
            print(f"❌ Queue-based processing failed: {e}")
            return {"error": str(e)}

    def _generate_byaldi_embeddings_queue(self, video_data: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Generate embeddings for Byaldi using queue-based processing"""
        # For now, delegate to the original method
        return self._generate_byaldi_embeddings(video_data, output_dir)

    def _generate_byaldi_embeddings(self, video_data: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Generate embeddings for Byaldi backend using ColPali"""
        video_id = video_data.get("video_id", "unknown")
        
        try:
            # Load keyframes and descriptions
            keyframes_data = video_data.get("keyframes", {})
            descriptions_data = video_data.get("descriptions", {})
            
            if not keyframes_data or not descriptions_data:
                print("  ⚠️ Missing keyframes or descriptions data")
                return {"error": "Missing required data"}
            
            # Process keyframes for Byaldi
            documents = []
            keyframes = keyframes_data.get("keyframes", [])
            
            for keyframe in keyframes:
                frame_id = str(keyframe["frame_id"])
                frame_path = keyframe["path"]
                description = descriptions_data.get(frame_id, "")
                
                if Path(frame_path).exists() and description:
                    documents.append({
                        "doc_id": f"{video_id}_frame_{frame_id}",
                        "video_id": video_id,
                        "frame_id": frame_id,
                        "image_path": frame_path,
                        "description": description,
                        "timestamp": keyframe.get("timestamp", 0)
                    })
            
            # Save Byaldi documents
            embeddings_file = output_dir / "embeddings" / f"{video_id}_byaldi.json"
            embeddings_file.parent.mkdir(parents=True, exist_ok=True)
            
            embeddings_data = {
                "video_id": video_id,
                "backend": "byaldi",
                "documents": documents,
                "total_documents": len(documents),
                "created_at": time.time()
            }
            
            with open(embeddings_file, 'w') as f:
                json.dump(embeddings_data, f, indent=2)
            
            print(f"  ✅ Generated Byaldi embeddings: {len(documents)} documents")
            return embeddings_data
            
        except Exception as e:
            print(f"  ❌ Byaldi embedding generation failed: {e}")
            return {"error": str(e)}
    
    def _load_colpali_model(self):
        """Load ColPali model for embedding generation"""
        try:
            self.logger.info("Starting ColPali model loading process...")
            
            self.logger.info("Importing ColPali dependencies...")
            from colpali_engine.models import ColIdefics3, ColIdefics3Processor
            
            # Get model name from config
            self.logger.info("Loading configuration...")
            import sys
            sys.path.append(str(Path(__file__).parent.parent.parent.parent))
            from src.tools.config import get_config
            config = get_config()
            
            colpali_model_name = config.get("colpali_model", "vidore/colsmol-500m")
            
            # Proper device detection with MPS support for Apple Silicon
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"  # For Apple Silicon devices
            else:
                device = "cpu"
            
            # Override with config if specified
            config_device = config.get("device")
            if config_device:
                device = config_device
                self.logger.info(f"Device overridden by config: {device}")
            
            # bfloat16 support is limited; fallback to float32 if needed
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            self.logger.info(f"Model configuration - Name: {colpali_model_name}")
            self.logger.info(f"Device detection - CUDA available: {torch.cuda.is_available()}, MPS available: {torch.backends.mps.is_available()}")
            self.logger.info(f"Using device: {device}, dtype: {dtype}")
            self.logger.info("Downloading/loading ColPali model from Hugging Face...")
            
            # Add more detailed logging during model loading
            self.logger.info(f"About to call ColIdefics3.from_pretrained with device_map={device}")
            
            # Load model with proper device handling
            if device == "mps":
                # For MPS, load to CPU first then move to MPS
                self.col_model = ColIdefics3.from_pretrained(
                    colpali_model_name,
                    torch_dtype=dtype,
                    device_map="cpu"  # Load to CPU first for MPS
                ).eval()
                self.logger.info("Moving model to MPS device...")
                self.col_model = self.col_model.to(device)
            else:
                # For CUDA or CPU, use direct device mapping
                self.col_model = ColIdefics3.from_pretrained(
                    colpali_model_name,
                    torch_dtype=dtype,
                    device_map=device
                ).eval()
            
            self.logger.info("ColIdefics3.from_pretrained completed successfully")
            
            self.logger.info("ColPali model loaded, now loading processor...")
            
            self.col_processor = ColIdefics3Processor.from_pretrained(colpali_model_name)
            
            self.logger.info("ColPali model and processor loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load ColPali model: {e}")
            self.col_model = None
            self.col_processor = None
    
    def _get_frame_embeddings(self, frame_path: Path) -> Optional[List[List[float]]]:
        """Generate ColPali embeddings for a single frame"""
        if not self.col_model or not self.col_processor:
            self.logger.warning("ColPali model or processor not loaded")
            return None
            
        try:
            # Load and process image
            self.logger.debug(f"🖼️  Loading image from {frame_path}")
            keyframe_image = Image.open(frame_path).convert("RGB")
            self.logger.debug(f"📐 Image size: {keyframe_image.size}")
            
            self.logger.debug(f"⚙️  Processing image with ColPali processor...")
            # Use the original working pattern
            batch_images = self.col_processor.process_images([keyframe_image]).to(self.col_model.device)
            self.logger.debug(f"📦 Batch images shape: {batch_images.get('pixel_values').shape if 'pixel_values' in batch_images else 'Unknown'}")
            
            self.logger.debug(f"🧠 Generating embeddings with ColPali model...")
            with torch.no_grad():
                image_embeddings = self.col_model(**batch_images)
            
            self.logger.debug(f"📊 Raw embeddings shape: {image_embeddings.shape}")
            self.logger.debug(f"🔄 Converting embeddings to CPU and list format...")
            # Use the original working pattern that was in the archived code
            result = image_embeddings.cpu().numpy().squeeze(0).tolist()
            self.logger.debug(f"✅ Generated {len(result)} embedding patches with {len(result[0]) if result else 0} dimensions each")
            
            # Log first few values for debugging
            if result and len(result) > 0 and len(result[0]) > 0:
                self.logger.debug(f"🔍 First patch, first 5 values: {result[0][:5]}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings for {frame_path}: {e}")
            print(f"    ❌ Failed to generate embeddings for {frame_path}: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def tensor_to_hex_bfloat16(self, tensor: torch.Tensor) -> str:
        """Convert tensor to hex-encoded bfloat16 format exactly like Vespa docs"""
        import struct
        
        if not tensor.is_floating_point():
            raise ValueError("Input tensor must be of float32 type.")

        def float_to_bfloat16_hex(f: float) -> str:
            packed_float = struct.pack("=f", f)
            bfloat16_bits = struct.unpack("=H", packed_float[2:])[0]
            return format(bfloat16_bits, "04X")

        hex_list = [float_to_bfloat16_hex(float(val)) for val in tensor.flatten()]
        return "".join(hex_list)
    
    def _generate_float_embeddings(self, embeddings: List[List[float]]) -> Dict[int, str]:
        """Generate float embeddings in Vespa hex format exactly like documentation"""
        try:
            embedding_full = {}
            for patch_idx, embedding in enumerate(embeddings):
                # Convert to torch tensor for the hex function
                tensor = torch.tensor(embedding, dtype=torch.float32)
                embedding_full[patch_idx] = self.tensor_to_hex_bfloat16(tensor)
            
            self.logger.debug(f"Generated float embeddings for {len(embedding_full)} patches")
            return embedding_full
            
        except Exception as e:
            self.logger.error(f"Could not generate float embeddings: {e}")
            return {}

    def _generate_binary_embeddings(self, embeddings: List[List[float]]) -> Dict[str, str]:
        """Generate binary embeddings for ColPali using hex encoding"""
        try:
            # Convert all embeddings to numpy array at once
            vectors = np.array(embeddings, dtype=np.float32)
            
            # Binarize: positive values -> 1, negative/zero -> 0, then pack bits
            binarized_token_vectors = np.packbits(np.where(vectors > 0, 1, 0), axis=1).astype(np.int8)
            
            # Convert to hex strings
            vespa_token_feed = {}
            for index in range(len(binarized_token_vectors)):
                vespa_token_feed[f"patch{index}"] = str(
                    hexlify(binarized_token_vectors[index].tobytes()), "utf-8"
                )
            
            return vespa_token_feed
            
        except Exception as e:
            print(f"    ⚠️ Could not generate binary embeddings: {e}")
            return {}
    
    def _feed_to_vespa(self, vespa_docs: List[Dict], video_id: str) -> bool:
        """Feed documents to Vespa instance using batch API for improved performance"""
        try:
            # Use batch feeding endpoint for much better performance
            vespa_batch_url = "http://localhost:8080/document/v1/"
            batch_size = 100  # Process 100 documents at once
            success_count = 0
            
            self.logger.info(f"Feeding {len(vespa_docs)} documents to Vespa for video: {video_id} (batch size: {batch_size})")
            
            # Process documents in batches
            for i in range(0, len(vespa_docs), batch_size):
                batch = vespa_docs[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(vespa_docs) + batch_size - 1) // batch_size
                
                print(f"  📦 Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                
                # Send batch using individual requests but with connection reuse
                batch_success = 0
                for doc in batch:
                    # Extract doc_id from the put field: "id:video:video_frame::doc_id" 
                    doc_id = doc["put"].split("::")[-1]
                    url = f"http://localhost:8080/document/v1/video/{self.vespa_schema}/docid/{doc_id}"
                    
                    response = self._get_session().post(url, json=doc, headers={'Content-Type': 'application/json'})
                    
                    if response.status_code == 200:
                        batch_success += 1
                    else:
                        self.logger.warning(f"Failed to feed document {doc_id}: HTTP {response.status_code}")
                        if response.status_code != 200:
                            print(f"    ⚠️ Failed to feed document {doc_id}: {response.status_code} - {response.text[:100]}")
                
                success_count += batch_success
                print(f"    ✅ Batch {batch_num}: {batch_success}/{len(batch)} documents fed successfully")
            
            self.logger.info(f"Fed {success_count}/{len(vespa_docs)} documents to Vespa")
            print(f"  🎉 Total: {success_count}/{len(vespa_docs)} documents fed successfully")
            return success_count == len(vespa_docs)
            
        except Exception as e:
            self.logger.error(f"Failed to feed documents to Vespa: {e}")
            print(f"  ❌ Failed to feed documents to Vespa: {e}")
            return False
    
    def _get_session(self):
        """Get or create HTTP session with connection pooling"""
        if not hasattr(self, '_session'):
            import requests.adapters
            self._session = requests.Session()
            # Configure connection pooling for better performance
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=20,
                max_retries=3
            )
            self._session.mount('http://', adapter)
            self._session.mount('https://', adapter)
        return self._session
    
    def _process_keyframe_chunk(self, chunk_keyframes: List[Dict], video_data: Dict[str, Any], output_dir: Path, vespa_docs: List[Dict]) -> bool:
        """Process a chunk of keyframes and add them to vespa_docs"""
        try:
            video_id = video_data.get("video_id", "unknown")
            descriptions_data = video_data.get("descriptions", {})
            transcript_data = video_data.get("transcript", {})
            video_title = video_data.get("video_id", "")
            audio_transcript = transcript_data.get("full_text", "")
            creation_timestamp = int(time.time())
            
            # Process in smaller batches within the chunk
            batch_size = 10  # Process 10 frames at a time
            total_processed = 0
            
            for batch_start in range(0, len(chunk_keyframes), batch_size):
                batch_end = min(batch_start + batch_size, len(chunk_keyframes))
                batch_keyframes = chunk_keyframes[batch_start:batch_end]
                batch_docs = []
                
                batch_num = batch_start//batch_size + 1
                total_batches = (len(chunk_keyframes) + batch_size - 1)//batch_size
                
                self.logger.info(f"      Processing batch {batch_num}/{total_batches}: frames {batch_start+1}-{batch_end}")
                print(f"      📦 Processing batch {batch_num}/{total_batches}: frames {batch_start+1}-{batch_end}")
                
                # Process each frame in the batch
                for i, keyframe in enumerate(batch_keyframes):
                    frame_idx = batch_start + i
                    frame_id = str(keyframe["frame_id"])
                    frame_path = Path(keyframe["path"])
                    description = descriptions_data.get("descriptions", {}).get(frame_id, "")
                    timestamp = keyframe.get("timestamp", 0)
                    
                    if frame_idx < 3 or (frame_idx + 1) % 25 == 0:
                        print(f"        🔄 Processing frame {frame_idx+1}: {frame_id}")
                    
                    if not frame_path.exists():
                        self.logger.warning(f"Frame not found: {frame_path}")
                        continue
                    
                    # Generate ColPali embeddings with timing
                    frame_start = time.time()
                    colpali_embeddings = self._get_frame_embeddings(frame_path)
                    frame_time = time.time() - frame_start
                    
                    if not colpali_embeddings:
                        self.logger.warning(f"Failed to generate embeddings for frame {frame_id}")
                        print(f"        ❌ Failed to generate embeddings for frame {frame_id}")
                        continue
                    
                    # Debug: Check embedding shape and content
                    if frame_idx < 3:
                        self.logger.info(f"🔍 DEBUG: Frame {frame_id} embeddings shape: {len(colpali_embeddings)}x{len(colpali_embeddings[0]) if colpali_embeddings else 0}")
                        self.logger.info(f"🔍 DEBUG: First few embedding values: {colpali_embeddings[0][:5] if colpali_embeddings and len(colpali_embeddings) > 0 else 'None'}")
                        print(f"        🔍 DEBUG: Frame {frame_id} embeddings shape: {len(colpali_embeddings)}x{len(colpali_embeddings[0]) if colpali_embeddings else 0}")
                        print(f"        🔍 DEBUG: First few embedding values: {colpali_embeddings[0][:5] if colpali_embeddings and len(colpali_embeddings) > 0 else 'None'}")
                    
                    total_processed += 1
                    
                    # Log progress with timing
                    if frame_idx < 3 or (frame_idx + 1) % 25 == 0:
                        self.logger.info(f"Frame {frame_id}: embedding generated in {frame_time:.2f}s")
                        print(f"        ✅ Frame {frame_id}: {frame_time:.2f}s")
                    
                    # Generate binary embeddings
                    self.logger.info(f"🔢 Generating binary embeddings for frame {frame_id}")
                    binary_embeddings = self._generate_binary_embeddings(colpali_embeddings)
                    self.logger.info(f"🔢 Binary embeddings generated: {len(binary_embeddings)} patches")
                    
                    # Generate float embeddings in correct Vespa format
                    float_embeddings = self._generate_float_embeddings(colpali_embeddings)
                    
                    # Convert binary embeddings to Vespa tensor format with cells
                    colpali_binary_cells = []
                    for patch_key, hex_string in binary_embeddings.items():
                        patch_idx = patch_key.replace("patch", "")
                        binary_bytes = bytes.fromhex(hex_string)
                        for v_idx, byte_val in enumerate(binary_bytes[:16]):
                            colpali_binary_cells.append({
                                "address": {"patch": patch_idx, "v": str(v_idx)},
                                "value": int(byte_val) if byte_val < 128 else int(byte_val) - 256
                            })
                    
                    doc_id = f"{video_id}_frame_{frame_id}"
                    
                    vespa_doc = {
                        "put": f"id:video:{self.vespa_schema}::{doc_id}",
                        "fields": {
                            "video_id": video_id,
                            "video_title": video_title,
                            "creation_timestamp": creation_timestamp,
                            "frame_id": int(frame_id),
                            "start_time": float(timestamp),
                            "end_time": float(timestamp + 1.0),
                            "frame_description": description,
                            "audio_transcript": audio_transcript,
                            "colpali_embedding": float_embeddings,
                            "colpali_binary": {"cells": colpali_binary_cells}
                        }
                    }
                    
                    # Debug: Check tensor cells
                    if frame_idx < 3:
                        self.logger.info(f"🔍 DEBUG: Frame {frame_id} colpali_embedding cells: {len(colpali_embedding_cells)}")
                        self.logger.info(f"🔍 DEBUG: Frame {frame_id} colpali_binary cells: {len(colpali_binary_cells)}")
                        self.logger.info(f"🔍 DEBUG: Binary embedding keys: {list(binary_embeddings.keys())[:3] if binary_embeddings else 'None'}")
                        print(f"        🔍 DEBUG: Frame {frame_id} colpali_embedding cells: {len(colpali_embedding_cells)}")
                        print(f"        🔍 DEBUG: Frame {frame_id} colpali_binary cells: {len(colpali_binary_cells)}")
                        print(f"        🔍 DEBUG: Binary embedding keys: {list(binary_embeddings.keys())[:3] if binary_embeddings else 'None'}")
                    
                    batch_docs.append(vespa_doc)
                    vespa_docs.append(vespa_doc)
                
                # Feed this batch to Vespa immediately
                if batch_docs:
                    self.logger.info(f"      Feeding batch of {len(batch_docs)} documents to Vespa...")
                    print(f"        📤 Feeding {len(batch_docs)} documents to Vespa...")
                    
                    batch_success = self._feed_to_vespa(batch_docs, video_id)
                    if batch_success:
                        self.logger.info(f"      Successfully fed batch.")
                        print(f"        ✅ Batch fed successfully.")
                    else:
                        self.logger.warning(f"      Batch feeding had issues.")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process keyframe chunk: {e}")
            print(f"❌ Failed to process keyframe chunk: {e}")
            return False

    def _generate_vespa_embeddings(self, video_data: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Generate embeddings for Vespa backend with actual ColPali embeddings and feed to Vespa"""
        video_id = video_data.get("video_id", "unknown")
        self.logger.info(f"Generating Vespa embeddings for video: {video_id}")
        
        # Check if embeddings already exist
        embeddings_file = output_dir / "embeddings" / f"{video_id}_vespa.json"
        if embeddings_file.exists():
            try:
                with open(embeddings_file, 'r') as f:
                    existing_data = json.load(f)
                if existing_data and existing_data.get("total_documents", 0) > 0:
                    doc_count = existing_data["total_documents"]
                    self.logger.info(f"Found existing embeddings for {video_id}: {doc_count} documents")
                    print(f"  ✅ Already have embeddings ({doc_count} documents)")
                    
                    # Still need to feed to Vespa if documents don't exist there
                    vespa_docs = existing_data.get("documents", [])
                    if vespa_docs:
                        print(f"  📤 Feeding {len(vespa_docs)} cached documents to Vespa...")
                        self._feed_to_vespa(vespa_docs, video_id)
                    
                    return existing_data
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Invalid embeddings file for {video_id}: {e}")
        
        # Only load ColPali model when we actually need to generate embeddings
        if not self.col_model or not self.col_processor:
            self.logger.info("ColPali model not loaded, starting model loading...")
            print("  🔄 ColPali model not loaded, starting model loading...")
            self._load_colpali_model()
            if not self.col_model or not self.col_processor:
                self.logger.error("Failed to load ColPali model, cannot generate embeddings")
                print("  ❌ Failed to load ColPali model, cannot generate embeddings")
                return {"error": "Failed to load ColPali model"}
        else:
            self.logger.info("ColPali model already loaded, proceeding with embedding generation...")
            print("  ✅ ColPali model already loaded, proceeding with embedding generation...")
        
        try:
            # Load all data from files to avoid large object passing issues
            self.logger.info(f"Loading data from files...")
            
            output_dir = Path(video_data.get("output_dir", output_dir))
            
            # Load keyframes metadata
            keyframes_file = output_dir / "metadata" / f"{video_id}_keyframes.json"
            self.logger.info(f"Loading keyframes from: {keyframes_file}")
            if keyframes_file.exists():
                with open(keyframes_file, 'r') as f:
                    keyframes_data = json.load(f)
            else:
                self.logger.error(f"Keyframes file not found: {keyframes_file}")
                return {"error": "Keyframes file not found"}
            
            # Load descriptions
            descriptions_file = output_dir / "descriptions" / f"{video_id}.json"
            self.logger.info(f"Loading descriptions from: {descriptions_file}")
            if descriptions_file.exists():
                with open(descriptions_file, 'r') as f:
                    descriptions_raw = json.load(f)
                descriptions_data = {"descriptions": descriptions_raw}
            else:
                self.logger.error(f"Descriptions file not found: {descriptions_file}")
                return {"error": "Descriptions file not found"}
            
            # Load transcript
            transcript_file = output_dir / "transcripts" / f"{video_id}.json"
            self.logger.info(f"Loading transcript from: {transcript_file}")
            if transcript_file.exists():
                with open(transcript_file, 'r') as f:
                    transcript_data = json.load(f)
            else:
                self.logger.warning(f"Transcript file not found: {transcript_file}")
                transcript_data = {}
            
            if not keyframes_data:
                self.logger.error(f"Missing keyframes data for video: {video_id}")
                print("  ⚠️ Missing keyframes data")
                return {"error": "Missing keyframes data"}
            
            # Process for Vespa format  
            vespa_docs = []
            keyframes = keyframes_data.get("keyframes", [])
            
            # Process in chunks of 100 frames to avoid memory/threading issues
            chunk_size = 100
            total_chunks = (len(keyframes) + chunk_size - 1) // chunk_size
            
            if len(keyframes) > chunk_size:
                self.logger.info(f"Processing {len(keyframes)} frames in {total_chunks} chunks of {chunk_size}")
                
                # Process in chunks
                for chunk_idx in range(total_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, len(keyframes))
                    chunk_keyframes = keyframes[start_idx:end_idx]
                    
                    self.logger.info(f"📦 Processing chunk {chunk_idx + 1}/{total_chunks}: frames {start_idx+1}-{end_idx}")
                    print(f"📦 Processing chunk {chunk_idx + 1}/{total_chunks}: frames {start_idx+1}-{end_idx}")
                    
                    # Process this chunk
                    chunk_result = self._process_keyframe_chunk(chunk_keyframes, video_data, output_dir, vespa_docs)
                    if not chunk_result:
                        self.logger.error(f"Failed to process chunk {chunk_idx + 1}")
                        return {"error": f"Failed to process chunk {chunk_idx + 1}"}
                    
                    # Save progress after each chunk
                    embeddings_data = {
                        "video_id": video_id,
                        "backend": "vespa", 
                        "documents": vespa_docs,
                        "total_documents": len(vespa_docs),
                        "chunks_completed": chunk_idx + 1,
                        "total_chunks": total_chunks,
                        "created_at": time.time()
                    }
                    
                    embeddings_file = output_dir / "embeddings" / f"{video_id}_vespa.json"
                    embeddings_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(embeddings_file, 'w') as f:
                        json.dump(embeddings_data, f, indent=2)
                    
                    self.logger.info(f"✅ Chunk {chunk_idx + 1}/{total_chunks} completed. Progress saved.")
                    print(f"✅ Chunk {chunk_idx + 1}/{total_chunks} completed. Progress saved.")
                
                self.logger.info(f"🎉 All {total_chunks} chunks completed!")
                print(f"🎉 All {total_chunks} chunks completed!")
                
                # Skip the normal processing since we did it in chunks
                keyframes = []
            video_title = video_data.get("video_id", "")
            audio_transcript = transcript_data.get("full_text", "")
            creation_timestamp = int(time.time())
            
            self.logger.info(f"Starting ColPali embedding generation for {len(keyframes)} frames of video: {video_id}")
            print(f"  🔮 Starting ColPali embedding generation for {len(keyframes)} frames...")
            
            # Process in batches and feed to Vespa concurrently
            batch_size = 10  # Process 10 frames at a time (reduced for better stability)
            total_fed = 0
            total_processed = 0
            
            self.logger.info(f"Processing will be done in batches of {batch_size} frames")
            print(f"  📦 Processing in batches of {batch_size} frames")
            
            for batch_start in range(0, len(keyframes), batch_size):
                batch_end = min(batch_start + batch_size, len(keyframes))
                batch_keyframes = keyframes[batch_start:batch_end]
                batch_docs = []
                
                batch_num = batch_start//batch_size + 1
                total_batches = (len(keyframes) + batch_size - 1)//batch_size
                
                self.logger.info(f"Processing batch {batch_num}/{total_batches}: frames {batch_start+1}-{batch_end}")
                print(f"    📦 Processing batch {batch_num}/{total_batches}: frames {batch_start+1}-{batch_end}")
                
                # Process each frame in the batch
                for i, keyframe in enumerate(batch_keyframes):
                    frame_idx = batch_start + i
                    frame_id = str(keyframe["frame_id"])
                    frame_path = Path(keyframe["path"])
                    description = descriptions_data.get(frame_id, "")
                    timestamp = keyframe.get("timestamp", 0)
                    
                    self.logger.info(f"Processing frame {frame_idx+1}/{len(keyframes)}: {frame_id}")
                    if frame_idx < 5 or (frame_idx + 1) % 50 == 0:
                        print(f"      🔄 Processing frame {frame_idx+1}/{len(keyframes)}: {frame_id}")
                    
                    if not frame_path.exists():
                        self.logger.warning(f"Frame not found: {frame_path}")
                        print(f"      ⚠️ Frame {frame_id}: file not found")
                        continue
                    
                    # Generate ColPali embeddings with timing
                    self.logger.info(f"Generating ColPali embeddings for frame {frame_id}...")
                    frame_start = time.time()
                    colpali_embeddings = self._get_frame_embeddings(frame_path)
                    frame_time = time.time() - frame_start
                    
                    if not colpali_embeddings:
                        self.logger.warning(f"Failed to generate embeddings for frame {frame_id}")
                        print(f"      ⚠️ Frame {frame_id}: embedding failed")
                        continue
                    
                    total_processed += 1
                    
                    # Log progress with timing
                    if (frame_idx + 1) % 10 == 0 or frame_idx < 5:
                        self.logger.info(f"Frame {frame_id}: embedding generated in {frame_time:.2f}s ({total_processed}/{len(keyframes)} completed)")
                        print(f"      ✅ Frame {frame_id}: {frame_time:.2f}s ({total_processed}/{len(keyframes)} completed)")
                    
                    # Generate binary embeddings
                    binary_embeddings = self._generate_binary_embeddings(colpali_embeddings)
                    
                    # Convert embeddings to Vespa tensor format with cells
                    colpali_embedding_cells = []
                    for patch_idx, embedding in enumerate(colpali_embeddings):
                        for v_idx, value in enumerate(embedding):
                            colpali_embedding_cells.append({
                                "address": {"patch": str(patch_idx), "v": str(v_idx)},
                                "value": float(value)
                            })
                    
                    # Convert binary embeddings to Vespa tensor format with cells
                    # Schema expects tensor<int8>(patch{}, v[16]) - 16 int8 values per patch
                    colpali_binary_cells = []
                    for patch_key, hex_string in binary_embeddings.items():
                        patch_idx = patch_key.replace("patch", "")
                        # Convert hex to bytes (each hex pair = 1 byte = 1 int8 value)
                        binary_bytes = bytes.fromhex(hex_string)
                        for v_idx, byte_val in enumerate(binary_bytes[:16]):  # Take first 16 bytes 
                            colpali_binary_cells.append({
                                "address": {"patch": patch_idx, "v": str(v_idx)},
                                "value": int(byte_val) if byte_val < 128 else int(byte_val) - 256  # Convert to signed int8
                            })
                    
                    doc_id = f"{video_id}_frame_{frame_id}"
                    
                    vespa_doc = {
                        "put": f"id:video:{self.vespa_schema}::{doc_id}",
                        "fields": {
                            "video_id": video_id,
                            "video_title": video_title,
                            "creation_timestamp": creation_timestamp,
                            "frame_id": int(frame_id),
                            "start_time": float(timestamp),
                            "end_time": float(timestamp + 1.0),  # Estimate 1 second duration
                            "frame_description": description,
                            "audio_transcript": audio_transcript,
                            "colpali_embedding": float_embeddings,
                            "colpali_binary": {"cells": colpali_binary_cells}
                        }
                    }
                    batch_docs.append(vespa_doc)
                    vespa_docs.append(vespa_doc)
                
                # Feed this batch to Vespa immediately
                if batch_docs:
                    self.logger.info(f"Feeding batch of {len(batch_docs)} documents to Vespa...")
                    print(f"      📤 Feeding {len(batch_docs)} documents to Vespa...")
                    
                    batch_success = self._feed_to_vespa(batch_docs, video_id)
                    if batch_success:
                        total_fed += len(batch_docs)
                        self.logger.info(f"Successfully fed batch. Total fed: {total_fed}/{len(vespa_docs)}")
                        print(f"      ✅ Batch fed successfully. Total: {total_fed}")
                    else:
                        self.logger.warning(f"Batch feeding had issues. Total fed so far: {total_fed}")
                
                # Save progress to file after each batch
                embeddings_data = {
                    "video_id": video_id,
                    "backend": "vespa",
                    "documents": vespa_docs,
                    "total_documents": len(vespa_docs),
                    "created_at": time.time()
                }
                
                embeddings_file = output_dir / "embeddings" / f"{video_id}_vespa.json"
                embeddings_file.parent.mkdir(parents=True, exist_ok=True)
                with open(embeddings_file, 'w') as f:
                    json.dump(embeddings_data, f, indent=2)
                
                self.logger.info(f"Progress saved: {len(vespa_docs)} documents processed")
            
            if not vespa_docs:
                self.logger.error(f"No valid documents generated for video: {video_id}")
                print("  ❌ No valid documents generated")
                return {"error": "No valid documents generated"}
            
            # Final save and summary
            embeddings_data = {
                "video_id": video_id,
                "backend": "vespa",
                "documents": vespa_docs,
                "total_documents": len(vespa_docs),
                "total_fed_to_vespa": total_fed,
                "created_at": time.time()
            }
            
            embeddings_file = output_dir / "embeddings" / f"{video_id}_vespa.json"
            embeddings_file.parent.mkdir(parents=True, exist_ok=True)
            with open(embeddings_file, 'w') as f:
                json.dump(embeddings_data, f, indent=2)
            
            self.logger.info(f"✅ Completed: {len(vespa_docs)} documents generated, {total_fed} fed to Vespa")
            print(f"  ✅ Completed: {len(vespa_docs)} documents generated, {total_fed} fed to Vespa")
                
            return embeddings_data
            
        except Exception as e:
            self.logger.error(f"Vespa embedding generation failed for video {video_id}: {e}")
            print(f"  ❌ Vespa embedding generation failed: {e}")
            return {"error": str(e)}