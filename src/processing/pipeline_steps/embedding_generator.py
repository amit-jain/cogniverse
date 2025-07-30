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
try:
    from .colqwen_audio_processor import ColQwenAudioEnabledProcessor
except ImportError:
    ColQwenAudioEnabledProcessor = None


class EmbeddingGenerator:
    """Handles embedding generation for different search backends"""
    
    def __init__(self, backend: str = "byaldi", embedding_type: str = None):
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
        
        # Set embedding type (will be used for direct video vs frame-based processing)
        self.embedding_type = embedding_type or self.config.get("embedding_type", "frame_based")
        
        # Setup logging - use the root logger to inherit the file handler from main pipeline
        self.logger = logging.getLogger("VideoIngestionPipeline.EmbeddingGenerator")
        
        if backend not in self.supported_backends:
            raise ValueError(f"Unsupported backend: {backend}. Supported: {self.supported_backends}")
        
        self.logger.info(f"Initialized EmbeddingGenerator with backend: {backend}")
        self.logger.info(f"Using embedding type: {self.embedding_type}")
        self.logger.info(f"Using Vespa schema: {self.vespa_schema}")
        
        # Load appropriate model during initialization for Vespa backend
        if backend == "vespa":
            if self.embedding_type in ["direct_video", "direct_video_segment", "direct_video_frame", "direct_video_global", "direct_video_global_large"]:
                self.logger.info("Loading multimodal model for direct video processing...")
                self._load_multimodal_model()
                if self.embedding_type.startswith("direct_video") and "videoprism" in str(self.config.get("colpali_model", "")).lower():
                    # VideoPrism doesn't use col_model/col_processor
                    self.logger.info("VideoPrism model loaded successfully during initialization")
                elif self.col_model and self.col_processor:
                    self.logger.info("Multimodal model loaded successfully during initialization")
                else:
                    self.logger.error("Failed to load multimodal model during initialization")
            else:
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
            if self.embedding_type in ["direct_video", "direct_video_segment", "direct_video_frame", "direct_video_global", "direct_video_global_large"]:
                self.logger.info(f"Using {self.embedding_type} processing for: {video_id}")
                result = self._generate_direct_video_embeddings(video_data, output_dir)
            else:
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
            # Check if FPS extraction is being used
            extraction_method = self.config.get("pipeline_config.keyframe_extraction_method", "histogram")
            if extraction_method == "fps":
                # Try FPS descriptions first
                fps_descriptions_file = output_dir / "descriptions" / f"{video_id}_fps.json"
                if fps_descriptions_file.exists():
                    descriptions_file = fps_descriptions_file
                else:
                    descriptions_file = output_dir / "descriptions" / f"{video_id}.json"
            else:
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
                        # patch_key is now an integer, not a string
                        patch_idx = str(patch_key)
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
                            "colpali_embedding": float_embeddings,  # Dict with patch indices and hex values
                            "colpali_binary": binary_embeddings      # Dict with patch indices and hex values
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
    
    def _load_multimodal_model(self):
        """Load multimodal model for direct video processing (e.g., ColQwen, VideoPrism)"""
        try:
            self.logger.info("Starting multimodal model loading process...")
            
            # Get model name from config
            from src.tools.config import get_config
            config = get_config()
            active_profile = config.get_active_profile()
            profiles = config.get("video_processing_profiles", {})
            
            if active_profile and active_profile in profiles:
                model_name = profiles[active_profile].get("embedding_model", "vidore/colqwen-omni-v0.1")
            else:
                model_name = self.config.get("colpali_model", "vidore/colqwen-omni-v0.1")
            
            self.logger.info(f"Loading multimodal model: {model_name}")
            
            if "videoprism" in model_name.lower():
                # Load VideoPrism model
                self.logger.info("Loading VideoPrism model...")
                from src.processing.pipeline_steps.videoprism_loader import get_videoprism_loader
                # Pass the profile config to VideoPrism loader
                profile_config = self.config
                self.videoprism_loader = get_videoprism_loader(model_name, profile_config)
                self.videoprism_loader.load_model()
                self.col_model = None  # Not using ColPali model
                self.col_processor = None
                self.logger.info(f"VideoPrism model loaded: {model_name}")
                
            elif "colqwen" in model_name.lower():
                # Load ColQwen-specific dependencies
                self.logger.info("Loading ColQwen dependencies...")
                if "omni" in model_name.lower():
                    from colpali_engine.models import ColQwen2_5Omni, ColQwen2_5OmniProcessor
                    model_class = ColQwen2_5Omni
                    processor_class = ColQwen2_5OmniProcessor
                else:
                    from colpali_engine.models import ColQwen2, ColQwen2Processor
                    model_class = ColQwen2
                    processor_class = ColQwen2Processor
                
                # Device detection
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
                
                # Override with config if specified
                config_device = self.config.get("device")
                if config_device:
                    device = config_device
                
                dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                
                self.logger.info(f"Using device: {device}, dtype: {dtype}")
                
                # Load model
                # Check for flash attention support
                try:
                    from transformers.utils import is_flash_attn_2_available
                    attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
                except:
                    attn_implementation = None
                
                if device == "mps":
                    # MPS doesn't support flash attention
                    attn_implementation = None
                    
                # Load model with appropriate attention implementation
                self.logger.info(f"Loading model with attention: {attn_implementation}")
                
                # Load ColQwen model directly as shown in documentation
                self.col_model = model_class.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
                    device_map=device,
                    attn_implementation=attn_implementation if attn_implementation and device != "mps" else None
                ).eval()
                
                # Use audio-enabled processor for ColQwen-Omni if available
                if "colqwen-omni" in model_name.lower() and ColQwenAudioEnabledProcessor:
                    self.col_processor = ColQwenAudioEnabledProcessor.from_pretrained(model_name)
                    self.logger.info("ColQwen model and audio-enabled processor loaded successfully")
                else:
                    self.col_processor = processor_class.from_pretrained(model_name)
                    self.logger.info("ColQwen model and processor loaded successfully")
            else:
                # For other models, log error and don't load
                self.logger.error(f"Unknown multimodal model type: {model_name}")
                self.col_model = None
                self.col_processor = None
                
        except ImportError as e:
            self.logger.error(f"Failed to import dependencies for {model_name}: {e}")
            self.col_model = None
            self.col_processor = None
            if hasattr(self, 'videoprism_loader'):
                self.videoprism_loader = None
        except Exception as e:
            self.logger.error(f"Failed to load multimodal model {model_name}: {e}")
            self.col_model = None
            self.col_processor = None
            if hasattr(self, 'videoprism_loader'):
                self.videoprism_loader = None
    
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
            # Convert to float32 first to handle BFloat16
            result = image_embeddings.cpu().to(torch.float32).numpy().squeeze(0).tolist()
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
        """Generate float embeddings in Vespa hex format exactly like documentation
        
        IMPORTANT: Vespa stores tensor embeddings in a nested structure:
        - The returned dict {0: "hex", 1: "hex", ...} gets stored as:
          {
            "type": "tensor<bfloat16>(patch{}, v[128])",
            "blocks": {
              "0": "hex_value",
              "1": "hex_value",
              ...
            }
          }
        - When querying via document API, you'll see 2 top-level keys: "type" and "blocks"
        - The actual patches are inside "blocks" (e.g., 5474 patches for ColQwen)
        - Don't confuse the 2 top-level keys with the actual patch count!
        """
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
            
            # Convert to hex strings with integer keys
            vespa_token_feed = {}
            for index in range(len(binarized_token_vectors)):
                vespa_token_feed[index] = str(
                    hexlify(binarized_token_vectors[index].tobytes()), "utf-8"
                )
            
            return vespa_token_feed
            
        except Exception as e:
            print(f"    ⚠️ Could not generate binary embeddings: {e}")
            return {}
    
    def _feed_to_vespa(self, vespa_docs: List[Dict], video_id: str) -> bool:
        """Feed documents to Vespa instance using pyvespa"""
        try:
            # Import pyvespa and connect
            from vespa.application import Vespa
            app = Vespa(url="http://localhost", port=8080)
            
            batch_size = 100  # Process 100 documents at once
            success_count = 0
            
            self.logger.info(f"Feeding {len(vespa_docs)} documents to Vespa for video: {video_id} (batch size: {batch_size})")
            
            # Process documents in batches
            for i in range(0, len(vespa_docs), batch_size):
                batch = vespa_docs[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(vespa_docs) + batch_size - 1) // batch_size
                
                print(f"  📦 Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                
                # Convert documents to pyvespa format
                feed_batch = []
                for doc in batch:
                    # Extract doc_id from the put field: "id:video:video_frame::doc_id" 
                    doc_id = doc["put"].split("::")[-1]
                    fields = doc.get("fields", {})
                    
                    # Debug: Log document being sent for global embeddings
                    if "global" in self.vespa_schema:
                        self.logger.info(f"📤 Feeding global doc {doc_id} to Vespa:")
                        self.logger.info(f"  Fields: {list(fields.keys())}")
                        if "embedding" in fields:
                            emb = fields["embedding"]
                            if isinstance(emb, list):
                                self.logger.info(f"  embedding: list of {len(emb)} floats")
                                self.logger.info(f"  First 5 values: {emb[:5]}")
                            else:
                                self.logger.info(f"  embedding type: {type(emb)}")
                        else:
                            self.logger.info(f"  ⚠️ embedding field MISSING!")
                        
                        if "embedding_binary" in fields:
                            emb_bin = fields["embedding_binary"]
                            if isinstance(emb_bin, list):
                                self.logger.info(f"  embedding_binary: list of {len(emb_bin)} int8")
                                self.logger.info(f"  First 5 values: {emb_bin[:5]}")
                            else:
                                self.logger.info(f"  embedding_binary type: {type(emb_bin)}")
                        else:
                            self.logger.info(f"  ⚠️ embedding_binary field MISSING!")
                    
                    # Add to batch in pyvespa format
                    feed_batch.append({
                        "id": doc_id,
                        "fields": fields
                    })
                
                # Feed batch using pyvespa
                try:
                    # Track results for this batch
                    batch_results = []
                    batch_success = 0
                    
                    def callback(response, doc_id):
                        """Callback to handle feed results"""
                        nonlocal batch_success
                        if response.status_code == 200:
                            batch_success += 1
                        else:
                            self.logger.warning(f"Failed to feed document {doc_id}: HTTP {response.status_code}")
                            if "global" in self.vespa_schema:
                                try:
                                    self.logger.warning(f"Response: {response.json}")
                                except:
                                    self.logger.warning(f"Response: {response.text[:500]}")
                    
                    # Use feed_iterable with the batch
                    app.feed_iterable(
                        iter=feed_batch,
                        schema=self.vespa_schema,
                        namespace="video",
                        callback=callback
                    )
                    
                    success_count += batch_success
                    print(f"    ✅ Batch {batch_num}: {batch_success}/{len(batch)} documents fed successfully")
                    
                except Exception as e:
                    self.logger.error(f"Batch feeding failed: {e}")
                    print(f"    ❌ Batch {batch_num} failed: {e}")
            
            self.logger.info(f"Fed {success_count}/{len(vespa_docs)} documents to Vespa")
            print(f"  🎉 Total: {success_count}/{len(vespa_docs)} documents fed successfully")
            return success_count == len(vespa_docs)
            
        except Exception as e:
            self.logger.error(f"Failed to feed documents to Vespa: {e}")
            print(f"  ❌ Failed to feed documents: {e}")
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
            
            # TEMPORARY: Limit to 10 frames for testing
            max_frames = 10
            if len(chunk_keyframes) > max_frames:
                self.logger.warning(f"TEMPORARY: Limiting to {max_frames} frames for testing")
                chunk_keyframes = chunk_keyframes[:max_frames]
            
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
                        # patch_key is now an integer, not a string
                        patch_idx = str(patch_key)
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
                            "colpali_embedding": float_embeddings,  # Dict with patch indices and hex values
                            "colpali_binary": binary_embeddings      # Dict with patch indices and hex values
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
            # Check if FPS extraction is being used
            extraction_method = self.config.get("pipeline_config.keyframe_extraction_method", "histogram")
            if extraction_method == "fps":
                # Try FPS descriptions first
                fps_descriptions_file = output_dir / "descriptions" / f"{video_id}_fps.json"
                if fps_descriptions_file.exists():
                    descriptions_file = fps_descriptions_file
                    self.logger.info(f"Using FPS-remapped descriptions from: {descriptions_file}")
                else:
                    descriptions_file = output_dir / "descriptions" / f"{video_id}.json"
                    self.logger.info(f"FPS descriptions not found, using regular descriptions from: {descriptions_file}")
            else:
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
                    
                    # Generate embeddings using existing methods
                    float_embeddings = self._generate_float_embeddings(colpali_embeddings)
                    binary_embeddings = self._generate_binary_embeddings(colpali_embeddings)
                    
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
                            "colpali_embedding": float_embeddings,  # Dict with patch indices and hex values
                            "colpali_binary": binary_embeddings      # Dict with patch indices and hex values
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
    
    def _generate_direct_video_embeddings(self, video_data: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Generate embeddings for direct video processing (ColQwen for segments, VideoPrism for frames)"""
        video_id = video_data.get('video_id', 'unknown')
        self.logger.info(f"Starting direct video embedding generation for: {video_id}")
        self.logger.info(f"Embedding type: {self.embedding_type}")
        
        # Get video path
        video_path = Path(video_data.get("video_path", ""))
        if not video_path or not video_path.exists():
            # Try to find video file in video directory
            from src.tools.config import get_config
            config = get_config()
            video_dir = Path(config.get("video_data_dir", "data/videos"))
            video_files = list(video_dir.glob(f"{video_id}.*"))
            if video_files:
                video_path = video_files[0]
            else:
                return {"error": f"Video file not found for {video_id}"}
        
        self.logger.info(f"Processing video file: {video_path}")
        
        try:
            # Load video data
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            self.logger.info(f"Video properties - FPS: {fps}, Total frames: {total_frames}, Duration: {duration:.2f}s")
            
            vespa_docs = []
            
            if self.embedding_type == "direct_video_segment":
                # ColQwen-style: Process video in segments
                max_patches = self.config.get("model_specific", {}).get("max_patches", 1024)
                segment_duration = self.config.get("model_specific", {}).get("segment_duration", 30.0)
                num_segments = max(1, int(np.ceil(duration / segment_duration)))
                
                # Process all segments
                
                self.logger.info(f"Using segment-based processing: {num_segments} segments of {segment_duration}s")
                
                for segment_idx in range(num_segments):
                    start_time = segment_idx * segment_duration
                    end_time = min((segment_idx + 1) * segment_duration, duration)
                
                    self.logger.info(f"Processing segment {segment_idx + 1}/{num_segments}: {start_time:.1f}s - {end_time:.1f}s")
                    
                    # Process video segment with the multimodal model
                    if hasattr(self, 'videoprism_loader') and self.videoprism_loader:
                        # Use VideoPrism loader - process segment
                        result = self.videoprism_loader.process_video_segment(video_path, start_time, end_time)
                        if result:
                            embeddings_np = result["embeddings"]
                        else:
                            self.logger.warning(f"Failed to process segment {segment_idx}")
                            continue
                    else:
                        # Use ColQwen-Omni model - cut video segment using ffmpeg
                        with torch.no_grad():
                            import tempfile
                            import subprocess
                            import os
                            
                            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                                tmp_path = tmp_file.name
                            
                            try:
                                # Skip segments that are too short (less than 1 second)
                                segment_duration_actual = end_time - start_time
                                if segment_duration_actual < 1.0:
                                    self.logger.warning(f"Skipping segment {segment_idx}: too short ({segment_duration_actual:.3f}s)")
                                    continue
                                
                                # For short segments or last segments, use re-encoding to ensure video stream is preserved
                                # This is slower but more reliable for edge cases
                                use_reencode = segment_duration_actual < 5.0 or segment_idx == num_segments - 1
                                
                                # Use ffmpeg to extract video segment with audio
                                if use_reencode:
                                    cmd = [
                                        'ffmpeg', '-i', str(video_path),
                                        '-ss', str(start_time),
                                        '-t', str(segment_duration_actual),
                                        '-c:v', 'libx264', '-preset', 'ultrafast',  # Fast encoding
                                        '-c:a', 'copy',  # Copy audio
                                        '-y',  # Overwrite output
                                        tmp_path
                                    ]
                                else:
                                    cmd = [
                                        'ffmpeg', '-i', str(video_path),
                                        '-ss', str(start_time),
                                        '-t', str(segment_duration_actual),
                                        '-c', 'copy',  # Copy codecs for speed
                                        '-y',  # Overwrite output
                                        tmp_path
                                    ]
                                
                                self.logger.info(f"Extracting segment: {' '.join(cmd)}")
                                subprocess.run(cmd, check=True, capture_output=True)
                                
                                # Process video segment file
                                if hasattr(self.col_processor, 'process_videos_with_audio'):
                                    # Use our custom audio-enabled processor
                                    self.logger.info("Using audio-enabled processor for segment")
                                    batch_inputs = self.col_processor.process_videos_with_audio([tmp_path]).to(self.col_model.device)
                                else:
                                    # Use standard processor
                                    batch_inputs = self.col_processor.process_videos([tmp_path]).to(self.col_model.device)
                                
                                embeddings = self.col_model(**batch_inputs)
                                
                            finally:
                                # Clean up temporary file
                                if os.path.exists(tmp_path):
                                    os.unlink(tmp_path)
                                    
                        # Convert to float32 first to handle BFloat16
                        embeddings_np = embeddings.cpu().to(torch.float32).numpy()
                    
                    
                    # For ColQwen, embeddings might be per-frame or aggregated
                    # We need to determine the actual output shape
                    self.logger.info(f"Embeddings shape: {embeddings_np.shape}")
                    
                    # Generate float and binary embeddings - use EXACT same format as documentation!
                    # First check the shape of embeddings
                    self.logger.info(f"Raw embeddings shape: {embeddings_np.shape}")
                    
                    # If embeddings are 2D (num_patches, embedding_dim), use directly
                    # If embeddings are 3D or higher, we need to flatten appropriately
                    if len(embeddings_np.shape) == 2:
                        # Standard 2D embeddings (num_patches, embedding_dim)
                        embeddings_list = embeddings_np.tolist()
                    elif len(embeddings_np.shape) == 3:
                        # Might be (1, num_patches, embedding_dim) - squeeze first dimension
                        embeddings_np = embeddings_np.squeeze(0)
                        embeddings_list = embeddings_np.tolist()
                    else:
                        # Flatten to 2D
                        embeddings_np = embeddings_np.reshape(-1, embeddings_np.shape[-1])
                        embeddings_list = embeddings_np.tolist()
                        self.logger.info(f"Flattened embeddings to shape: {embeddings_np.shape}")
                    
                    # Generate embeddings using existing methods that work for ColPali
                    float_embeddings = self._generate_float_embeddings(embeddings_list)
                    binary_embeddings = self._generate_binary_embeddings(embeddings_list)
                    
                    # Create Vespa document for this segment with proper metadata
                    doc_id = f"{video_id}_segment_{segment_idx}"
                    
                    # Add segment description with timing info
                    segment_description = f"Video segment {segment_idx + 1}/{num_segments} ({start_time:.1f}s-{end_time:.1f}s)"
                    
                    # Log the actual number of patches being stored
                    self.logger.info(f"Storing {len(float_embeddings)} patches for segment {segment_idx}")
                    
                    # Determine schema field names based on which schema we're using
                    self.logger.info(f"Creating document for schema: {self.vespa_schema}")
                    if self.vespa_schema == "video_colqwen":
                        # Using ColQwen schema - minimal fields, no frame_description or audio_transcript
                        vespa_doc = {
                            "put": f"id:video:{self.vespa_schema}::{doc_id}",
                            "fields": {
                                "video_id": video_id,
                                "video_title": video_id,
                                "creation_timestamp": int(time.time()),
                                "start_time": float(start_time),
                                "end_time": float(end_time),
                                "embedding": float_embeddings,        # Dict with patch indices and hex values
                                "embedding_binary": binary_embeddings, # Dict with patch indices and hex values
                                # Segment metadata
                                "segment_id": segment_idx,
                                "total_segments": num_segments,
                                "segment_duration": float(end_time - start_time)
                            }
                        }
                    elif "videoprism" in self.vespa_schema:
                        # Using VideoPrism schema - different field names
                        vespa_doc = {
                            "put": f"id:video:{self.vespa_schema}::{doc_id}",
                            "fields": {
                                "video_id": video_id,
                                "video_title": video_id,
                                "creation_timestamp": int(time.time()),
                                "frame_id": segment_idx,
                                "start_time": float(start_time),
                                "end_time": float(end_time),
                                "embedding": float_embeddings,        # Dict with patch indices and float values
                                "embedding_binary": binary_embeddings, # Dict with patch indices and hex values
                            }
                        }
                    else:
                        # Using original schema with fixed field names
                        vespa_doc = {
                            "put": f"id:video:{self.vespa_schema}::{doc_id}",
                            "fields": {
                                "video_id": video_id,
                                "video_title": video_id,
                                "creation_timestamp": int(time.time()),
                                "frame_id": segment_idx,
                                "start_time": float(start_time),
                                "end_time": float(end_time),
                                "frame_description": segment_description,
                                "audio_transcript": "",   # Will be populated if audio processing is enabled
                                "colpali_embedding": float_embeddings,
                                "colpali_binary": binary_embeddings,
                                "embedding_type": self.embedding_type,
                                # Segment metadata  
                                "segment_id": segment_idx,
                                "total_segments": num_segments,
                                "segment_duration": float(end_time - start_time)
                            }
                        }
                    
                    vespa_docs.append(vespa_doc)
                    
                    # Debug: log first document structure
                    if segment_idx == 0:
                        self.logger.info(f"First document structure: {list(vespa_doc['fields'].keys())}")
                    
                    self.logger.info(f"Created document for segment {segment_idx}")
                    
                    # Feed this segment to Vespa immediately
                    self.logger.info(f"Feeding segment {segment_idx} to Vespa immediately...")
                    if self._feed_to_vespa([vespa_doc], video_id):
                        self.logger.info(f"✅ Successfully fed segment {segment_idx} to Vespa")
                    else:
                        self.logger.warning(f"⚠️ Failed to feed segment {segment_idx} to Vespa")
                
            elif self.embedding_type == "direct_video_frame":
                # VideoPrism: Process entire video at once
                sampling_fps = self.config.get("model_specific", {}).get("sampling_fps", 1.0)
                
                self.logger.info(f"Processing entire video with VideoPrism")
                self.logger.info(f"  Video duration: {duration:.1f}s, Sampling FPS: {sampling_fps}")
                
                # Process entire video at once
                result = self.videoprism_loader.process_entire_video(video_path, sampling_fps)
                
                # Create single document with all patches
                doc_id = f"{video_id}_videoprism"
                
                vespa_doc = {
                    "put": f"id:video:{self.vespa_schema}::{doc_id}",
                    "fields": {
                        "video_id": video_id,
                        "video_title": video_id,
                        "creation_timestamp": int(time.time()),
                        "frame_id": 0,  # Single document for entire video
                        "start_time": 0.0,
                        "end_time": float(duration),
                        "embedding": result["float_embeddings"],  # Dict with cells
                        "embedding_binary": result["binary_embeddings"],  # Dict with cells
                    }
                }
                
                vespa_docs.append(vespa_doc)
                
                self.logger.info(f"Created VideoPrism document with {result['num_patches']} patches")
                self.logger.info(f"  Sampled {result['num_frames_sampled']} frames at {result['sampling_fps']} FPS")
            
            elif self.embedding_type in ["direct_video_global", "direct_video_global_large"]:
                # VideoPrism Global (LVT): Process entire video into single global embedding
                sampling_fps = self.config.get("model_specific", {}).get("sampling_fps", 1.0)
                
                self.logger.info(f"Processing entire video with VideoPrism LVT for global embedding")
                self.logger.info(f"  Video duration: {duration:.1f}s, Sampling FPS: {sampling_fps}")
                
                # Process entire video at once
                result = self.videoprism_loader.process_entire_video(video_path, sampling_fps)
                
                # Create single document with global embedding
                doc_id = f"{video_id}_global"
                
                # Debug: Log the embeddings we're about to index
                float_emb = result["float_embeddings"]
                binary_emb = result["binary_embeddings"]
                
                self.logger.info(f"Global embeddings to index:")
                if isinstance(float_emb, list):
                    self.logger.info(f"  Float embedding: list of {len(float_emb)} values")
                    self.logger.info(f"  First 5 float values: {float_emb[:5]}")
                else:
                    self.logger.info(f"  Float embedding type: {type(float_emb)}, shape: {getattr(float_emb, 'shape', 'N/A')}")
                
                if isinstance(binary_emb, np.ndarray):
                    binary_list = binary_emb.tolist()
                    self.logger.info(f"  Binary embedding: numpy array shape {binary_emb.shape}, converting to list of {len(binary_list)}")
                    self.logger.info(f"  First 5 binary values: {binary_list[:5]}")
                elif isinstance(binary_emb, list):
                    self.logger.info(f"  Binary embedding: list of {len(binary_emb)} values")
                    self.logger.info(f"  First 5 binary values: {binary_emb[:5]}")
                else:
                    self.logger.info(f"  Binary embedding type: {type(binary_emb)}")
                
                vespa_doc = {
                    "put": f"id:video:{self.vespa_schema}::{doc_id}",
                    "fields": {
                        "video_id": video_id,
                        "video_title": video_id,
                        "creation_timestamp": int(time.time()),
                        "frame_id": 0,  # Single document for entire video
                        "start_time": 0.0,
                        "end_time": float(duration),
                        "embedding": float_emb if isinstance(float_emb, list) else float_emb.tolist() if hasattr(float_emb, 'tolist') else float_emb,
                        "embedding_binary": binary_emb.tolist() if isinstance(binary_emb, np.ndarray) else binary_emb,
                    }
                }
                
                vespa_docs.append(vespa_doc)
                
                self.logger.info(f"Created VideoPrism global document")
                self.logger.info(f"  Sampled {result['num_frames_sampled']} frames at {result['sampling_fps']} FPS")
                self.logger.info(f"  Global embedding dimension: {len(result['float_embeddings']) if isinstance(result['float_embeddings'], list) else result['float_embeddings'].shape}")
                
                # Debug: Print the actual document being created
                self.logger.info(f"\n🔍 DEBUG: VideoPrism Global Document for {video_id}:")
                self.logger.info(f"  Document ID: {doc_id}")
                self.logger.info(f"  Fields in document: {list(vespa_doc['fields'].keys())}")
                
                # Check embedding field
                emb_field = vespa_doc['fields'].get('embedding')
                if emb_field is not None:
                    if isinstance(emb_field, list):
                        self.logger.info(f"  embedding field: list of {len(emb_field)} floats")
                        self.logger.info(f"  First 5 values: {emb_field[:5]}")
                    elif isinstance(emb_field, dict):
                        self.logger.info(f"  embedding field: dict with {len(emb_field)} keys")
                        self.logger.info(f"  First 3 keys: {list(emb_field.keys())[:3]}")
                    else:
                        self.logger.info(f"  embedding field type: {type(emb_field)}")
                else:
                    self.logger.info(f"  ⚠️ embedding field is None or missing!")
                
                # Check binary embedding field
                emb_bin_field = vespa_doc['fields'].get('embedding_binary')
                if emb_bin_field is not None:
                    if isinstance(emb_bin_field, list):
                        self.logger.info(f"  embedding_binary field: list of {len(emb_bin_field)} int8")
                        self.logger.info(f"  First 5 values: {emb_bin_field[:5]}")
                    else:
                        self.logger.info(f"  embedding_binary field type: {type(emb_bin_field)}")
                else:
                    self.logger.info(f"  ⚠️ embedding_binary field is None or missing!")
            
            else:
                cap.release()
                return {"error": f"Unknown embedding type: {self.embedding_type}"}
            
            cap.release()
            
            # Feed documents to Vespa
            # For direct_video_segment, we already fed each segment immediately
            # For direct_video_frame and direct_video_global, we need to feed now
            if vespa_docs:
                if self.embedding_type in ["direct_video_frame", "direct_video_global", "direct_video_global_large"]:
                    self.logger.info(f"Feeding {len(vespa_docs)} documents to Vespa...")
                    if self._feed_to_vespa(vespa_docs, video_id):
                        self.logger.info(f"✅ Successfully fed all documents to Vespa")
                    else:
                        self.logger.warning(f"⚠️ Failed to feed some documents to Vespa")
                self.logger.info(f"Completed processing {len(vespa_docs)} segments for {video_id}")
                
                # Save embeddings metadata
                embeddings_data = {
                    "video_id": video_id,
                    "backend": "vespa",
                    "embedding_type": self.embedding_type,
                    "total_documents": len(vespa_docs),
                    "video_duration": duration,
                    "created_at": time.time()
                }
                
                # Add type-specific metadata
                if self.embedding_type == "direct_video_segment":
                    embeddings_data["segments_processed"] = num_segments
                    embeddings_data["segment_duration"] = segment_duration
                elif self.embedding_type == "direct_video_frame":
                    embeddings_data["frames_processed"] = len(vespa_docs)
                    embeddings_data["sampling_fps"] = sampling_fps
                    embeddings_data["frames_per_second"] = len(vespa_docs) / duration if duration > 0 else 0
                elif self.embedding_type in ["direct_video_global", "direct_video_global_large"]:
                    embeddings_data["global_embedding"] = True
                    embeddings_data["embedding_dim"] = 768 if "large" not in self.embedding_type else 1024
                    embeddings_data["sampling_fps"] = sampling_fps
                
                embeddings_file = output_dir / "embeddings" / f"{video_id}_{self.embedding_type}.json"
                embeddings_file.parent.mkdir(parents=True, exist_ok=True)
                with open(embeddings_file, 'w') as f:
                    json.dump(embeddings_data, f, indent=2)
                
                self.logger.info(f"✅ Direct video processing completed: {len(vespa_docs)} segments")
                return embeddings_data
            else:
                return {"error": "No segments processed"}
                
        except Exception as e:
            self.logger.error(f"Direct video embedding generation failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}