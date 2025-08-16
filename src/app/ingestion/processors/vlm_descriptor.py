#!/usr/bin/env python3
"""
VLM Description Generation Step

Generates visual descriptions for keyframes using Modal VLM service.
"""

import os
import json
import time
import base64
import zipfile
import tempfile
import requests
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional


class VLMDescriptor:
    """Handles VLM description generation for keyframes"""
    
    def __init__(self, vlm_endpoint: str, batch_size: int = 500, timeout: int = 10800, auto_start: bool = True):
        self.vlm_endpoint = vlm_endpoint
        self.batch_size = batch_size
        self.timeout = timeout  # 3 hours default
        self.auto_start = auto_start
        self._modal_process = None
        self._service_started = False
        
        # Setup logging
        self.logger = logging.getLogger("VLMDescriptor")
        self.logger.info(f"Initialized VLMDescriptor with endpoint: {vlm_endpoint}")
        self.logger.info(f"Batch size: {batch_size}, Timeout: {timeout}s, Auto-start: {auto_start}")
        
        # Don't auto-start service in __init__ - only start when actually needed
    
    def _ensure_service_running(self):
        """Ensure the Modal VLM service is running"""
        # Check if service is already running
        try:
            response = requests.get(self.vlm_endpoint, timeout=5)
            if response.status_code != 404:
                self.logger.info("VLM service is already running")
                print("  ✅ VLM service is already running")
                return
        except Exception as e:
            self.logger.debug(f"Service check failed: {e}")
            pass
        
        # Try to start Modal service
        self.logger.info("Starting Modal VLM service...")
        print("  🚀 Starting Modal VLM service...")
        try:
            # Check if modal_vlm_service.py exists
            service_file = Path("scripts/modal_vlm_service.py")
            if not service_file.exists():
                self.logger.warning("scripts/modal_vlm_service.py not found, cannot auto-start service")
                print("  ⚠️ scripts/modal_vlm_service.py not found, cannot auto-start service")
                return
            
            # Deploy the Modal service
            result = subprocess.run(
                ["modal", "deploy", "scripts/modal_vlm_service.py"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info("Modal VLM service started successfully")
                print("  ✅ Modal VLM service started successfully")
                # Wait a bit for service to be ready
                time.sleep(5)
            else:
                self.logger.error(f"Failed to start Modal service: {result.stderr}")
                print(f"  ❌ Failed to start Modal service: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error starting Modal service: {e}")
            print(f"  ❌ Error starting Modal service: {e}")
    
    def stop_service(self):
        """Stop the Modal VLM service if it was started"""
        if self._service_started:
            self.logger.info("Modal VLM service will auto-stop after inactivity")
            print("  🛑 Modal VLM service will auto-stop after inactivity")
            # Modal serverless functions automatically stop after ~5-10 minutes of inactivity
            # The 'modal stop' command is primarily for long-running services, not serverless functions
            # Optionally try to stop it explicitly
            try:
                result = subprocess.run(
                    ["modal", "stop", "cogniverse-vlm"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if "stopped" in result.stdout.lower() or result.returncode == 0:
                    self.logger.info("Modal VLM service stop command sent")
                    print("  ✅ Modal VLM service stop command sent")
            except Exception as e:
                self.logger.debug(f"Modal stop command failed: {e}")
                # It's okay if this fails - serverless functions auto-stop anyway
                pass
            self._service_started = False
        else:
            self.logger.info("Modal VLM service was not started by this pipeline")
            print("  ℹ️ Modal VLM service was not started by this pipeline")
    
    
    def generate_descriptions(self, keyframes_metadata: Dict[str, Any], output_dir: Path = None) -> Dict[str, Any]:
        """Generate VLM descriptions for keyframes using Modal service"""
        # Check if keyframes_metadata is empty or doesn't have required data
        if not keyframes_metadata or "video_id" not in keyframes_metadata:
            self.logger.info("No keyframes to generate descriptions for")
            return {"descriptions": {}}
        
        video_id = keyframes_metadata["video_id"]
        self.logger.info(f"Starting VLM description generation for video: {video_id}")
        print(f"🤖 Generating VLM descriptions for: {video_id}")
        
        # Use OutputManager for consistent directory structure
        if output_dir is None:
            from src.utils.output_manager import get_output_manager
            output_manager = get_output_manager()
            descriptions_file = output_manager.get_processing_dir("descriptions") / f"{video_id}.json"
        else:
            # Legacy path support
            descriptions_file = output_dir / "descriptions" / f"{video_id}.json"
        
        # Remove caching - always regenerate descriptions
        
        # Only start service when we actually need to generate descriptions
        if self.auto_start and not self._service_started:
            self._ensure_service_running()
            self._service_started = True
        
        keyframes = keyframes_metadata["keyframes"]
        if not keyframes:
            self.logger.warning(f"No keyframes found for video: {video_id}")
            print("  ⚠️ No keyframes found")
            return {}
        
        self.logger.info(f"Processing {len(keyframes)} keyframes for video: {video_id}")
        print(f"  🔄 Processing {len(keyframes)} keyframes...")
        
        # Process in batches
        descriptions = {}
        total_batches = (len(keyframes) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(keyframes), self.batch_size):
            batch_num = i // self.batch_size + 1
            batch = keyframes[i:i + self.batch_size]
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} frames)")
            batch_descriptions = self._process_vlm_batch(batch)
            descriptions.update(batch_descriptions)
            
            print(f"  📊 Processed batch {batch_num}/{total_batches}")
            
            # Save progress periodically
            descriptions_file.parent.mkdir(parents=True, exist_ok=True)
            with open(descriptions_file, 'w') as f:
                json.dump(descriptions, f, indent=2)
        
        self.logger.info(f"Successfully generated {len(descriptions)} descriptions for video: {video_id}")
        print(f"  ✅ Generated {len(descriptions)} descriptions")
        
        # Return in the expected format for the pipeline
        return {
            "video_id": video_id,
            "descriptions": descriptions,
            "total_descriptions": len(descriptions),
            "created_at": time.time()
        }
    
    def _process_vlm_batch(self, keyframes: List[Dict]) -> Dict[str, str]:
        """Process a batch of keyframes through Modal VLM service"""
        
        # Create frame mapping for batch processing
        frame_mapping = {}
        frame_paths = []
        
        for keyframe in keyframes:
            frame_path = Path(keyframe["path"])
            if frame_path.exists():
                frame_mapping[frame_path.name] = str(keyframe["frame_id"])
                frame_paths.append(frame_path)
        
        if not frame_paths:
            self.logger.warning("No valid frame paths found in batch")
            return {}
        
        self.logger.debug(f"Processing batch of {len(frame_paths)} frames")
        
        # Create temporary zip file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
            with zipfile.ZipFile(temp_zip.name, 'w') as zf:
                for frame_path in frame_paths:
                    zf.write(frame_path, frame_path.name)
            
            # Upload and process batch
            with open(temp_zip.name, 'rb') as f:
                zip_data = f.read()
                
                payload = {
                    "zip_data": base64.b64encode(zip_data).decode('utf-8'),
                    "frame_mapping": frame_mapping
                }
                
                batch_endpoint = self.vlm_endpoint.replace('generate-description', 'upload-and-process-frames')
                
                try:
                    self.logger.info(f"Uploading batch of {len(frame_mapping)} frames...")
                    print(f"    📦 Uploading batch of {len(frame_mapping)} frames...")
                    response = requests.post(
                        batch_endpoint,
                        json=payload,
                        headers={'Content-Type': 'application/json'},
                        timeout=self.timeout
                    )
                    
                    self.logger.info(f"Batch request completed with status: {response.status_code}")
                    print(f"    📨 Batch request completed with status: {response.status_code}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        descriptions_returned = result.get("descriptions", {})
                        self.logger.info(f"Batch processing successful: {len(descriptions_returned)} descriptions returned")
                        print(f"    ✅ Batch processing successful: {len(descriptions_returned)} descriptions returned")
                        return descriptions_returned
                    else:
                        self.logger.error(f"Batch processing failed: {response.status_code} - {response.text}")
                        print(f"    ❌ Batch processing failed: {response.status_code} - {response.text}")
                        return {}
                        
                except Exception as e:
                    self.logger.error(f"Batch processing error: {e}")
                    print(f"    ❌ Batch processing error: {e}")
                    # If service is not running, try to start it
                    if "Connection" in str(e) and self.auto_start:
                        self._ensure_service_running()
                        # Retry once after starting service
                        try:
                            response = requests.post(
                                batch_endpoint,
                                json=payload,
                                headers={'Content-Type': 'application/json'},
                                timeout=self.timeout
                            )
                            if response.status_code == 200:
                                result = response.json()
                                return result.get("descriptions", {})
                        except Exception as retry_e:
                            self.logger.error(f"Retry also failed: {retry_e}")
                            pass
                    return {}
                finally:
                    # Cleanup temp file
                    os.unlink(temp_zip.name)
    
    def process_single_frame(self, frame_path: Path) -> str:
        """Process a single frame (fallback method)"""
        try:
            self.logger.debug(f"Processing single frame: {frame_path}")
            with open(frame_path, 'rb') as f:
                frame_data = f.read()
            frame_base64 = base64.b64encode(frame_data).decode('utf-8')
            
            payload = {
                "frame_base64": frame_base64,
                "prompt": "Provide a detailed description of this video frame, including objects, people, actions, scene setting, and visual details."
            }
            
            response = requests.post(
                self.vlm_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300  # 5 minute timeout for single frames
            )
            
            if response.status_code == 200:
                result = response.json()
                description = result.get("description", "")
                self.logger.debug(f"Single frame processed successfully: {len(description)} chars")
                return description
            else:
                error_msg = f"Error: VLM API error {response.status_code}"
                self.logger.error(error_msg)
                return error_msg
                
        except Exception as e:
            error_msg = f"Error: {e}"
            self.logger.error(f"Single frame processing failed: {e}")
            return error_msg