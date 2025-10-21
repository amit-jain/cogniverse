"""
Modal VLM Service for Video Processing Pipeline
Based on Modal's SGLang VLM example, modified to accept direct image data.
"""

import base64
import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Optional
from uuid import uuid4

import modal

# GPU Configuration - Using H100 for best performance with large model
GPU_TYPE = os.environ.get("GPU_TYPE", "h100")
GPU_COUNT = int(os.environ.get("GPU_COUNT", 1))
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

SGL_LOG_LEVEL = "error"  # try "debug" or "info" if you have issues
MINUTES = 60  # seconds

# Model Configuration - Using larger, more capable Qwen2-VL model
MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"
MODEL_REVISION = "a7a06a1cc11b4514ce9edcde0e3ca1d16e5ff2fc"
TOKENIZER_PATH = "Qwen/Qwen2-VL-7B-Instruct"
MODEL_CHAT_TEMPLATE = "qwen2-vl"

# Volume setup for model caching and frame storage
MODEL_VOL_PATH = Path("/models")
MODEL_VOL = modal.Volume.from_name("sgl-cache", create_if_missing=True)
FRAMES_VOL_PATH = Path("/shared_frames")
FRAMES_VOL = modal.Volume.from_name("cogniverse-frames", create_if_missing=True)
volumes = {MODEL_VOL_PATH: MODEL_VOL, FRAMES_VOL_PATH: FRAMES_VOL}

def download_model():
    """Download the VLM model from Hugging Face."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_PATH,
        local_dir=str(MODEL_VOL_PATH / MODEL_PATH),
        revision=MODEL_REVISION,
        ignore_patterns=["*.pt", "*.bin"],
    )

# Container image definition
cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

vlm_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .pip_install(  # add sglang and Python dependencies
        "transformers==4.47.1",
        "numpy<2",
        "fastapi[standard]==0.115.4",
        "pydantic==2.9.2",
        "requests==2.32.3",
        "starlette==0.41.2",
        "torch==2.4.0",
        "sglang[all]==0.4.1",
        "sgl-kernel==0.1.0",
        "hf-xet==1.1.5",
        "Pillow",  # For image processing
        extra_options="--find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/",
    )
    .env(
        {
            "HF_HOME": str(MODEL_VOL_PATH),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .run_function(  # download the model
        download_model, volumes=volumes
    )
)

app = modal.App("cogniverse-vlm")

@app.cls(
    gpu=GPU_CONFIG,
    timeout=180 * MINUTES,  # 3 hour timeout for batch processing
    scaledown_window=300,  # Keep warm for 5 minutes
    image=vlm_image,
    volumes=volumes,
)
@modal.concurrent(max_inputs=50)  # Reduced for stability
class VLMModel:
    @modal.enter()
    def start_runtime(self):
        """Starts an SGL runtime to execute inference."""
        import sglang as sgl

        self.runtime = sgl.Runtime(
            model_path=MODEL_PATH,
            tokenizer_path=TOKENIZER_PATH,
            tp_size=GPU_COUNT,  # tensor parallel size
            log_level=SGL_LOG_LEVEL,
        )
        self.runtime.endpoint.chat_template = sgl.lang.chat_template.get_chat_template(
            MODEL_CHAT_TEMPLATE
        )
        sgl.set_default_backend(self.runtime)
        print("✅ VLM Model runtime started successfully")

    @modal.fastapi_endpoint(method="POST", docs=True)
    def generate_description(self, request: dict) -> dict:
        """
        Generate description for a video frame sent as base64 data.
        
        Args:
            request: {
                "frame_base64": "base64 encoded video frame data",
                "prompt": "optional custom prompt (default: describe this frame)"
            }
            
        Returns:
            {"description": "generated text description"}
        """
        import sglang as sgl
        
        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"🎯 Generating description for request {request_id}")

        # Get frame data from request - support path, remote path, and base64
        frame_path = request.get("frame_path")
        remote_frame_path = request.get("remote_frame_path")
        frame_base64 = request.get("frame_base64")
        
        if not frame_path and not remote_frame_path and not frame_base64:
            return {"error": "No frame_path, remote_frame_path, or frame_base64 provided in request"}

        # Get prompt (detailed for better analysis)
        prompt = request.get("prompt", "Provide a detailed description of this video frame, including objects, people, actions, scene setting, and visual details.")

        try:
            import os
            
            if remote_frame_path:
                # Use remote file path (from uploaded zip)
                temp_filename = remote_frame_path
                print(f"📁 Using remote file path: {temp_filename}")
                
                # Verify file exists
                if not os.path.exists(temp_filename):
                    return {"error": f"Remote frame file not found: {temp_filename}"}
            elif frame_path:
                # Use provided file path directly
                temp_filename = frame_path
                print(f"📁 Using file path: {temp_filename}")
                
                # Verify file exists
                if not os.path.exists(temp_filename):
                    return {"error": f"Frame file not found: {temp_filename}"}
            else:
                # Decode base64 frame and save to persistent file
                frame_data = base64.b64decode(frame_base64)
                
                # Create a persistent file path (don't use random names that can disappear)
                temp_filename = f"/tmp/vlm_frame_{request_id}.jpg"
                
                # Write file and ensure it's flushed to disk
                with open(temp_filename, 'wb') as f:
                    f.write(frame_data)
                    f.flush()
                    os.fsync(f.fileno())
                
                print(f"📁 Created temp file: {temp_filename} ({len(frame_data)} bytes)")
            
            # Verify file exists
            if not os.path.exists(temp_filename):
                return {"error": f"Failed to create temporary file: {temp_filename}"}

            # Define the SGLang function (matching Modal example pattern)
            @sgl.function
            def image_qa(s, image_path, question):
                s += sgl.user(sgl.image(image_path) + question)
                s += sgl.assistant(sgl.gen("answer"))

            # Execute the function
            state = image_qa.run(
                image_path=temp_filename,
                question=prompt
            )
            
            print("🤖 SGLang execution completed")
            
            duration = round((time.monotonic_ns() - start) / 1e9, 2)
            print(f"✅ Request {request_id} completed in {duration} seconds")
            
            # Extract the answer from state (following Modal example pattern)
            try:
                description = state["answer"]
                print(f"📝 Generated description: {description[:100]}..." if len(description) > 100 else f"📝 Generated description: {description}")
                
                # Debug: Check if description is empty
                if not description or description.strip() == "":
                    print(f"⚠️ Empty description generated! State: {state}")
                    print(f"   State type: {type(state)}")
                    print(f"   State keys: {list(state.keys()) if hasattr(state, 'keys') else 'No keys'}")
                    
            except KeyError:
                print(f"⚠️ No 'answer' key in state. Available keys: {list(state.keys()) if hasattr(state, 'keys') else 'Not a dict'}")
                print(f"   Full state: {state}")
                description = str(state) if state else "No description generated"
            
            # Clean up temp file only if we created it (not if it was provided path)
            if not frame_path and not remote_frame_path:  # Only cleanup if we created the temp file
                try:
                    os.unlink(temp_filename)
                    print(f"🗑️ Cleaned up temp file: {temp_filename}")
                except Exception as cleanup_error:
                    print(f"⚠️ Cleanup warning: {cleanup_error}")
            
            return {
                "description": description,
                "request_id": str(request_id),
                "duration_seconds": duration
            }
            
        except Exception as e:
            print(f"❌ Error processing request {request_id}: {e}")
            import traceback
            print(f"📋 Full traceback: {traceback.format_exc()}")
            
            # Clean up temp file if it exists and we created it
            if 'temp_filename' in locals() and not frame_path and not remote_frame_path:
                try:
                    import os
                    if os.path.exists(temp_filename):
                        os.unlink(temp_filename)
                        print(f"🗑️ Cleaned up temp file after error: {temp_filename}")
                except Exception as cleanup_error:
                    print(f"⚠️ Cleanup error: {cleanup_error}")
            
            return {"error": str(e), "request_id": str(request_id)}
    
    @modal.fastapi_endpoint(method="POST", docs=True)
    def upload_and_process_frames(self, request: dict) -> dict:
        """
        Upload zip file, extract frames, and process them all in the same container.
        Request: {
            "zip_data": "base64 encoded zip file",
            "frame_mapping": {frame_filename: frame_key}
        }
        """
        import base64
        import os
        from uuid import uuid4

        import sglang as sgl
        
        upload_id = str(uuid4())
        start = time.monotonic_ns()
        print(f"📦 Processing zip upload and descriptions {upload_id}")
        
        try:
            # Decode zip data
            zip_data = base64.b64decode(request["zip_data"])
            frame_mapping = request["frame_mapping"]
            
            # Create extraction directory
            extract_dir = f"/tmp/frames_{upload_id}"
            os.makedirs(extract_dir, exist_ok=True)
            
            # Save zip data to temporary file
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
                temp_zip.write(zip_data)
                temp_zip_path = temp_zip.name
            
            # Extract zip contents
            with zipfile.ZipFile(temp_zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            
            # Cleanup zip file
            os.unlink(temp_zip_path)
            
            # Count extracted files
            extracted_files = list(Path(extract_dir).glob("*.jpg")) + list(Path(extract_dir).glob("*.png"))
            print(f"✅ Extracted {len(extracted_files)} frames, starting processing...")
            
            # Process all frames in this container
            results = {}
            prompt = "Provide a detailed description of this video frame, including objects, people, actions, scene setting, and visual details."
            
            @sgl.function
            def image_qa(s, image_path, question):
                s += sgl.user(sgl.image(image_path) + question)
                s += sgl.assistant(sgl.gen("answer"))
            
            for frame_file in extracted_files:
                frame_filename = frame_file.name
                frame_key = frame_mapping.get(frame_filename)
                
                if not frame_key:
                    print(f"⚠️ No mapping found for {frame_filename}")
                    continue
                
                try:
                    print(f"  🔄 Processing {frame_filename} -> {frame_key}")
                    
                    # Execute the function
                    state = image_qa.run(
                        image_path=str(frame_file),
                        question=prompt
                    )
                    
                    # Extract description
                    description = state["answer"]
                    if description and description.strip():
                        results[frame_key] = description
                        print(f"  ✅ {frame_key}: {description[:50]}...")
                    else:
                        results[frame_key] = "Error: Empty description generated"
                        print(f"  ❌ {frame_key}: Empty description")
                        
                except Exception as e:
                    error_msg = f"Error processing {frame_filename}: {e}"
                    results[frame_key] = f"Error: {error_msg}"
                    print(f"  ❌ {frame_key}: {error_msg}")
            
            # Cleanup extraction directory
            import shutil
            shutil.rmtree(extract_dir)
            
            duration = round((time.monotonic_ns() - start) / 1e9, 2)
            print(f"✅ Batch processing completed in {duration}s: {len(results)} descriptions")
            
            return {
                "descriptions": results,
                "upload_id": upload_id,
                "processed_frames": len(results),
                "duration_seconds": duration
            }
            
        except Exception as e:
            print(f"❌ Batch processing {upload_id} failed: {e}")
            import traceback
            print(f"📋 Full traceback: {traceback.format_exc()}")
            return {"error": str(e), "upload_id": upload_id}

    @modal.exit()
    def shutdown_runtime(self):
        """Clean shutdown of the runtime."""
        print("🔄 Shutting down VLM runtime...")
        self.runtime.shutdown()

# Separate upload function for zip files using FastAPI
@app.function(timeout=30 * 60, volumes=volumes)  # 30 minute timeout for uploads
@modal.asgi_app()
def upload_app():

    from fastapi import FastAPI, Request
    
    upload_web_app = FastAPI()
    
    @upload_web_app.post("/")
    async def upload_frames(request: Request):
        """Upload and extract zip file containing frames."""
        import os
        import time
        from pathlib import Path
        from uuid import uuid4
        
        request_data = await request.body()
        upload_id = str(uuid4())
        start = time.time()
        print(f"📦 Processing zip upload {upload_id}")
        
        try:
            # Create extraction directory in shared volume
            extract_dir = f"/shared_frames/frames_{upload_id}"
            os.makedirs(extract_dir, exist_ok=True)
            
            # Save zip data to temporary file
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
                temp_zip.write(request_data)
                temp_zip_path = temp_zip.name
            
            # Extract zip contents
            with zipfile.ZipFile(temp_zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            
            # Cleanup zip file
            os.unlink(temp_zip_path)
            
            # Count extracted files
            extracted_files = list(Path(extract_dir).glob("*.jpg")) + list(Path(extract_dir).glob("*.png"))
            
            # Commit changes to volume
            FRAMES_VOL.commit()
            
            duration = time.time() - start
            print(f"✅ Upload {upload_id} completed in {duration:.2f}s: {len(extracted_files)} frames extracted")
            
            return {
                "remote_path": extract_dir,
                "upload_id": upload_id,
                "extracted_files": len(extracted_files),
                "duration_seconds": duration
            }
            
        except Exception as e:
            print(f"❌ Upload {upload_id} failed: {e}")
            return {"error": str(e), "upload_id": upload_id}
    
    return upload_web_app

# Local testing entrypoint
@app.local_entrypoint()
def test_vlm(frame_path: Optional[str] = None):
    """Test the VLM service with a local video frame."""
    import json
    import urllib.request
    
    if not frame_path:
        print("Please provide --frame-path argument for testing")
        return
    
    # Load and encode test frame
    with open(frame_path, "rb") as f:
        frame_data = f.read()
    
    frame_base64 = base64.b64encode(frame_data).decode('utf-8')
    
    model = VLMModel()
    
    payload = json.dumps({
        "frame_base64": frame_base64,
        "prompt": "What do you see in this video frame?"
    })
    
    req = urllib.request.Request(
        model.generate_description.get_web_url(),
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    print("🚀 Testing VLM service...")
    with urllib.request.urlopen(req) as response:
        assert response.getcode() == 200, f"HTTP {response.getcode()}"
        result = json.loads(response.read().decode())
        print(f"📝 Description: {result.get('description')}")
        print(f"⏱️  Duration: {result.get('duration_seconds')}s") 
