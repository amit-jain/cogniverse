# Deploy Modal VLM Service for Video Processing

**Note**: VLM functionality is part of `cogniverse-runtime` (Application Layer). This guide covers deploying the Modal VLM service.

This guide walks you through deploying the Modal VLM service for image description generation in your video processing pipeline.

**Note**: There's also an automated setup script available at `docs/modal/setup_modal_vlm.py` that handles all these steps automatically.

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com) if you don't have an account
2. **Modal CLI**: Install the Modal Python package

```bash
pip install modal
```

3. **Authentication**: Set up Modal authentication

```bash
modal setup
```

## Step 1: Deploy the Modal VLM Service

Deploy the VLM service to Modal:

```bash
modal deploy scripts/modal_vlm_service.py
```

This will:

- Download and cache the Qwen2-VL-7B-Instruct model
- Create a serverless GPU-accelerated inference service
- Return a web endpoint URL for API access

## Step 2: Get Your Endpoint URL

After deployment, Modal will provide an endpoint URL that looks like:
```text
https://username--cogniverse-vlm-vlmmodel-generate-description.modal.run
```

**Copy this URL** - you'll need it for the next step.

## Step 3: Update Configuration

Edit your `configs/config.json` file and replace the placeholder:

```json
{
  "vlm_endpoint_url": "https://your-actual-vlm-endpoint-url-here"
}
```

## Step 4: Test the Integration

Test that everything works:

```bash
# Test just the Modal service (uses --frame-path parameter)
modal run scripts/modal_vlm_service.py::test_vlm --frame-path /path/to/test/frame.jpg
```

## Step 5: Run Video Processing

Now your video processing pipeline will use Modal instead of Ollama:

```bash
# Run ingestion with Modal VLM backend
JAX_PLATFORM_NAME=cpu uv run python scripts/run_ingestion.py \
    --video_dir data/testset/evaluation/sample_videos \
    --backend vespa
```

## Expected Benefits

- **Better Reliability**: No local Ollama service dependencies
- **GPU Acceleration**: Fast inference on Modal's H100/A100 GPUs (default: H100)
- **Scalability**: Handles multiple videos concurrently
- **Quality**: Qwen2-VL-7B produces high-quality descriptions

## Troubleshooting

### "VLM endpoint not configured" error
- Ensure you've deployed the service: `modal deploy scripts/modal_vlm_service.py`
- Check that `configs/config.json` has the correct endpoint URL
- Verify the endpoint is working: `curl -X POST your-endpoint-url -d '{"frame_base64":"..."}'`

### GPU/Memory errors
- Try reducing GPU type in `modal_vlm_service.py`: Change default from `"h100"` to `"a100"`, `"l40s"`, or `"t4"`
- Set environment variable: `GPU_TYPE=a100 modal deploy scripts/modal_vlm_service.py`
- Reduce batch size or concurrent requests

### Authentication errors
- Run `modal setup` again to refresh credentials
- Check your Modal account has sufficient credits

## Cost Estimation

- **Cold start**: ~10-15 seconds (first request after idle)
- **Warm inference**: ~2-4 seconds per image
- **Cost**: ~$0.01-0.05 per image (depending on GPU type)

For our elephant_dream_clip test (719 frames):

- **Total cost**: ~$7-35

- **Processing time**: ~25-50 minutes (including cold starts)

## Scaling Configuration

For high-volume processing, modify these settings in `scripts/modal_vlm_service.py`:

**Increase concurrent requests** (line 91):
```python
@modal.concurrent(max_inputs=100)  # Default is 50
```

**Change GPU type** (line 18 or via environment variable):
```bash
# Set via environment variable before deploying
GPU_TYPE=a100 modal deploy scripts/modal_vlm_service.py
```

**Keep service warm longer** (line 87):
```python
@app.cls(
    gpu=GPU_CONFIG,
    timeout=180 * MINUTES,
    scaledown_window=600,  # Default is 300 (5 minutes), increase to 10 minutes
    image=vlm_image,
    volumes=volumes,
)
```
