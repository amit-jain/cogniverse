# Deploy Modal VLM Service for Video Processing

**Note**: VLM functionality is part of `cogniverse-runtime` (Application Layer). This guide covers deploying the Modal VLM service.

This guide walks you through deploying the Modal VLM service for image description generation in your video processing pipeline.

**Note**: `docs/modal/setup_modal_vlm.py` attempts to automate steps 1, 3, and 4 below, but it looks for `config.json` and `modal_vlm_service.py` in the current working directory — this repo keeps those files at `configs/config.json` and `scripts/modal_vlm_service.py`, so the script's deploy/config-update calls will not find them unless you copy it next to those paths first. The manual steps below always work and are the recommended path.

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com) if you don't have an account
2. **Modal CLI**: Already a pinned workspace dependency (`modal==1.4.1` in the root `pyproject.toml`) — `uv sync` installs it into `.venv`. Verify with:

```bash
uv run modal --version
```

If you're working outside this repo's `uv`-managed environment, install it standalone instead:

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

- Download and cache the Qwen3-VL-8B-Instruct model
- Create a serverless GPU-accelerated inference service
- Return web endpoint URLs for API access

## Step 2: Get Your Endpoint URL

Deploying `scripts/modal_vlm_service.py` publishes three endpoints:

- **`VLMModel.generate_description`** — the one you need. It looks like:
  ```text
  https://username--cogniverse-vlm-vlmmodel-generate-description.modal.run
  ```
- **`VLMModel.upload_and_process_frames`** — a batch endpoint (zip of frames in, all descriptions out) used automatically for multi-frame batches.
- **`upload_app`** — a standalone ASGI app that extracts an uploaded frame zip into the shared `cogniverse-frames` volume.

**Copy only the `generate_description` URL** — you'll need it for the next step. Do not substitute one of the other two: the ingestion pipeline derives the batch endpoint itself by string-replacing `generate-description` with `upload-and-process-frames` in whatever URL you configure (see `libs/runtime/cogniverse_runtime/ingestion/processors/vlm_descriptor.py`), so configuring anything other than the `generate_description` URL breaks batch processing.

## Step 3: Update Configuration

The pipeline reads the VLM endpoint from the **backend profile** that runs `VLMDescriptionStrategy`, not from any top-level config key. In this repo that's the `video_colpali_smol500_mv_frame` profile, at:

```text
backend.profiles.video_colpali_smol500_mv_frame.strategies.description.params.vlm_endpoint
```

`configs/config.json` also has a top-level `vlm_endpoint_url` key — that field is only a convenience value written by the setup helper scripts (`scripts/setup_video_processing.py`, `docs/modal/setup_modal_vlm.py`) for reference; the ingestion pipeline never reads it. Editing it alone does **not** change what the pipeline calls.

Update the profile's `vlm_endpoint` using any of these:

**Option A — edit `configs/config.json` directly:**

```json
{
  "backend": {
    "profiles": {
      "video_colpali_smol500_mv_frame": {
        "strategies": {
          "description": {
            "class": "VLMDescriptionStrategy",
            "params": {
              "vlm_endpoint": "https://your-actual-vlm-endpoint-url-here/",
              "batch_size": 500,
              "timeout": 10800,
              "auto_start": true
            }
          }
        }
      }
    }
  }
}
```

**Option B — patch it through the admin API** (backend profiles are backed by the config service `"backend"`; `ConfigManager` methods default to `service="backend"`, and profile updates are deep-merged so a partial `strategies` body only touches the keys you send):

```bash
curl -X PUT http://localhost:8000/admin/profiles/video_colpali_smol500_mv_frame \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "flywheel_org:production",
    "strategies": {
      "description": {
        "params": {
          "vlm_endpoint": "https://your-actual-vlm-endpoint-url-here/"
        }
      }
    }
  }'
```

**Option C — use the dashboard**: open the Streamlit dashboard's Backend Profile tab, select `video_colpali_smol500_mv_frame`, edit the description strategy's `vlm_endpoint`, and save.

## Step 4: Test the Integration

Test that everything works:

```bash
# Test just the Modal service (uses --frame-path parameter)
modal run scripts/modal_vlm_service.py::test_vlm --frame-path /path/to/test/frame.jpg
```

Or hit the deployed endpoint directly:

```bash
curl -X POST your-generate-description-endpoint-url \
  -H "Content-Type: application/json" \
  -d '{"frame_base64":"..."}'
```

## Step 5: Run Video Processing

Now your video processing pipeline will use Modal instead of Ollama:

```bash
# Run ingestion with Modal VLM backend
JAX_PLATFORM_NAME=cpu uv run python scripts/run_ingestion.py \
    --video_dir data/testset/evaluation/sample_videos \
    --backend vespa
```

Description generation only runs when the active profile's `pipeline_config.generate_descriptions` is `true` — it already is for `video_colpali_smol500_mv_frame`.

## Expected Benefits

- **Better Reliability**: No local Ollama service dependencies
- **GPU Acceleration**: Fast inference on Modal's H100/A100 GPUs (default: H100)
- **Scalability**: Handles multiple videos concurrently
- **Quality**: Qwen3-VL-8B produces high-quality descriptions

## Troubleshooting

### "VLMProcessor requires 'vlm_endpoint' in config" error
- Ensure you've deployed the service: `modal deploy scripts/modal_vlm_service.py`
- Set `vlm_endpoint` in the `video_colpali_smol500_mv_frame` profile's `strategies.description.params` (see Step 3) — the top-level `vlm_endpoint_url` key is not read by the pipeline
- Verify the endpoint is working: `curl -X POST your-endpoint-url -H "Content-Type: application/json" -d '{"frame_base64":"..."}'`

### Batch requests return no descriptions
- This happens when the configured `vlm_endpoint` isn't the `generate_description` URL (see Step 2) — the pipeline derives the batch (`upload-and-process-frames`) URL from it via string substitution, so any other URL shape breaks batch mode silently
- Re-check the value against the URL Modal printed for `VLMModel.generate_description`

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

For our elephant_dream_clip test (2,009 keyframes):

- **Total cost**: ~$20-100

- **Processing time**: ~65-135 minutes (including cold starts)

## Scaling Configuration

For high-volume processing, modify these settings in `scripts/modal_vlm_service.py`:

**Increase concurrent requests** (`@modal.concurrent` decorator on `VLMModel`):
```python
@modal.concurrent(max_inputs=100)  # Default is 50
```

**Change GPU type** (line 18 or via environment variable):
```bash
# Set via environment variable before deploying
GPU_TYPE=a100 modal deploy scripts/modal_vlm_service.py
```

**Keep service warm longer** (`@app.cls` decorator on `VLMModel`):
```python
@app.cls(
    gpu=GPU_CONFIG,
    timeout=180 * MINUTES,
    scaledown_window=600,  # Default is 300 (5 minutes), increase to 10 minutes
    image=vlm_image,
    volumes=volumes,
)
```
