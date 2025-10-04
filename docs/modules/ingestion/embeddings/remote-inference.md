# Remote Model Inference Support

The model loader now supports remote inference providers, allowing you to offload model inference to dedicated services like Infinity or Modal.

## Configuration

To use remote inference, add these fields to your config:

```json
{
  "remote_inference_url": "https://your-endpoint.com",
  "remote_inference_api_key": "your-api-key",  // Optional
  "remote_inference_provider": "infinity"      // Optional: "infinity", "modal", "custom"
}
```

**TODO**: These fields need to be added to the main config schema.

## How It Works

1. **Automatic Detection**: When `remote_inference_url` is present in config, the ModelLoaderFactory automatically creates a remote loader instead of a local one.

2. **Supported Models**:
   - **ColPali/ColQwen**: Uses `RemoteColPaliLoader` 
   - **VideoPrism**: Uses `RemoteVideoPrismLoader`

3. **API Endpoints**:
   - Images: `POST {endpoint}/v1/embeddings`
   - Videos: `POST {endpoint}/v1/video/embeddings`

4. **Fallback**: If remote inference fails, the client logs a warning and returns mock embeddings for development.

## Usage Examples

### Remote ColPali with Infinity
```python
config = {
    "remote_inference_url": "http://localhost:8080/infinity",
    "remote_inference_api_key": "your-infinity-key"
}

# This will automatically use RemoteColPaliLoader
model, processor = get_or_load_model("vidore/colpali-v1.2", config, logger)
```

### Remote VideoPrism with Modal
```python
config = {
    "remote_inference_url": "https://your-app.modal.run/videoprism",
    "remote_inference_api_key": "modal-secret"
}

# This will automatically use RemoteVideoPrismLoader
model, processor = get_or_load_model("videoprism_public_v1_base_hf", config, logger)
```

### Local Fallback
```python
# Without remote_inference_url, uses local models
config = {"device": "cuda"}
model, processor = get_or_load_model("vidore/colpali-v1.2", config, logger)
```

## Implementation Details

1. **Caching**: Models are cached using `model_name@endpoint_url` as the key, allowing different endpoints for the same model.

2. **Image Processing**:
   - Converts images to base64 PNG format
   - Supports both file paths and PIL Images
   - Sends batch requests for efficiency

3. **Video Processing**:
   - Extracts video segments using ffmpeg
   - Encodes segments as base64 MP4
   - Handles large video files with 10-minute timeout

4. **Error Handling**:
   - Graceful fallback to mock embeddings on failure
   - Detailed logging of errors
   - HTTP session reuse for performance

## Testing

Run the test script to verify functionality:
```bash
python test_remote_inference.py
```

The test verifies:
- Remote model loading
- Endpoint URL configuration
- Caching behavior
- Local fallback