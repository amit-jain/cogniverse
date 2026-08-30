"""Modal App for the production Whisper transcription model."""

from cogniverse_cli.modal_inference.vllm import build_vllm_app
from cogniverse_foundation.inference_specs import get_inference_service_spec

app = build_vllm_app(get_inference_service_spec("vllm_asr"))
