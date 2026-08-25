"""Modal App for the production DSPy teacher model."""

from cogniverse_cli.modal_inference.vllm import build_vllm_app
from cogniverse_cli.modal_inference_config import get_inference_service_spec

app = build_vllm_app(get_inference_service_spec("vllm_llm_teacher"))
