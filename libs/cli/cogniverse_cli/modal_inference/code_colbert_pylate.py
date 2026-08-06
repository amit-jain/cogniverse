"""Modal deployment for the LateOn code embedding service."""

from cogniverse_cli.modal_inference.pylate import build_pylate_app
from cogniverse_cli.modal_inference_config import get_inference_service_spec

app = build_pylate_app(get_inference_service_spec("code_colbert_pylate"))
