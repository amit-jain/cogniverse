"""Modal deployment for the LateOn ColBERT embedding service."""

from cogniverse_cli.modal_inference.pylate import build_pylate_app
from cogniverse_foundation.inference_specs import get_inference_service_spec

app = build_pylate_app(get_inference_service_spec("colbert_pylate"))
