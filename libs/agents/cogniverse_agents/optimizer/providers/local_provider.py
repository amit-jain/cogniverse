"""
Local Provider Implementation

Implements the provider interfaces for local infrastructure:
- LiteLLM for unified model access (any provider: Ollama, vLLM, OpenAI, etc.)
- Local filesystem for artifact storage
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

import litellm

from .base_provider import ArtifactProvider, ModelProvider, ProviderFactory

logger = logging.getLogger(__name__)


class LocalModelProvider(ModelProvider):
    """Local implementation using LiteLLM for unified model access."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if "base_url" not in config:
            raise ValueError("LocalModelProvider requires 'base_url' in config")
        self.base_url = config["base_url"].rstrip("/")
        self.api_key = config.get("api_key", "no-key")

    def call_model(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 150,
    ) -> str:
        """Call a model via LiteLLM."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = litellm.completion(
                model=model_id,
                messages=messages,
                api_base=self.base_url,
                api_key=self.api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=120,
            )
            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"Local model call failed: {e}")

    def deploy_model_service(self, model_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        Check that the local LLM server is running and model is accessible.
        """
        logger.info("Checking local LLM service via LiteLLM...")

        try:
            # Use a minimal completion to verify the server is reachable
            litellm.completion(
                model=model_id or "test",
                messages=[{"role": "user", "content": "ping"}],
                api_base=self.base_url,
                api_key=self.api_key,
                max_tokens=1,
                timeout=10,
            )
            logger.info(f"LLM server reachable, model {model_id} responded")
            return {
                "inference_endpoint": self.base_url,
                "model": model_id,
                "status": "available",
            }

        except Exception as e:
            raise Exception(f"Failed to connect to local LLM server: {e}")

    def health_check(self) -> Dict[str, Any]:
        """Check LLM server health."""
        try:
            # LiteLLM doesn't have a generic health check,
            # so we check if the base_url is reachable
            import requests

            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "healthy",
                    "provider": "local",
                    "url": self.base_url,
                    "models_count": len(data.get("data", [])),
                }
            else:
                return {
                    "status": "unhealthy",
                    "provider": "local",
                    "error": f"HTTP {response.status_code}",
                }
        except Exception as e:
            return {"status": "error", "provider": "local", "error": str(e)}


class LocalArtifactProvider(ArtifactProvider):
    """Local filesystem implementation of ArtifactProvider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_path = Path(config.get("base_path", "./artifacts"))
        self.base_path.mkdir(parents=True, exist_ok=True)

    def upload_artifact(self, local_path: str, remote_path: str) -> bool:
        """Copy artifact to local artifacts directory."""
        try:
            source = Path(local_path)
            target = self.base_path / remote_path.lstrip("/")

            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            logger.info(f"Artifact copied: {source} -> {target}")
            return True

        except Exception as e:
            logger.error(f"Local copy error: {e}")
            return False

    def download_artifact(self, remote_path: str, local_path: str) -> bool:
        """Copy artifact from local artifacts directory."""
        try:
            source = self.base_path / remote_path.lstrip("/")
            target = Path(local_path)

            if not source.exists():
                logger.warning(f"Artifact not found: {source}")
                return False

            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            logger.info(f"Artifact copied: {source} -> {target}")
            return True

        except Exception as e:
            logger.error(f"Local copy error: {e}")
            return False

    def list_artifacts(self, path_prefix: str = "") -> List[str]:
        """List artifacts in local directory."""
        try:
            search_path = self.base_path
            if path_prefix:
                search_path = self.base_path / path_prefix.lstrip("/")

            if not search_path.exists():
                return []

            artifacts = []
            if search_path.is_file():
                artifacts.append(str(search_path.relative_to(self.base_path)))
            else:
                for item in search_path.rglob("*"):
                    if item.is_file():
                        artifacts.append(str(item.relative_to(self.base_path)))

            return sorted(artifacts)

        except Exception as e:
            logger.error(f"Local list error: {e}")
            return []


# Register the Local providers
ProviderFactory.register_model_provider("local", LocalModelProvider)
ProviderFactory.register_artifact_provider("local", LocalArtifactProvider)
