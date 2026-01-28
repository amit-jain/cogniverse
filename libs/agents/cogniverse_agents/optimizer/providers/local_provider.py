"""
Local Provider Implementation

Implements the provider interfaces for local infrastructure:
- Ollama for model hosting
- Local filesystem for artifact storage
"""

import shutil
from pathlib import Path
from typing import Any, Dict, List

import requests

from .base_provider import ArtifactProvider, ModelProvider, ProviderFactory


class LocalModelProvider(ModelProvider):
    """Local implementation using Ollama."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ollama_base_url = config.get("ollama_base_url", "http://localhost:11434")

    def call_model(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 150,
    ) -> str:
        """Call a model via Ollama API."""

        # Remove any provider prefix from model_id
        if model_id.startswith("ollama/"):
            model_id = model_id[7:]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request_data = {
            "model": model_id,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/chat", json=request_data, timeout=120
            )

            if response.status_code != 200:
                raise Exception(f"Ollama call failed: {response.text}")

            result = response.json()
            return result["message"]["content"]

        except Exception as e:
            raise Exception(f"Local model call failed: {e}")

    def deploy_model_service(self, model_id: str = None, **kwargs) -> Dict[str, str]:
        """
        For local provider, this ensures Ollama is running and model is available.
        """
        print("🔧 Checking local Ollama service...")

        try:
            # Check if Ollama is running
            health_response = requests.get(
                f"{self.ollama_base_url}/api/tags", timeout=5
            )

            if health_response.status_code != 200:
                raise Exception(f"Ollama returned status {health_response.status_code}")

            models = health_response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            print(f"✅ Ollama is running with {len(models)} models")

            # Check if specific model is available
            if model_id:
                clean_model_id = model_id.replace("ollama/", "")
                if not any(clean_model_id in name for name in model_names):
                    print(
                        f"⚠️ Model {clean_model_id} not found. Available models: {model_names}"
                    )
                else:
                    print(f"✅ Model {clean_model_id} is available")

            return {
                "inference_endpoint": f"{self.ollama_base_url}/api/chat",
                "health_endpoint": f"{self.ollama_base_url}/api/tags",
                "models_available": model_names,
            }

        except Exception as e:
            raise Exception(f"Failed to connect to local Ollama: {e}")

    def health_check(self) -> Dict[str, Any]:
        """Check Ollama health."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "healthy",
                    "provider": "local",
                    "ollama_url": self.ollama_base_url,
                    "models_count": len(data.get("models", [])),
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

            # Create parent directories if needed
            target.parent.mkdir(parents=True, exist_ok=True)

            print(f"📁 Copying {source} to {target}")
            shutil.copy2(source, target)

            print("✅ Artifact copied successfully")
            return True

        except Exception as e:
            print(f"⚠️ Local copy error: {e}")
            return False

    def download_artifact(self, remote_path: str, local_path: str) -> bool:
        """Copy artifact from local artifacts directory."""
        try:
            source = self.base_path / remote_path.lstrip("/")
            target = Path(local_path)

            if not source.exists():
                print(f"⚠️ Artifact not found: {source}")
                return False

            # Create parent directories if needed
            target.parent.mkdir(parents=True, exist_ok=True)

            print(f"📁 Copying {source} to {target}")
            shutil.copy2(source, target)

            print("✅ Artifact copied successfully")
            return True

        except Exception as e:
            print(f"⚠️ Local copy error: {e}")
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
            print(f"⚠️ Local list error: {e}")
            return []


# Register the Local providers
ProviderFactory.register_model_provider("local", LocalModelProvider)
ProviderFactory.register_artifact_provider("local", LocalArtifactProvider)
