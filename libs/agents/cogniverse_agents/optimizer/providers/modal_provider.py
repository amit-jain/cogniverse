"""
Modal Provider Implementation

Implements the provider interfaces for Modal infrastructure.
This contains all the existing Modal-specific logic.
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, List

import requests

from cogniverse_core.config.utils import get_config

from .base_provider import ArtifactProvider, ModelProvider, ProviderFactory


class ModalModelProvider(ModelProvider):
    """Modal implementation of ModelProvider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.deployed_services = {}
    
    def call_model(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 150
    ) -> str:
        """Call a model hosted on Modal."""
        if "inference_endpoint" not in self.deployed_services:
            raise Exception("Modal model service not deployed")
        
        # Convert to OpenAI-compatible format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        request_data = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            self.deployed_services["inference_endpoint"],
            json=request_data,
            timeout=120
        )
        
        if response.status_code != 200:
            raise Exception(f"Modal model call failed: {response.text}")
        
        result = response.json()
        
        # Extract text from OpenAI-compatible response
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Unexpected response format: {result}")
    
    def deploy_model_service(self, model_id: str = None, **kwargs) -> Dict[str, str]:
        """Deploy the Modal model service."""
        # Check if we already have a deployed endpoint in config
        config = get_config()
        
        existing_endpoint = config.get('inference.modal_endpoint')
        if existing_endpoint:
            print(f"📍 Using existing Modal endpoint from config: {existing_endpoint}")
            
            # Test if it's still working
            try:
                health_url = existing_endpoint.replace("--generate", "--health")
                response = requests.get(health_url, timeout=10)
                if response.status_code == 200:
                    print("✅ Existing Modal endpoint is healthy")
                    self.deployed_services["inference_endpoint"] = existing_endpoint
                    self.deployed_services["health_endpoint"] = health_url
                    return self.deployed_services
            except Exception as e:
                print(f"⚠️ Existing endpoint not responding, will redeploy: {e}")
        
        print("🚀 Deploying Modal model service...")
        
        try:
            # Deploy the modal inference service
            result = subprocess.run([
                "modal", "deploy", "src/inference/modal_inference_service.py"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent)
            
            if result.returncode != 0:
                raise Exception(f"Modal deployment failed: {result.stderr}")
            
            print("✅ Modal model service deployed successfully")
            
            # Extract URLs from deployment output
            service_urls = {}
            output_lines = result.stdout.split('\n')
            
            # Sometimes URLs are on the next line after =>
            i = 0
            while i < len(output_lines):
                line = output_lines[i]
                
                if "=>" in line and "Created web function" in line:
                    # Get the URL - it might be on this line or the next
                    if "https://" in line:
                        url = line.split("=>")[-1].strip()
                    elif i + 1 < len(output_lines) and "https://" in output_lines[i + 1]:
                        url = output_lines[i + 1].strip()
                        i += 1  # Skip next line since we used it
                    else:
                        url = None
                    
                    if url:
                        # Clean up the URL - remove tree characters and whitespace
                        url = url.strip()
                        # Remove common tree/box drawing characters
                        for char in ['│', '├', '└', '─', '🔨']:
                            url = url.replace(char, '')
                        url = url.strip()
                        
                        # Ensure it's a valid URL
                        if url.startswith('https://'):
                            # Determine which endpoint this is
                            if "serve" in line or "--general-inference-service-serve" in url:
                                service_urls["inference_endpoint"] = url  # vLLM serve endpoint
                            elif "health_check" in line or "--health" in url:
                                service_urls["health_endpoint"] = url
                            elif "list_models" in line or "--models" in url:
                                service_urls["list_models_endpoint"] = url
                
                i += 1
            
            # Validate we got all required URLs
            if not service_urls.get("inference_endpoint"):
                print("⚠️ Could not parse inference endpoint from deployment output")
                print("Deployment output:")
                print(result.stdout)
                raise Exception("Failed to extract Modal inference endpoint")
            
            print("📍 Deployed endpoints:")
            for name, url in service_urls.items():
                print(f"   {name}: {url}")
            
            # Update config with the deployed endpoint
            try:
                config = get_config()
                
                # Update the modal endpoint in config
                config.set('inference.modal_endpoint', service_urls["inference_endpoint"])
                
                # Save back to config file
                config.save()
                
                print(f"✅ Updated config.json with Modal endpoint: {service_urls['inference_endpoint']}")
            except Exception as e:
                print(f"⚠️ Could not update config.json: {e}")
            
            # Test the service
            print("🧪 Testing deployed service...")
            try:
                health_response = requests.get(service_urls["health_endpoint"], timeout=30)
                if health_response.status_code == 200:
                    print("✅ Service health check passed")
                else:
                    print(f"⚠️ Service health check failed: {health_response.status_code}")
            except Exception as e:
                print(f"⚠️ Health check error: {e}")
            
            self.deployed_services.update(service_urls)
            return service_urls
            
        except Exception as e:
            raise Exception(f"Failed to deploy Modal model service: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check Modal service health."""
        if "health_endpoint" not in self.deployed_services:
            return {"status": "not_deployed", "provider": "modal"}
        
        try:
            response = requests.get(self.deployed_services["health_endpoint"], timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                health_data["provider"] = "modal"
                return health_data
            else:
                return {
                    "status": "unhealthy",
                    "provider": "modal",
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "error",
                "provider": "modal", 
                "error": str(e)
            }


class ModalArtifactProvider(ArtifactProvider):
    """Modal implementation of ArtifactProvider using Modal volumes."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.volume_name = config.get("volume_name", "optimization-artifacts")
    
    def upload_artifact(self, local_path: str, remote_path: str) -> bool:
        """Upload artifact to Modal volume."""
        try:
            print(f"📤 Uploading {local_path} to Modal volume {self.volume_name}:{remote_path}")
            
            cmd = [
                "modal", "volume", "put",
                self.volume_name,
                local_path,
                remote_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"⚠️ Modal upload failed: {result.stderr}")
                return False
            
            print("✅ Artifact uploaded to Modal volume successfully")
            return True
            
        except Exception as e:
            print(f"⚠️ Modal upload error: {e}")
            return False
    
    def download_artifact(self, remote_path: str, local_path: str) -> bool:
        """Download artifact from Modal volume."""
        try:
            print(f"📥 Downloading {self.volume_name}:{remote_path} to {local_path}")
            
            cmd = [
                "modal", "volume", "get",
                self.volume_name,
                remote_path,
                local_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"⚠️ Modal download failed: {result.stderr}")
                return False
            
            print("✅ Artifact downloaded from Modal volume successfully")
            return True
            
        except Exception as e:
            print(f"⚠️ Modal download error: {e}")
            return False
    
    def list_artifacts(self, path_prefix: str = "") -> List[str]:
        """List artifacts in Modal volume."""
        try:
            cmd = ["modal", "volume", "ls", self.volume_name]
            if path_prefix:
                cmd.append(path_prefix)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"⚠️ Modal list failed: {result.stderr}")
                return []
            
            # Parse the output (simplified)
            files = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    files.append(line.strip())
            
            return files
            
        except Exception as e:
            print(f"⚠️ Modal list error: {e}")
            return []


# Register the Modal providers
ProviderFactory.register_model_provider("modal", ModalModelProvider)
ProviderFactory.register_artifact_provider("modal", ModalArtifactProvider)
