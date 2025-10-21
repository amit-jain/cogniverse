"""
Base Provider Interfaces

Abstract base classes for different provider types to enable
clean separation and future extensibility to AWS, GCP, etc.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import dspy


class ModelProvider(ABC):
    """Abstract base class for model hosting providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def call_model(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 150
    ) -> str:
        """
        Call a model through the provider's infrastructure.
        
        Args:
            model_id: Model identifier
            prompt: Input prompt
            system_prompt: Optional system prompt
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def deploy_model_service(self, model_id: str, **kwargs) -> Dict[str, str]:
        """
        Deploy a model service.
        
        Args:
            model_id: Model to deploy
            **kwargs: Provider-specific configuration
            
        Returns:
            Dictionary with service endpoints/URLs
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Check provider health and status.
        
        Returns:
            Health status information
        """
        pass


class ArtifactProvider(ABC):
    """Abstract base class for artifact storage providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def upload_artifact(self, local_path: str, remote_path: str) -> bool:
        """
        Upload an artifact to the provider's storage.
        
        Args:
            local_path: Path to local artifact file
            remote_path: Target path in provider storage
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def download_artifact(self, remote_path: str, local_path: str) -> bool:
        """
        Download an artifact from the provider's storage.
        
        Args:
            remote_path: Path in provider storage
            local_path: Target local path
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def list_artifacts(self, path_prefix: str = "") -> List[str]:
        """
        List available artifacts.
        
        Args:
            path_prefix: Optional path prefix to filter
            
        Returns:
            List of artifact paths
        """
        pass


class DSPyLMProvider(dspy.LM):
    """
    DSPy LM wrapper that uses a ModelProvider.
    
    This allows any ModelProvider to be used with DSPy's optimization.
    """
    
    def __init__(self, model_provider: ModelProvider, model_id: str, model_type: str = ""):
        super().__init__(model=f"{model_type}_{model_id}")
        self.provider = model_provider
        self.model_id = model_id
        self.model_type = model_type
        
    def basic_generate(self, prompt, **kwargs):
        """Generate using the model provider."""
        temperature = kwargs.get("temperature", 0.1)
        max_tokens = kwargs.get("max_tokens", 150)
        
        response = self.provider.call_model(
            model_id=self.model_id,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return [response]


class ProviderFactory:
    """Factory for creating providers based on configuration."""
    
    _model_providers = {}
    _artifact_providers = {}
    
    @classmethod
    def register_model_provider(cls, name: str, provider_class: type):
        """Register a model provider class."""
        cls._model_providers[name] = provider_class
    
    @classmethod
    def register_artifact_provider(cls, name: str, provider_class: type):
        """Register an artifact provider class."""
        cls._artifact_providers[name] = provider_class
    
    @classmethod
    def create_model_provider(cls, provider_type: str, config: Dict[str, Any]) -> ModelProvider:
        """Create a model provider instance."""
        if provider_type not in cls._model_providers:
            raise ValueError(f"Unknown model provider: {provider_type}")
        
        return cls._model_providers[provider_type](config)
    
    @classmethod
    def create_artifact_provider(cls, provider_type: str, config: Dict[str, Any]) -> ArtifactProvider:
        """Create an artifact provider instance."""
        if provider_type not in cls._artifact_providers:
            raise ValueError(f"Unknown artifact provider: {provider_type}")
        
        return cls._artifact_providers[provider_type](config)
