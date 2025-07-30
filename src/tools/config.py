# src/tools/config.py
import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
import logging

class Config:
    """Singleton configuration manager for the multi-agent system."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.config_data = {}
        self.config_path = None
        self.logger = logging.getLogger(__name__)
        self.active_profile = None
        self.load_config()
        self._initialized = True
    
    def load_config(self):
        """Load configuration from environment variables and config files."""
        # Load from environment variables
        self.config_data = {
            # Agent URLs
            "text_agent_url": os.getenv("TEXT_AGENT_URL", "http://localhost:8002"),
            "video_agent_url": os.getenv("VIDEO_AGENT_URL", "http://localhost:8001"),
            "composing_agent_port": int(os.getenv("COMPOSING_AGENT_PORT", "8000")),
            
            # Search Backend Configuration
            "search_backend": os.getenv("SEARCH_BACKEND", "byaldi").lower(),
            
            # Elasticsearch Configuration
            "elastic_cloud_id": os.getenv("ELASTIC_CLOUD_ID"),
            "elastic_api_key": os.getenv("ELASTIC_API_KEY"),
            "elastic_index": os.getenv("ELASTIC_INDEX", "your-text-index"),
            
            # Vespa Configuration
            "vespa_url": os.getenv("VESPA_URL", "http://localhost"),
            "vespa_port": int(os.getenv("VESPA_PORT", "8080")),
            "vespa_schema": os.getenv("VESPA_SCHEMA", "video_content"),
            
            # Byaldi Configuration
            "byaldi_index_name": os.getenv("BYALDI_INDEX_NAME", ".byaldi/my_video_index"),
            
            # Model Configuration
            "vllm_model": os.getenv("VLLM_MODEL", "Qwen/Qwen2-VL-7B-Instruct"),
            "colpali_model": os.getenv("COLPALI_MODEL", "vidore/colsmol-500m"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            
            # Local LLM Configuration
            "local_llm_model": os.getenv("LOCAL_LLM_MODEL", "deepseek-r1:1.5b"),
            "base_url": os.getenv("BASE_URL", "http://localhost:11434"),
            
            # System Configuration
            "device": "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu",
            "max_workers": int(os.getenv("MAX_WORKERS", "4")),
            "timeout": float(os.getenv("TIMEOUT", "60.0")),
            
            # Logging Configuration
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "log_file": os.getenv("LOG_FILE", "multi_agent_system.log"),
            
            # Data Paths
            "video_data_dir": os.getenv("VIDEO_DATA_DIR", "data/videos"),
            "video_dir": os.getenv("VIDEO_DIR", "data/videos"),  # Alias for video_data_dir
            "text_data_dir": os.getenv("TEXT_DATA_DIR", "data/text"),
            "index_dir": os.getenv("INDEX_DIR", "data/indexes"),
            
            # Video Server Configuration
            "static_server_port": int(os.getenv("STATIC_SERVER_PORT", "8888")),
        }
        
        # Load from config file if it exists
        # Look for config.json in configs directory
        config_paths = [
            Path("configs/config.json"),  # Configs directory from current location
            Path(__file__).parent.parent.parent / "configs" / "config.json",  # Project configs dir
            Path.cwd() / "configs" / "config.json",  # Configs dir from current working directory
            Path.home() / ".cogniverse" / "config.json"  # User home directory
        ]
        
        self.config_path = None
        for path in config_paths:
            if path.exists():
                self.config_path = path
                break
        
        if self.config_path:
            try:
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                    self.config_data.update(file_config)
                    self.logger.info(f"Configuration loaded from: {self.config_path}")
            except Exception as e:
                self.logger.error(f"Could not load config file {self.config_path}: {e}")
                print(f"Warning: Could not load config file {self.config_path}: {e}")
        else:
            self.logger.warning(f"config.json not found in any of: {config_paths}")
            print(f"Warning: config.json not found in any of: {config_paths}")
        
        # Apply video processing profile if specified
        profile_name = os.getenv("VIDEO_PROFILE") or self.config_data.get("active_video_profile")
        if profile_name:
            self._apply_video_profile(profile_name)
    
    def _apply_video_profile(self, profile_name: str):
        """Apply video processing profile settings."""
        profiles = self.config_data.get("video_processing_profiles", {})
        if profile_name not in profiles:
            self.logger.warning(f"Video profile '{profile_name}' not found. Available profiles: {list(profiles.keys())}")
            return
        
        profile = profiles[profile_name]
        self.active_profile = profile_name
        
        # Override vespa_schema
        if "vespa_schema" in profile:
            self.config_data["vespa_schema"] = profile["vespa_schema"]
        
        # Override pipeline_config
        if "pipeline_config" in profile:
            self.config_data["pipeline_config"] = profile["pipeline_config"]
        
        # Override embedding model
        if "embedding_model" in profile:
            self.config_data["colpali_model"] = profile["embedding_model"]
        
        # Add profile-specific settings
        self.config_data["embedding_type"] = profile.get("embedding_type", "frame_based")
        
        if "model_specific" in profile:
            self.config_data["model_specific"] = profile["model_specific"]
        
        self.logger.info(f"Applied video processing profile: {profile_name}")
    
    def reload(self):
        """Reload configuration from file"""
        self.config_data = {}
        self.load_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value. Supports dot notation for nested values."""
        # Support dot notation for nested access
        keys = key.split('.')
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any):
        """Set a configuration value. Supports dot notation for nested values."""
        keys = key.split('.')
        
        # Navigate to the parent of the target key
        config = self.config_data
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        # Set the value
        config[keys[-1]] = value
    
    def get_agent_urls(self) -> Dict[str, str]:
        """Get all agent URLs."""
        return {
            "text_agent": self.get("text_agent_url"),
            "video_agent": self.get("video_agent_url"),
        }
    
    def get_elasticsearch_config(self) -> Dict[str, Any]:
        """Get Elasticsearch configuration."""
        return {
            "cloud_id": self.get("elastic_cloud_id"),
            "api_key": self.get("elastic_api_key"),
            "index": self.get("elastic_index"),
        }
    
    def get_vespa_config(self) -> Dict[str, Any]:
        """Get Vespa configuration."""
        return {
            "url": self.get("vespa_url"),
            "port": self.get("vespa_port"),
            "schema": self.get("vespa_schema"),
        }
    
    def get_byaldi_config(self) -> Dict[str, Any]:
        """Get Byaldi configuration."""
        return {
            "index_name": self.get("byaldi_index_name"),
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "vllm_model": self.get("vllm_model"),
            "colpali_model": self.get("colpali_model"),
            "embedding_model": self.get("embedding_model"),
            "device": self.get("device"),
        }
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return {
            "model": self.get("local_llm_model"),
            "base_url": self.get("base_url"),
        }
    
    def get_active_profile(self) -> Optional[str]:
        """Get the active video processing profile."""
        return self.active_profile
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration."""
        return self.get("pipeline_config", {})
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return os.getenv("ENVIRONMENT", "development").lower() == "development"
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    def validate_required_config(self) -> Dict[str, str]:
        """Validate required configuration and return missing items."""
        missing = {}
        
        # Check if local LLM server is running
        try:
            import requests
            response = requests.get(f"{self.get('base_url')}/api/tags", timeout=5)
            if response.status_code != 200:
                missing["llm_server"] = "Local LLM server not running. Start with: ollama serve"
        except Exception:
            missing["llm_server"] = "Local LLM server not accessible. Install Ollama and run: ollama serve"
        
        # Check search backend specific requirements
        if self.get("search_backend") == "vespa":
            if not self.get("vespa_url"):
                missing["vespa_url"] = "Required for Vespa backend"
        
        # Check Elasticsearch configuration if being used
        if self.get("elastic_cloud_id") and not self.get("elastic_api_key"):
            missing["elastic_api_key"] = "Required when using Elasticsearch Cloud"
        
        return missing
    
    # Convenience properties for commonly used configs
    @property
    def vespa_schema(self) -> str:
        """Get Vespa schema name"""
        return self.get("vespa_schema", "video_frame")
    
    @property
    def vespa_url(self) -> str:
        """Get Vespa URL"""
        return self.get("vespa_url", "http://localhost")
    
    @property
    def vespa_port(self) -> int:
        """Get Vespa port"""
        return self.get("vespa_port", 8080)
    
    @property
    def colpali_model(self) -> str:
        """Get ColPali model name"""
        return self.get("colpali_model", "vidore/colsmol-500m")
    
    @property
    def search_backend(self) -> str:
        """Get search backend"""
        return self.get("search_backend", "vespa")
    
    @property
    def log_level(self) -> str:
        """Get log level"""
        return self.get("log_level", "INFO")
    
    def save_config(self, filepath: str = "config.json"):
        """Save current configuration to a file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.config_data, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def print_config(self):
        """Print current configuration (excluding sensitive data)."""
        safe_config = self.config_data.copy()
        
        # Hide sensitive keys
        sensitive_keys = ["api_key", "password", "token", "secret"]
        for key in safe_config:
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                if safe_config[key]:
                    safe_config[key] = "***HIDDEN***"
        
        print("Current Configuration:")
        print(json.dumps(safe_config, indent=2))

# Global configuration instance
config = Config()

# Convenience functions
def get_config() -> Config:
    """Get the global configuration instance."""
    return config

def setup_environment():
    """Set up the environment based on configuration."""
    # Create necessary directories
    directories = [
        config.get("video_data_dir"),
        config.get("text_data_dir"),
        config.get("index_dir"),
    ]
    
    for directory in directories:
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Set environment variables for models
    if config.get("device") == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    
    # Validate configuration
    missing = config.validate_required_config()
    if missing:
        print("Warning: Missing required configuration:")
        for key, description in missing.items():
            print(f"  - {key}: {description}")
        print("Please set these environment variables or add them to config.json")
    
    return config 