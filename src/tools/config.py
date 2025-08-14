# src/tools/config.py
"""
Config wrapper that maintains backward compatibility.
"""
import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
import logging

# Try to use the common config module if available
try:
    from cogniverse_common import get_config as common_get_config, Config as CommonConfig
    
    # Use the common config directly
    Config = CommonConfig
    get_config = common_get_config
    
except ImportError:
    # Fallback to original implementation if common module not available
    
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
                
                # LLM Configuration
                "local_llm_model": os.getenv("LOCAL_LLM_MODEL", "llama3.2:latest"),
                "base_url": os.getenv("BASE_URL", "http://localhost:11434"),
                
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
        
        def get(self, key: str, default: Any = None) -> Any:
            """Get configuration value."""
            return self.config_data.get(key, default)
        
        def update(self, updates: Dict[str, Any]):
            """Update configuration."""
            self.config_data.update(updates)
        
        def to_dict(self) -> Dict[str, Any]:
            """Get configuration as dictionary."""
            return self.config_data.copy()
    
    # Create singleton instance
    _config = Config()
    
    def get_config() -> Dict[str, Any]:
        """Get configuration dictionary."""
        return _config.to_dict()