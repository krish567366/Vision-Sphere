"""
Configuration Management for VisionAgent Framework
Centralized configuration handling with environment variable support.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for individual models."""
    name: str
    path: Optional[str] = None
    device: Optional[str] = None
    confidence_threshold: float = 0.5
    batch_size: int = 1
    custom_params: Optional[Dict[str, Any]] = None


@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    enabled: bool = True
    model: Optional[ModelConfig] = None
    processing_params: Optional[Dict[str, Any]] = None


@dataclass
class ServerConfig:
    """Configuration for the API server."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_file_size_mb: int = 100
    cors_origins: List[str] = None
    enable_websocket: bool = True
    rate_limit_per_minute: int = 60


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size_mb: int = 10
    backup_count: int = 5


@dataclass
class VisionAgentConfig:
    """Main configuration class for VisionAgent framework."""
    
    # Global settings
    default_device: str = "auto"
    model_cache_dir: str = "./models"
    temp_dir: str = "./temp"
    
    # Agent configurations
    face_agent: AgentConfig = None
    object_agent: AgentConfig = None
    video_agent: AgentConfig = None
    classification_agent: AgentConfig = None
    
    # Server configuration
    server: ServerConfig = None
    
    # Logging configuration
    logging: LoggingConfig = None
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.face_agent is None:
            self.face_agent = AgentConfig(
                model=ModelConfig(
                    name="face_recognition",
                    custom_params={
                        'face_detection_model': 'hog',
                        'num_jitters': 1,
                        'tolerance': 0.6
                    }
                )
            )
        
        if self.object_agent is None:
            self.object_agent = AgentConfig(
                model=ModelConfig(
                    name="yolov8s.pt",
                    confidence_threshold=0.5,
                    custom_params={
                        'iou_threshold': 0.45,
                        'max_detections': 100
                    }
                )
            )
        
        if self.video_agent is None:
            self.video_agent = AgentConfig(
                processing_params={
                    'frame_skip': 1,
                    'max_frames': 1000,
                    'track_objects': True,
                    'track_faces': True
                }
            )
        
        if self.classification_agent is None:
            self.classification_agent = AgentConfig(
                model=ModelConfig(
                    name="microsoft/resnet-50",
                    custom_params={
                        'top_k': 5,
                        'threshold': 0.1,
                        'return_features': False
                    }
                )
            )
        
        if self.server is None:
            self.server = ServerConfig()
        
        if self.logging is None:
            self.logging = LoggingConfig()


class ConfigManager:
    """Manages configuration loading, saving, and environment variable integration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or self._get_default_config_path()
        self._config: Optional[VisionAgentConfig] = None
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        # Check environment variable first
        env_path = os.getenv('VISIONAGENT_CONFIG')
        if env_path and os.path.exists(env_path):
            return env_path
        
        # Check common locations
        possible_paths = [
            './config.yaml',
            './config.yml', 
            './config.json',
            './configs/config.yaml',
            os.path.expanduser('~/.visionagent/config.yaml')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return './config.yaml'  # Default
    
    def load_config(self) -> VisionAgentConfig:
        """
        Load configuration from file with environment variable overrides.
        
        Returns:
            VisionAgentConfig object
        """
        if self._config is not None:
            return self._config
        
        # Load from file if exists
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.endswith('.json'):
                    data = json.load(f)
                else:
                    data = yaml.safe_load(f)
            
            # Convert to config object
            self._config = self._dict_to_config(data)
        else:
            # Use defaults
            self._config = VisionAgentConfig()
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Ensure directories exist
        self._ensure_directories()
        
        return self._config
    
    def save_config(self, config: VisionAgentConfig) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration object to save
        """
        # Ensure directory exists
        config_dir = os.path.dirname(self.config_path)
        if config_dir:
            os.makedirs(config_dir, exist_ok=True)
        
        # Convert to dictionary
        config_dict = asdict(config)
        
        # Save to file
        with open(self.config_path, 'w', encoding='utf-8') as f:
            if self.config_path.endswith('.json'):
                json.dump(config_dict, f, indent=2)
            else:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def _dict_to_config(self, data: Dict[str, Any]) -> VisionAgentConfig:
        """Convert dictionary to configuration object."""
        # Helper function to create nested config objects
        def create_model_config(model_data):
            if model_data is None:
                return None
            return ModelConfig(**model_data)
        
        def create_agent_config(agent_data):
            if agent_data is None:
                return None
            model_data = agent_data.get('model')
            return AgentConfig(
                enabled=agent_data.get('enabled', True),
                model=create_model_config(model_data),
                processing_params=agent_data.get('processing_params')
            )
        
        # Create configuration object
        config = VisionAgentConfig()
        
        # Global settings
        config.default_device = data.get('default_device', config.default_device)
        config.model_cache_dir = data.get('model_cache_dir', config.model_cache_dir)
        config.temp_dir = data.get('temp_dir', config.temp_dir)
        
        # Agent configurations
        config.face_agent = create_agent_config(data.get('face_agent'))
        config.object_agent = create_agent_config(data.get('object_agent'))
        config.video_agent = create_agent_config(data.get('video_agent'))
        config.classification_agent = create_agent_config(data.get('classification_agent'))
        
        # Server configuration
        server_data = data.get('server', {})
        config.server = ServerConfig(**server_data)
        
        # Logging configuration
        logging_data = data.get('logging', {})
        config.logging = LoggingConfig(**logging_data)
        
        return config
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        if not self._config:
            return
        
        # Global overrides
        if os.getenv('VISIONAGENT_DEVICE'):
            self._config.default_device = os.getenv('VISIONAGENT_DEVICE')
        
        if os.getenv('VISIONAGENT_MODEL_CACHE_DIR'):
            self._config.model_cache_dir = os.getenv('VISIONAGENT_MODEL_CACHE_DIR')
        
        # Server overrides
        if os.getenv('VISIONAGENT_HOST'):
            self._config.server.host = os.getenv('VISIONAGENT_HOST')
        
        if os.getenv('VISIONAGENT_PORT'):
            self._config.server.port = int(os.getenv('VISIONAGENT_PORT'))
        
        # Logging overrides
        if os.getenv('VISIONAGENT_LOG_LEVEL'):
            self._config.logging.level = os.getenv('VISIONAGENT_LOG_LEVEL')
        
        if os.getenv('VISIONAGENT_LOG_FILE'):
            self._config.logging.file_path = os.getenv('VISIONAGENT_LOG_FILE')
    
    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        if not self._config:
            return
        
        directories = [
            self._config.model_cache_dir,
            self._config.temp_dir
        ]
        
        for directory in directories:
            if directory:
                os.makedirs(directory, exist_ok=True)
    
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """
        Get configuration for a specific agent.
        
        Args:
            agent_name: Name of the agent ('face', 'object', 'video', 'classification')
            
        Returns:
            AgentConfig object or None if not found
        """
        if not self._config:
            self.load_config()
        
        agent_configs = {
            'face': self._config.face_agent,
            'object': self._config.object_agent,
            'video': self._config.video_agent,
            'classification': self._config.classification_agent
        }
        
        return agent_configs.get(agent_name)
    
    def update_agent_config(self, agent_name: str, config: AgentConfig) -> None:
        """
        Update configuration for a specific agent.
        
        Args:
            agent_name: Name of the agent
            config: New agent configuration
        """
        if not self._config:
            self.load_config()
        
        if agent_name == 'face':
            self._config.face_agent = config
        elif agent_name == 'object':
            self._config.object_agent = config
        elif agent_name == 'video':
            self._config.video_agent = config
        elif agent_name == 'classification':
            self._config.classification_agent = config


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> VisionAgentConfig:
    """Get the global configuration."""
    return config_manager.load_config()


def get_agent_config(agent_name: str) -> Optional[AgentConfig]:
    """Get configuration for a specific agent."""
    return config_manager.get_agent_config(agent_name)


def save_config(config: VisionAgentConfig) -> None:
    """Save configuration to file."""
    config_manager.save_config(config)
