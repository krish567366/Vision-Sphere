"""
Advanced Configuration System
Hierarchical configuration management inspired by Youtu-agent's Pydantic + Hydra approach.
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, List, Union, Type
from dataclasses import dataclass, field
from pathlib import Path
from pydantic import BaseModel, Field, validator
from abc import ABC, abstractmethod


class ConfigBaseModel(BaseModel):
    """Base configuration model with common functionality."""
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Prevent unknown fields
        validate_assignment = True
        use_enum_values = True


class ModelProfile(ConfigBaseModel):
    """Configuration for individual AI models."""
    name: str = Field(..., description="Model name or identifier")
    path: Optional[str] = Field(None, description="Local model path")
    device: str = Field("auto", description="Target device")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold")
    batch_size: int = Field(1, gt=0, description="Batch size for inference")
    cache_enabled: bool = Field(True, description="Enable result caching")
    custom_params: Dict[str, Any] = Field(default_factory=dict, description="Custom model parameters")
    
    @validator('device')
    def validate_device(cls, v):
        """Validate device specification."""
        allowed_devices = ['auto', 'cpu', 'cuda', 'mps']
        if v not in allowed_devices:
            raise ValueError(f"Device must be one of {allowed_devices}")
        return v


class ToolkitConfig(ConfigBaseModel):
    """Configuration for agent toolkits."""
    mode: str = Field("builtin", description="Toolkit mode")
    activated_tools: Optional[List[str]] = Field(None, description="List of activated tool names")
    config: Dict[str, Any] = Field(default_factory=dict, description="Toolkit-specific configuration")
    cache_config: Dict[str, Any] = Field(default_factory=dict, description="Caching configuration for tools")


class AgentProfile(ConfigBaseModel):
    """Configuration profile for individual agents."""
    name: str = Field(..., description="Agent name")
    agent_type: str = Field(..., description="Agent type (face, object, video, classification)")
    instructions: str = Field("", description="Agent instructions or system prompt")
    model: ModelProfile = Field(..., description="Model configuration")
    toolkits: Dict[str, ToolkitConfig] = Field(default_factory=dict, description="Available toolkits")
    max_iterations: int = Field(10, gt=0, description="Maximum processing iterations")
    timeout_seconds: int = Field(300, gt=0, description="Processing timeout")
    parallel_processing: bool = Field(False, description="Enable parallel processing")
    
    @validator('agent_type')
    def validate_agent_type(cls, v):
        """Validate agent type."""
        allowed_types = ['face', 'object', 'video', 'classification']
        if v not in allowed_types:
            raise ValueError(f"Agent type must be one of {allowed_types}")
        return v


class EnvironmentConfig(ConfigBaseModel):
    """Configuration for processing environments."""
    type: str = Field("local", description="Environment type")
    docker_config: Optional[Dict[str, Any]] = Field(None, description="Docker configuration")
    resource_limits: Dict[str, Any] = Field(default_factory=dict, description="Resource limits")
    cleanup_on_exit: bool = Field(True, description="Cleanup resources on exit")


class TracingConfig(ConfigBaseModel):
    """Configuration for tracing and observability."""
    enabled: bool = Field(True, description="Enable tracing")
    processors: List[str] = Field(default_factory=lambda: ["console"], description="Active tracing processors")
    console_verbose: bool = Field(False, description="Verbose console logging")
    file_output: Optional[str] = Field(None, description="File output path for traces")
    streaming_enabled: bool = Field(False, description="Enable streaming traces")
    span_attributes: Dict[str, Any] = Field(default_factory=dict, description="Default span attributes")


class PerformanceConfig(ConfigBaseModel):
    """Configuration for performance optimizations."""
    cache_expire_time: int = Field(3600, gt=0, description="Cache expiration time in seconds")
    max_cache_size_mb: int = Field(1000, gt=0, description="Maximum cache size in MB")
    async_concurrency_limit: int = Field(10, gt=0, description="Maximum concurrent async operations")
    batch_processing_enabled: bool = Field(True, description="Enable batch processing")
    memory_optimization: bool = Field(True, description="Enable memory optimizations")


class SecurityConfig(ConfigBaseModel):
    """Configuration for security settings."""
    api_key_masking: bool = Field(True, description="Mask API keys in logs")
    secure_temp_files: bool = Field(True, description="Use secure temporary files")
    sandbox_mode: bool = Field(False, description="Enable sandbox mode")
    allowed_file_types: List[str] = Field(
        default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.mp4', '.avi', '.mov'],
        description="Allowed file extensions"
    )
    max_file_size_mb: int = Field(100, gt=0, description="Maximum file size in MB")


class VisionAgentConfig(ConfigBaseModel):
    """Complete Vision Agent configuration with all subsystems."""
    
    # Agent configurations
    agents: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Agent-specific configurations"
    )
    
    # System configurations
    system: Dict[str, Any] = Field(
        default_factory=dict,
        description="System-level configuration"
    )
    
    # Tracing configuration
    tracing: TracingConfig = Field(
        default_factory=TracingConfig,
        description="Tracing and observability configuration"
    )
    
    # Caching configuration  
    caching: Dict[str, Any] = Field(
        default_factory=dict,
        description="Caching system configuration"
    )


class HierarchicalConfig:
    """Simple hierarchical configuration manager."""
    
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self.data = data or {}
    
    def load_from_file(self, file_path: str):
        """Load configuration from file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                self.data = yaml.safe_load(f)
            elif path.suffix == '.json':
                self.data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        self._deep_update(self.data, updates)
    
    def _deep_update(self, base: dict, updates: dict):
        """Deep update dictionary."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.data.copy()


class VisionAgentConfig(ConfigBaseModel):
    """
    Main configuration class for VisionAgent framework.
    Hierarchical configuration with composition support.
    """
    
    # Meta configuration
    config_version: str = Field("1.0", description="Configuration version")
    config_name: str = Field("default", description="Configuration profile name")
    
    # Global settings
    default_device: str = Field("auto", description="Default processing device")
    model_cache_dir: str = Field("./models", description="Model cache directory")
    temp_dir: str = Field("./temp", description="Temporary files directory")
    
    # Component configurations
    agents: Dict[str, AgentProfile] = Field(default_factory=dict, description="Agent configurations")
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig, description="Environment configuration")
    tracing: TracingConfig = Field(default_factory=TracingConfig, description="Tracing configuration")
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig, description="Performance settings")
    security: SecurityConfig = Field(default_factory=SecurityConfig, description="Security settings")
    
    # Server configuration (from previous implementation)
    server: Dict[str, Any] = Field(default_factory=lambda: {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,
        "cors_origins": ["*"],
        "enable_websocket": True
    }, description="Server configuration")
    
    def get_agent_config(self, agent_name: str) -> Optional[AgentProfile]:
        """Get configuration for a specific agent."""
        return self.agents.get(agent_name)
    
    def add_agent_config(self, agent_name: str, config: AgentProfile):
        """Add or update agent configuration."""
        self.agents[agent_name] = config
    
    def validate_config(self) -> List[str]:
        """
        Validate the entire configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate directories exist or can be created
        for dir_path in [self.model_cache_dir, self.temp_dir]:
            try:
                os.makedirs(dir_path, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create directory {dir_path}: {str(e)}")
        
        # Validate agent configurations
        for agent_name, agent_config in self.agents.items():
            if not agent_config.model:
                errors.append(f"Agent {agent_name} missing model configuration")
        
        # Validate server configuration
        server_port = self.server.get('port', 8000)
        if not isinstance(server_port, int) or server_port < 1 or server_port > 65535:
            errors.append(f"Invalid server port: {server_port}")
        
        return errors


class ConfigLoader:
    """
    Advanced configuration loader with composition support.
    Inspired by Youtu-agent's Hydra integration.
    """
    
    def __init__(self, config_dir: str = "./configs"):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Base directory for configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, VisionAgentConfig] = {}
    
    def load_config(self, 
                   config_name: str = "default",
                   overrides: Optional[Dict[str, Any]] = None) -> VisionAgentConfig:
        """
        Load configuration with composition and overrides.
        
        Args:
            config_name: Configuration profile name
            overrides: Configuration overrides
            
        Returns:
            Loaded configuration object
        """
        # Check cache first
        cache_key = f"{config_name}:{hash(str(overrides))}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Load base configuration
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        else:
            # Create default configuration
            config_data = self._create_default_config()
            self.save_config(config_name, VisionAgentConfig(**config_data))
        
        # Apply overrides
        if overrides:
            config_data = self._deep_merge(config_data, overrides)
        
        # Apply environment variable overrides
        config_data = self._apply_env_overrides(config_data)
        
        # Create configuration object
        config = VisionAgentConfig(**config_data)
        
        # Cache the result
        self._cache[cache_key] = config
        
        return config
    
    def save_config(self, config_name: str, config: VisionAgentConfig):
        """
        Save configuration to file.
        
        Args:
            config_name: Configuration profile name
            config: Configuration object to save
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        # Convert to dictionary
        config_dict = config.dict()
        
        # Save to YAML
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def list_configs(self) -> List[str]:
        """List available configuration profiles."""
        configs = []
        for config_file in self.config_dir.glob("*.yaml"):
            configs.append(config_file.stem)
        return configs
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration structure."""
        return {
            "config_version": "1.0",
            "config_name": "default",
            "default_device": "auto",
            "model_cache_dir": "./models",
            "temp_dir": "./temp",
            "agents": {
                "face": {
                    "name": "face_agent",
                    "agent_type": "face",
                    "instructions": "Face detection and recognition agent",
                    "model": {
                        "name": "opencv_face_detector",
                        "confidence_threshold": 0.7,
                        "custom_params": {
                            "nms_threshold": 0.4,
                            "input_size": [300, 300]
                        }
                    }
                },
                "object": {
                    "name": "object_agent", 
                    "agent_type": "object",
                    "instructions": "Object detection and classification agent",
                    "model": {
                        "name": "yolov8s.pt",
                        "confidence_threshold": 0.5,
                        "custom_params": {
                            "iou_threshold": 0.45,
                            "max_detections": 100
                        }
                    }
                },
                "video": {
                    "name": "video_agent",
                    "agent_type": "video", 
                    "instructions": "Video analysis and tracking agent",
                    "model": {
                        "name": "video_processor",
                        "custom_params": {
                            "frame_skip": 1,
                            "max_frames": 1000,
                            "track_objects": True,
                            "track_faces": True
                        }
                    }
                },
                "classification": {
                    "name": "classification_agent",
                    "agent_type": "classification",
                    "instructions": "Image classification agent",
                    "model": {
                        "name": "microsoft/resnet-50",
                        "confidence_threshold": 0.1,
                        "custom_params": {
                            "top_k": 5,
                            "return_features": False
                        }
                    }
                }
            },
            "tracing": {
                "enabled": True,
                "processors": ["console"],
                "console_verbose": False
            },
            "performance": {
                "cache_expire_time": 3600,
                "max_cache_size_mb": 1000,
                "async_concurrency_limit": 10
            }
        }
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides.
        
        Args:
            config_data: Configuration data
            
        Returns:
            Configuration with environment overrides applied
        """
        # Global environment variables
        env_mappings = {
            'VISIONAGENT_DEVICE': 'default_device',
            'VISIONAGENT_MODEL_CACHE_DIR': 'model_cache_dir', 
            'VISIONAGENT_TEMP_DIR': 'temp_dir',
            'VISIONAGENT_HOST': 'server.host',
            'VISIONAGENT_PORT': 'server.port',
            'VISIONAGENT_LOG_LEVEL': 'tracing.console_verbose'
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                # Navigate to nested configuration
                keys = config_path.split('.')
                current = config_data
                
                # Navigate to parent
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Set the value with type conversion
                final_key = keys[-1]
                if env_var == 'VISIONAGENT_PORT':
                    current[final_key] = int(env_value)
                elif env_var == 'VISIONAGENT_LOG_LEVEL':
                    current[final_key] = env_value.lower() == 'true'
                else:
                    current[final_key] = env_value
        
        return config_data


class ConfigComposer:
    """
    Configuration composition system for building complex configs.
    Supports inheritance and mixing of configuration profiles.
    """
    
    def __init__(self, loader: ConfigLoader):
        self.loader = loader
    
    def compose_config(self, 
                      base_config: str = "default",
                      mixins: Optional[List[str]] = None,
                      overrides: Optional[Dict[str, Any]] = None) -> VisionAgentConfig:
        """
        Compose configuration from multiple sources.
        
        Args:
            base_config: Base configuration name
            mixins: List of mixin configuration names
            overrides: Direct overrides
            
        Returns:
            Composed configuration
        """
        # Load base configuration
        config = self.loader.load_config(base_config)
        config_dict = config.dict()
        
        # Apply mixins
        if mixins:
            for mixin_name in mixins:
                mixin_config = self.loader.load_config(mixin_name)
                config_dict = self.loader._deep_merge(config_dict, mixin_config.dict())
        
        # Apply direct overrides
        if overrides:
            config_dict = self.loader._deep_merge(config_dict, overrides)
        
        return VisionAgentConfig(**config_dict)


# Factory functions for creating standard configurations
def create_production_config() -> VisionAgentConfig:
    """Create production-ready configuration."""
    loader = ConfigLoader()
    
    production_overrides = {
        "tracing": {
            "enabled": True,
            "processors": ["console", "file"],
            "file_output": "./logs/production_traces.jsonl"
        },
        "performance": {
            "cache_expire_time": 7 * 24 * 3600,  # 7 days
            "max_cache_size_mb": 5000,  # 5GB
            "async_concurrency_limit": 50
        },
        "security": {
            "sandbox_mode": True,
            "max_file_size_mb": 500
        },
        "server": {
            "workers": 4,
            "cors_origins": []  # Restrict CORS in production
        }
    }
    
    return loader.load_config("default", production_overrides)


def create_development_config() -> VisionAgentConfig:
    """Create development-friendly configuration."""
    loader = ConfigLoader()
    
    dev_overrides = {
        "tracing": {
            "console_verbose": True,
            "streaming_enabled": True
        },
        "performance": {
            "cache_expire_time": 300,  # 5 minutes for fast iteration
            "async_concurrency_limit": 5
        },
        "server": {
            "port": 8001,  # Different port for dev
            "cors_origins": ["*"]  # Allow all origins in dev
        }
    }
    
    return loader.load_config("default", dev_overrides)


def create_benchmark_config() -> VisionAgentConfig:
    """Create configuration optimized for benchmarking."""
    loader = ConfigLoader()
    
    benchmark_overrides = {
        "tracing": {
            "enabled": True,
            "processors": ["file"],
            "file_output": "./benchmark_traces.jsonl"
        },
        "performance": {
            "batch_processing_enabled": True,
            "memory_optimization": True,
            "async_concurrency_limit": 20
        },
        "agents": {
            "object": {
                "model": {
                    "batch_size": 8,
                    "confidence_threshold": 0.3  # Lower threshold for recall
                }
            }
        }
    }
    
    return loader.load_config("default", benchmark_overrides)


# Global configuration instances
_config_loader = ConfigLoader()
_config_composer = ConfigComposer(_config_loader)


def get_config(profile: str = "default", 
              mixins: Optional[List[str]] = None,
              overrides: Optional[Dict[str, Any]] = None) -> VisionAgentConfig:
    """
    Get configuration with composition support.
    
    Args:
        profile: Configuration profile name
        mixins: Mixin configurations to apply
        overrides: Direct configuration overrides
        
    Returns:
        Composed configuration
    """
    return _config_composer.compose_config(profile, mixins, overrides)


def save_config(profile: str, config: VisionAgentConfig):
    """Save configuration profile."""
    _config_loader.save_config(profile, config)


def list_config_profiles() -> List[str]:
    """List available configuration profiles."""
    return _config_loader.list_configs()


# Global configuration instance
_global_config: Optional[HierarchicalConfig] = None


def get_hierarchical_config() -> Dict[str, Any]:
    """
    Get the global hierarchical configuration.
    
    Returns:
        Configuration dictionary
    """
    global _global_config
    
    if _global_config is None:
        # Load default configuration
        _global_config = HierarchicalConfig()
        
        # Try to load from config files
        config_files = ['config.yaml', 'config.yml', 'config.json']
        
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    _global_config.load_from_file(config_file)
                    break
                except Exception as e:
                    logging.warning(f"Failed to load {config_file}: {e}")
    
    return _global_config.to_dict()


def update_global_config(updates: Dict[str, Any]):
    """Update global configuration."""
    global _global_config
    if _global_config is None:
        _global_config = HierarchicalConfig()
    
    _global_config.update_config(updates)
