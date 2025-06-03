"""
Configuration management for Open3D reconstruction platform.

This module provides centralized configuration management with support for
multiple sources (files, environment variables, defaults) and validation.
"""

import os
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import yaml
import json
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, validator
from loguru import logger

from .types import DeviceType, RenderingBackend


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_per_minute: int = 100
    max_request_size: int = 100 * 1024 * 1024  # 100MB
    timeout: int = 300  # 5 minutes
    workers: int = 1


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = "sqlite:///open3d.db"
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class RedisConfig:
    """Redis configuration."""
    url: str = "redis://localhost:6379/0"
    max_connections: int = 10
    socket_timeout: int = 30


@dataclass
class OpenAIConfig:
    """OpenAI API configuration."""
    api_key: str = ""
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 4000
    timeout: int = 60


@dataclass
class RenderingConfig:
    """Default rendering configuration."""
    device: DeviceType = DeviceType.CUDA
    backend: RenderingBackend = RenderingBackend.PYTORCH
    max_memory_gb: float = 8.0
    mixed_precision: bool = True
    compile_models: bool = True


@dataclass
class SimulationConfig:
    """Default simulation configuration."""
    max_fps: int = 60
    physics_enabled: bool = True
    level_of_detail: bool = True
    max_objects: int = 1000


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    file_path: Optional[str] = None
    max_file_size: str = "10 MB"
    retention: str = "1 week"


@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str = "your-secret-key-change-this"
    api_key_header: str = "X-API-Key"
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    enable_https: bool = False
    cert_file: Optional[str] = None
    key_file: Optional[str] = None


class Config:
    """
    Main configuration class for Open3D platform.
    
    This class manages all configuration settings and provides methods
    to load from various sources with proper validation.
    """

    def __init__(
        self,
        api: Optional[APIConfig] = None,
        database: Optional[DatabaseConfig] = None,
        redis: Optional[RedisConfig] = None,
        openai: Optional[OpenAIConfig] = None,
        rendering: Optional[RenderingConfig] = None,
        simulation: Optional[SimulationConfig] = None,
        logging: Optional[LoggingConfig] = None,
        security: Optional[SecurityConfig] = None,
        **kwargs
    ):
        """
        Initialize configuration.
        
        Args:
            api: API server configuration
            database: Database configuration  
            redis: Redis configuration
            openai: OpenAI API configuration
            rendering: Rendering configuration
            simulation: Simulation configuration
            logging: Logging configuration
            security: Security configuration
            **kwargs: Additional configuration options
        """
        self.api = api or APIConfig()
        self.database = database or DatabaseConfig()
        self.redis = redis or RedisConfig()
        self.openai = openai or OpenAIConfig()
        self.rendering = rendering or RenderingConfig()
        self.simulation = simulation or SimulationConfig()
        self.logging = logging or LoggingConfig()
        self.security = security or SecurityConfig()
        
        # Additional settings
        self.environment = kwargs.get("environment", "development")
        self.debug = kwargs.get("debug", self.environment == "development")
        self.testing = kwargs.get("testing", False)
        
        # Data directories
        self.data_dir = Path(kwargs.get("data_dir", "data"))
        self.models_dir = Path(kwargs.get("models_dir", "models"))
        self.output_dir = Path(kwargs.get("output_dir", "output"))
        self.logs_dir = Path(kwargs.get("logs_dir", "logs"))
        
        # Ensure directories exist
        self._create_directories()

    def _create_directories(self) -> None:
        """Create required directories if they don't exist."""
        for directory in [self.data_dir, self.models_dir, self.output_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load_from_file(cls, config_path: Union[str, Path]) -> "Config":
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            
        Returns:
            Loaded configuration
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return cls.load_default()
        
        logger.info(f"Loading configuration from {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """
        Create configuration from dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            Configuration object
        """
        # Extract subsection configurations
        api_config = APIConfig(**data.get("api", {}))
        database_config = DatabaseConfig(**data.get("database", {}))
        redis_config = RedisConfig(**data.get("redis", {}))
        openai_config = OpenAIConfig(**data.get("openai", {}))
        rendering_config = RenderingConfig(**data.get("rendering", {}))
        simulation_config = SimulationConfig(**data.get("simulation", {}))
        logging_config = LoggingConfig(**data.get("logging", {}))
        security_config = SecurityConfig(**data.get("security", {}))
        
        # Remove known sections from data
        remaining_data = {
            k: v for k, v in data.items() 
            if k not in ["api", "database", "redis", "openai", "rendering", 
                        "simulation", "logging", "security"]
        }
        
        return cls(
            api=api_config,
            database=database_config,
            redis=redis_config,
            openai=openai_config,
            rendering=rendering_config,
            simulation=simulation_config,
            logging=logging_config,
            security=security_config,
            **remaining_data
        )

    @classmethod
    def load_from_env(cls) -> "Config":
        """
        Load configuration from environment variables.
        
        Returns:
            Configuration loaded from environment
        """
        logger.info("Loading configuration from environment variables")
        
        # API configuration
        api_config = APIConfig(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            cors_origins=os.getenv("CORS_ORIGINS", "*").split(","),
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "100")),
        )
        
        # Database configuration
        database_config = DatabaseConfig(
            url=os.getenv("DATABASE_URL", "sqlite:///open3d.db"),
            echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
        )
        
        # Redis configuration
        redis_config = RedisConfig(
            url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        )
        
        # OpenAI configuration
        openai_config = OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
        )
        
        # Rendering configuration
        rendering_config = RenderingConfig(
            device=DeviceType(os.getenv("RENDER_DEVICE", "cuda")),
            backend=RenderingBackend(os.getenv("RENDER_BACKEND", "pytorch")),
        )
        
        # Other settings
        environment = os.getenv("ENVIRONMENT", "development")
        debug = os.getenv("DEBUG", "false").lower() == "true"
        
        return cls(
            api=api_config,
            database=database_config,
            redis=redis_config,
            openai=openai_config,
            rendering=rendering_config,
            environment=environment,
            debug=debug,
        )

    @classmethod
    def load_default(cls) -> "Config":
        """
        Load default configuration.
        
        Returns:
            Default configuration
        """
        logger.info("Loading default configuration")
        
        # Try to load from file first, then environment, then pure defaults
        config_files = [
            "config.yaml", "config.yml", "config.json",
            "open3d.yaml", "open3d.yml", "open3d.json"
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                config = cls.load_from_file(config_file)
                # Override with environment variables
                config.update_from_env()
                return config
        
        # No config file found, load from environment
        return cls.load_from_env()

    def update_from_env(self) -> None:
        """Update configuration with environment variables."""
        # API settings
        if os.getenv("API_HOST"):
            self.api.host = os.getenv("API_HOST")
        if os.getenv("API_PORT"):
            self.api.port = int(os.getenv("API_PORT"))
        
        # Database settings
        if os.getenv("DATABASE_URL"):
            self.database.url = os.getenv("DATABASE_URL")
        
        # Redis settings
        if os.getenv("REDIS_URL"):
            self.redis.url = os.getenv("REDIS_URL")
        
        # OpenAI settings
        if os.getenv("OPENAI_API_KEY"):
            self.openai.api_key = os.getenv("OPENAI_API_KEY")
        if os.getenv("OPENAI_MODEL"):
            self.openai.model = os.getenv("OPENAI_MODEL")
        
        # General settings
        if os.getenv("ENVIRONMENT"):
            self.environment = os.getenv("ENVIRONMENT")
        if os.getenv("DEBUG"):
            self.debug = os.getenv("DEBUG").lower() == "true"

    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration file
        """
        config_path = Path(config_path)
        
        data = self.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(data, f, default_flow_style=False)
            elif config_path.suffix.lower() == '.json':
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        logger.info(f"Configuration saved to {config_path}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return {
            "api": self.api.__dict__,
            "database": self.database.__dict__,
            "redis": self.redis.__dict__,
            "openai": self.openai.__dict__,
            "rendering": self.rendering.__dict__,
            "simulation": self.simulation.__dict__,
            "logging": self.logging.__dict__,
            "security": self.security.__dict__,
            "environment": self.environment,
            "debug": self.debug,
            "testing": self.testing,
            "data_dir": str(self.data_dir),
            "models_dir": str(self.models_dir),
            "output_dir": str(self.output_dir),
            "logs_dir": str(self.logs_dir),
        }

    def validate(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            True if configuration is valid
        """
        errors = []
        
        # Validate API configuration
        if self.api.port < 1 or self.api.port > 65535:
            errors.append(f"Invalid API port: {self.api.port}")
        
        # Validate OpenAI configuration
        if not self.openai.api_key and self.environment == "production":
            errors.append("OpenAI API key is required in production")
        
        # Validate rendering configuration
        if self.rendering.device == DeviceType.CUDA:
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA device specified but not available, falling back to CPU")
                    self.rendering.device = DeviceType.CPU
            except ImportError:
                errors.append("PyTorch not available for CUDA rendering")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            return False
        
        return True

    def __repr__(self) -> str:
        return f"Config(environment={self.environment}, debug={self.debug})"


class ConfigManager:
    """
    Configuration manager with caching and reloading capabilities.
    """

    def __init__(self):
        self._config: Optional[Config] = None
        self._config_path: Optional[Path] = None

    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> Config:
        """
        Load configuration with caching.
        
        Args:
            config_path: Optional path to configuration file
            
        Returns:
            Loaded configuration
        """
        if config_path:
            self._config_path = Path(config_path)
            self._config = Config.load_from_file(self._config_path)
        else:
            self._config = Config.load_default()
        
        # Validate configuration
        if not self._config.validate():
            logger.warning("Configuration validation failed, but continuing with current settings")
        
        return self._config

    def get_config(self) -> Config:
        """
        Get current configuration, loading default if not loaded.
        
        Returns:
            Current configuration
        """
        if self._config is None:
            self._config = Config.load_default()
        return self._config

    def reload_config(self) -> Config:
        """
        Reload configuration from source.
        
        Returns:
            Reloaded configuration
        """
        if self._config_path:
            self._config = Config.load_from_file(self._config_path)
        else:
            self._config = Config.load_default()
        
        logger.info("Configuration reloaded")
        return self._config

    def update_config(self, **kwargs) -> None:
        """
        Update configuration with new values.
        
        Args:
            **kwargs: Configuration updates
        """
        if self._config is None:
            self._config = Config.load_default()
        
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")


# Global configuration manager instance
config_manager = ConfigManager() 