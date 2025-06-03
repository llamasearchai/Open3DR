"""
Core module for Open3D reconstruction platform.

This module provides fundamental data structures, configuration management,
and utility functions used throughout the Open3D platform.
"""

from .engine import Engine
from .config import Config, ConfigManager
from .types import (
    Vector3,
    Matrix4x4,
    Camera,
    Transform,
    BoundingBox,
    Ray,
    RenderConfig,
    SensorConfig,
    SimulationConfig,
)
from .utils import (
    Timer,
    Logger,
    MemoryMonitor,
    GPUMonitor,
    FileManager,
    DataLoader,
    Validator,
)

__all__ = [
    # Engine
    "Engine",
    # Configuration
    "Config",
    "ConfigManager",
    # Types
    "Vector3",
    "Matrix4x4", 
    "Camera",
    "Transform",
    "BoundingBox",
    "Ray",
    "RenderConfig",
    "SensorConfig",
    "SimulationConfig",
    # Utils
    "Timer",
    "Logger",
    "MemoryMonitor",
    "GPUMonitor",
    "FileManager",
    "DataLoader",
    "Validator",
] 