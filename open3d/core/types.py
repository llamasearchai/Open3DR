"""
Core data types and structures for Open3D reconstruction platform.

This module defines the fundamental data types used throughout the platform
including geometric primitives, camera models, and configuration structures.
"""

from __future__ import annotations
from typing import List, Optional, Union, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field, validator


class DeviceType(str, Enum):
    """Supported device types for computation."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal Performance Shaders


class RenderingBackend(str, Enum):
    """Supported rendering backends."""
    PYTORCH = "pytorch"
    TENSORRT = "tensorrt"
    ONNX = "onnx"
    TRITON = "triton"


class SensorType(str, Enum):
    """Types of sensors supported."""
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    IMU = "imu"
    GPS = "gps"


@dataclass
class Vector3:
    """3D vector representation."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __array__(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __mul__(self, scalar: float) -> Vector3:
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def normalize(self) -> Vector3:
        """Normalize the vector to unit length."""
        magnitude = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        if magnitude > 0:
            return Vector3(self.x / magnitude, self.y / magnitude, self.z / magnitude)
        return Vector3()
    
    def dot(self, other: Vector3) -> float:
        """Compute dot product with another vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z


@dataclass
class Matrix4x4:
    """4x4 transformation matrix."""
    data: np.ndarray = field(default_factory=lambda: np.eye(4))
    
    def __post_init__(self):
        if self.data.shape != (4, 4):
            raise ValueError("Matrix must be 4x4")
    
    @classmethod
    def identity(cls) -> Matrix4x4:
        """Create identity matrix."""
        return cls()
    
    @classmethod
    def translation(cls, translation: Vector3) -> Matrix4x4:
        """Create translation matrix."""
        matrix = np.eye(4)
        matrix[0:3, 3] = [translation.x, translation.y, translation.z]
        return cls(matrix)
    
    @classmethod
    def rotation_x(cls, angle: float) -> Matrix4x4:
        """Create rotation matrix around X axis."""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        matrix = np.eye(4)
        matrix[1:3, 1:3] = [[cos_a, -sin_a], [sin_a, cos_a]]
        return cls(matrix)
    
    def __matmul__(self, other: Matrix4x4) -> Matrix4x4:
        """Matrix multiplication."""
        return Matrix4x4(self.data @ other.data)


class Camera(BaseModel):
    """Camera model with intrinsic and extrinsic parameters."""
    
    # Intrinsic parameters
    width: int = Field(..., gt=0, description="Image width in pixels")
    height: int = Field(..., gt=0, description="Image height in pixels")
    fx: float = Field(..., gt=0, description="Focal length in x direction")
    fy: float = Field(..., gt=0, description="Focal length in y direction")
    cx: float = Field(..., description="Principal point x coordinate")
    cy: float = Field(..., description="Principal point y coordinate")
    
    # Distortion parameters (Brown-Conrady model)
    k1: float = Field(0.0, description="Radial distortion coefficient k1")
    k2: float = Field(0.0, description="Radial distortion coefficient k2")
    k3: float = Field(0.0, description="Radial distortion coefficient k3")
    p1: float = Field(0.0, description="Tangential distortion coefficient p1")
    p2: float = Field(0.0, description="Tangential distortion coefficient p2")
    
    # Extrinsic parameters
    position: Vector3 = Field(default_factory=Vector3)
    rotation: Vector3 = Field(default_factory=Vector3)  # Euler angles
    
    @validator('cx')
    def validate_cx(cls, v, values):
        if 'width' in values and v >= values['width']:
            raise ValueError('cx must be less than width')
        return v
    
    @validator('cy')
    def validate_cy(cls, v, values):
        if 'height' in values and v >= values['height']:
            raise ValueError('cy must be less than height')
        return v
    
    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """Get 3x3 intrinsic camera matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
    
    @property
    def distortion_coefficients(self) -> np.ndarray:
        """Get distortion coefficients array."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])


@dataclass
class Transform:
    """3D transformation containing translation and rotation."""
    translation: Vector3 = field(default_factory=Vector3)
    rotation: Vector3 = field(default_factory=Vector3)  # Euler angles in radians
    scale: Vector3 = field(default_factory=lambda: Vector3(1.0, 1.0, 1.0))
    
    def to_matrix(self) -> Matrix4x4:
        """Convert to 4x4 transformation matrix."""
        # Create individual transformation matrices
        t_matrix = Matrix4x4.translation(self.translation)
        rx_matrix = Matrix4x4.rotation_x(self.rotation.x)
        ry_matrix = Matrix4x4.rotation_x(self.rotation.y)  # Would implement rotation_y
        rz_matrix = Matrix4x4.rotation_x(self.rotation.z)  # Would implement rotation_z
        
        # Combine transformations: T * Rz * Ry * Rx * S
        return t_matrix @ rz_matrix @ ry_matrix @ rx_matrix
    
    def inverse(self) -> Transform:
        """Get inverse transformation."""
        # Simplified inverse (would need proper implementation)
        return Transform(
            translation=Vector3(-self.translation.x, -self.translation.y, -self.translation.z),
            rotation=Vector3(-self.rotation.x, -self.rotation.y, -self.rotation.z),
            scale=Vector3(1.0/self.scale.x, 1.0/self.scale.y, 1.0/self.scale.z)
        )


@dataclass
class BoundingBox:
    """3D axis-aligned bounding box."""
    min_point: Vector3 = field(default_factory=Vector3)
    max_point: Vector3 = field(default_factory=Vector3)
    
    @property
    def center(self) -> Vector3:
        """Get center point of bounding box."""
        return Vector3(
            (self.min_point.x + self.max_point.x) / 2,
            (self.min_point.y + self.max_point.y) / 2,
            (self.min_point.z + self.max_point.z) / 2
        )
    
    @property
    def size(self) -> Vector3:
        """Get size of bounding box."""
        return Vector3(
            self.max_point.x - self.min_point.x,
            self.max_point.y - self.min_point.y,
            self.max_point.z - self.min_point.z
        )
    
    def contains(self, point: Vector3) -> bool:
        """Check if point is inside bounding box."""
        return (
            self.min_point.x <= point.x <= self.max_point.x and
            self.min_point.y <= point.y <= self.max_point.y and
            self.min_point.z <= point.z <= self.max_point.z
        )


@dataclass
class Ray:
    """3D ray representation."""
    origin: Vector3 = field(default_factory=Vector3)
    direction: Vector3 = field(default_factory=Vector3)
    t_min: float = 0.0
    t_max: float = float('inf')
    
    def __post_init__(self):
        """Normalize direction vector after initialization."""
        self.direction = self.direction.normalize()
    
    def at(self, t: float) -> Vector3:
        """Get point along ray at parameter t."""
        return self.origin + self.direction * t


class RenderConfig(BaseModel):
    """Configuration for neural rendering."""
    
    # Model parameters
    model_type: str = Field("instant_ngp", description="Type of neural model")
    resolution: int = Field(1024, gt=0, description="Rendering resolution")
    num_iterations: int = Field(10000, gt=0, description="Training iterations")
    batch_size: int = Field(1024, gt=0, description="Batch size for training")
    learning_rate: float = Field(1e-3, gt=0, description="Learning rate")
    
    # Rendering parameters
    near_plane: float = Field(0.1, gt=0, description="Near clipping plane")
    far_plane: float = Field(100.0, gt=0, description="Far clipping plane")
    num_samples: int = Field(64, gt=0, description="Number of samples per ray")
    
    # Hardware configuration
    device: DeviceType = Field(DeviceType.CUDA, description="Compute device")
    backend: RenderingBackend = Field(RenderingBackend.PYTORCH, description="Rendering backend")
    use_mixed_precision: bool = Field(True, description="Use mixed precision training")
    
    # Output settings
    output_format: str = Field("ply", description="Output mesh format")
    export_video: bool = Field(False, description="Export training video")


class SensorConfig(BaseModel):
    """Configuration for sensor simulation."""
    
    sensor_type: SensorType = Field(..., description="Type of sensor")
    frequency: float = Field(10.0, gt=0, description="Sensor frequency in Hz")
    
    # Position and orientation
    position: Vector3 = Field(default_factory=Vector3)
    rotation: Vector3 = Field(default_factory=Vector3)
    
    # Noise parameters
    noise_enabled: bool = Field(True, description="Enable sensor noise")
    noise_std: float = Field(0.01, ge=0, description="Noise standard deviation")
    
    # Sensor-specific parameters (will be extended by subclasses)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class SimulationConfig(BaseModel):
    """Configuration for simulation environment."""
    
    # Environment settings
    scene_path: Optional[str] = Field(None, description="Path to scene file")
    weather_type: str = Field("clear", description="Weather conditions")
    time_of_day: str = Field("noon", description="Time of day")
    
    # Physics settings
    enable_physics: bool = Field(True, description="Enable physics simulation")
    gravity: Vector3 = Field(default_factory=lambda: Vector3(0, 0, -9.81))
    time_step: float = Field(0.01, gt=0, description="Simulation time step")
    
    # Performance settings
    max_fps: int = Field(60, gt=0, description="Maximum simulation FPS")
    level_of_detail: bool = Field(True, description="Enable level-of-detail optimization")
    
    # Output settings
    record_data: bool = Field(False, description="Record simulation data")
    output_directory: str = Field("simulation_output", description="Output directory")


# Type aliases for common types
Position = Vector3
Rotation = Vector3
Scale = Vector3
RGBColor = Tuple[float, float, float]
RGBAColor = Tuple[float, float, float, float]
ImageArray = np.ndarray
PointCloudArray = np.ndarray 