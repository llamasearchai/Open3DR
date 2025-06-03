"""
Data transformation utilities for preprocessing and augmentation.

This module provides various transformation functions for images,
point clouds, and geometric data used in 3D reconstruction.
"""

from typing import Union, Tuple, Optional, List, Any, Dict
import numpy as np
import cv2
import torch
from dataclasses import dataclass

@dataclass
class TransformConfig:
    """Configuration for data transformations."""
    resize_resolution: Optional[Tuple[int, int]] = None
    crop_size: Optional[Tuple[int, int]] = None
    normalize: bool = True
    augment: bool = False
    noise_std: float = 0.01


class ImageTransform:
    """Image transformation utilities."""
    
    def __init__(self, config: Optional[TransformConfig] = None):
        self.config = config or TransformConfig()
    
    def resize(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image to specified size."""
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        return image.astype(np.float32) / 255.0
    
    def crop_center(self, image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
        """Crop image from center."""
        h, w = image.shape[:2]
        crop_h, crop_w = crop_size
        
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        return image[start_h:start_h + crop_h, start_w:start_w + crop_w]
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply configured transformations."""
        result = image.copy()
        
        if self.config.resize_resolution:
            result = self.resize(result, self.config.resize_resolution)
        
        if self.config.crop_size:
            result = self.crop_center(result, self.config.crop_size)
        
        if self.config.normalize:
            result = self.normalize(result)
        
        return result


class PointCloudTransform:
    """Point cloud transformation utilities."""
    
    def __init__(self, config: Optional[TransformConfig] = None):
        self.config = config or TransformConfig()
    
    def normalize(self, points: np.ndarray) -> np.ndarray:
        """Normalize points to unit sphere."""
        center = np.mean(points, axis=0)
        points_centered = points - center
        scale = np.max(np.linalg.norm(points_centered, axis=1))
        return points_centered / scale
    
    def add_noise(self, points: np.ndarray, std: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to points."""
        noise = np.random.normal(0, std, points.shape)
        return points + noise
    
    def subsample(self, points: np.ndarray, num_points: int) -> np.ndarray:
        """Randomly subsample points."""
        if len(points) <= num_points:
            return points
        
        indices = np.random.choice(len(points), num_points, replace=False)
        return points[indices]
    
    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply configured transformations."""
        result = points.copy()
        
        if self.config.augment and self.config.noise_std > 0:
            result = self.add_noise(result, self.config.noise_std)
        
        return result


class GeometryTransform:
    """Geometric transformation utilities."""
    
    def __init__(self):
        pass
    
    def apply_transform(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Apply 4x4 transformation matrix to points."""
        if points.shape[1] == 3:
            # Add homogeneous coordinate
            points_homo = np.hstack([points, np.ones((len(points), 1))])
        else:
            points_homo = points
        
        # Apply transformation
        transformed = (transform @ points_homo.T).T
        
        # Return 3D points
        return transformed[:, :3]
    
    def rotate_x(self, angle: float) -> np.ndarray:
        """Create rotation matrix around X axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ])
    
    def rotate_y(self, angle: float) -> np.ndarray:
        """Create rotation matrix around Y axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ])
    
    def rotate_z(self, angle: float) -> np.ndarray:
        """Create rotation matrix around Z axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    def translate(self, translation: np.ndarray) -> np.ndarray:
        """Create translation matrix."""
        t = np.eye(4)
        t[:3, 3] = translation
        return t
    
    def scale(self, scale_factor: Union[float, np.ndarray]) -> np.ndarray:
        """Create scale matrix."""
        if isinstance(scale_factor, float):
            scale_factor = np.array([scale_factor, scale_factor, scale_factor])
        
        s = np.eye(4)
        s[0, 0] = scale_factor[0]
        s[1, 1] = scale_factor[1]
        s[2, 2] = scale_factor[2]
        return s 