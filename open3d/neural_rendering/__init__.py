"""
Neural Rendering module for Open3D reconstruction platform.

This module provides state-of-the-art neural rendering techniques including
NeRF variants, Gaussian Splatting, and other neural field representations.
"""

from .nerf import NeRFReconstructor, InstantNGP, MipNeRF, TensorF, Nerfacto
from .gaussian_splatting import GaussianSplattingRenderer
from .neural_fields import SDFNetwork, DensityField, FeatureNetwork
from .base import BaseNeuralRenderer, RenderingResult
from .utils import CameraPath, ViewSynthesis, MeshExtractor

__all__ = [
    # Base classes
    "BaseNeuralRenderer",
    "RenderingResult",
    # NeRF implementations
    "NeRFReconstructor",
    "InstantNGP",
    "MipNeRF", 
    "TensorF",
    "Nerfacto",
    # Gaussian Splatting
    "GaussianSplattingRenderer",
    # Neural Fields
    "SDFNetwork",
    "DensityField",
    "FeatureNetwork",
    # Utilities
    "CameraPath",
    "ViewSynthesis",
    "MeshExtractor",
] 