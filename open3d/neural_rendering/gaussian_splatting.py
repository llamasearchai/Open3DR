"""
3D Gaussian Splatting implementation for high-quality real-time rendering.

This module implements the 3D Gaussian Splatting technique that represents
scenes as collections of 3D Gaussians and renders them efficiently.
"""

from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger

from .base import BaseNeuralRenderer, RenderingResult
from ..core.types import RenderConfig, Camera
from ..data import ImageDataset


class GaussianModel(nn.Module):
    """
    3D Gaussian model for Gaussian Splatting.
    
    This model represents the scene as a collection of 3D Gaussians,
    each with position, rotation, scale, opacity, and color.
    """

    def __init__(self, num_gaussians: int = 100000):
        super().__init__()
        
        self.num_gaussians = num_gaussians
        
        # Gaussian parameters (learnable)
        self.positions = nn.Parameter(torch.randn(num_gaussians, 3))
        self.rotations = nn.Parameter(torch.randn(num_gaussians, 4))  # Quaternions
        self.scales = nn.Parameter(torch.randn(num_gaussians, 3))
        self.opacities = nn.Parameter(torch.randn(num_gaussians, 1))
        self.features = nn.Parameter(torch.randn(num_gaussians, 3))  # RGB colors
        
        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize Gaussian parameters."""
        # Initialize positions randomly in unit cube
        with torch.no_grad():
            self.positions.uniform_(-1, 1)
            
            # Initialize rotations as identity quaternions
            self.rotations.zero_()
            self.rotations[:, 0] = 1.0  # w component
            
            # Initialize scales to small values
            self.scales.fill_(-2.0)  # Will be exp() to get actual scales
            
            # Initialize opacities to small values
            self.opacities.fill_(-6.0)  # Will be sigmoid() to get actual opacities
            
            # Initialize colors to neutral
            self.features.fill_(0.5)

    def get_gaussians(self) -> Dict[str, torch.Tensor]:
        """Get processed Gaussian parameters."""
        return {
            'positions': self.positions,
            'rotations': F.normalize(self.rotations, dim=-1),  # Normalize quaternions
            'scales': torch.exp(self.scales),  # Ensure positive scales
            'opacities': torch.sigmoid(self.opacities),  # Ensure [0, 1] opacities
            'colors': torch.sigmoid(self.features)  # Ensure [0, 1] colors
        }

    def prune_gaussians(self, mask: torch.Tensor) -> None:
        """
        Prune Gaussians based on a boolean mask.
        
        Args:
            mask: Boolean tensor indicating which Gaussians to keep
        """
        with torch.no_grad():
            self.positions.data = self.positions.data[mask]
            self.rotations.data = self.rotations.data[mask]
            self.scales.data = self.scales.data[mask]
            self.opacities.data = self.opacities.data[mask]
            self.features.data = self.features.data[mask]
            
            self.num_gaussians = mask.sum().item()

    def densify_gaussians(self, positions: torch.Tensor, **kwargs) -> None:
        """
        Add new Gaussians at specified positions.
        
        Args:
            positions: New Gaussian positions [N, 3]
            **kwargs: Additional parameters for new Gaussians
        """
        num_new = positions.shape[0]
        
        with torch.no_grad():
            # Expand existing parameters
            new_positions = torch.cat([self.positions.data, positions], dim=0)
            
            new_rotations = torch.cat([
                self.rotations.data,
                torch.tensor([[1, 0, 0, 0]] * num_new, 
                           device=self.rotations.device, 
                           dtype=self.rotations.dtype)
            ], dim=0)
            
            new_scales = torch.cat([
                self.scales.data,
                torch.full((num_new, 3), -2.0, 
                          device=self.scales.device, 
                          dtype=self.scales.dtype)
            ], dim=0)
            
            new_opacities = torch.cat([
                self.opacities.data,
                torch.full((num_new, 1), -6.0, 
                          device=self.opacities.device, 
                          dtype=self.opacities.dtype)
            ], dim=0)
            
            new_features = torch.cat([
                self.features.data,
                torch.full((num_new, 3), 0.5, 
                          device=self.features.device, 
                          dtype=self.features.dtype)
            ], dim=0)
            
            # Update parameters
            self.positions = nn.Parameter(new_positions)
            self.rotations = nn.Parameter(new_rotations)
            self.scales = nn.Parameter(new_scales)
            self.opacities = nn.Parameter(new_opacities)
            self.features = nn.Parameter(new_features)
            
            self.num_gaussians += num_new


def quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation matrices.
    
    Args:
        quaternions: Quaternion tensor [N, 4] (w, x, y, z)
        
    Returns:
        Rotation matrices [N, 3, 3]
    """
    w, x, y, z = quaternions.unbind(-1)
    
    # Compute rotation matrix elements
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    
    # Assemble rotation matrix
    R = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)
    ], dim=-1).reshape(-1, 3, 3)
    
    return R


def build_covariance_3d(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """
    Build 3D covariance matrices from scales and rotations.
    
    Args:
        scales: Scale factors [N, 3]
        rotations: Rotation matrices [N, 3, 3]
        
    Returns:
        Covariance matrices [N, 3, 3]
    """
    # Create scale matrix
    S = torch.diag_embed(scales)  # [N, 3, 3]
    
    # Compute covariance: R * S * S^T * R^T
    RS = torch.bmm(rotations, S)  # [N, 3, 3]
    covariance = torch.bmm(RS, RS.transpose(-1, -2))  # [N, 3, 3]
    
    return covariance


def project_gaussians_2d(
    positions: torch.Tensor,
    covariances: torch.Tensor,
    camera: Camera,
    image_size: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project 3D Gaussians to 2D screen space.
    
    Args:
        positions: 3D positions [N, 3]
        covariances: 3D covariance matrices [N, 3, 3]
        camera: Camera parameters
        image_size: (height, width)
        
    Returns:
        Tuple of (2D positions, 2D covariances, depths)
    """
    # Simple perspective projection (simplified)
    fx, fy = camera.fx, camera.fy
    cx, cy = camera.cx, camera.cy
    
    # Project positions
    z = positions[:, 2]
    x_2d = (positions[:, 0] * fx / z) + cx
    y_2d = (positions[:, 1] * fy / z) + cy
    positions_2d = torch.stack([x_2d, y_2d], dim=-1)
    
    # Project covariances (simplified Jacobian approach)
    # In practice, this would use the full perspective projection Jacobian
    jacobian = torch.tensor([[fx / z.mean(), 0], [0, fy / z.mean()]], 
                           device=positions.device).expand(len(positions), 2, 2)
    
    # Transform 3D covariance to 2D (simplified)
    covariance_2d = torch.bmm(torch.bmm(jacobian, covariances[:, :2, :2]), 
                             jacobian.transpose(-1, -2))
    
    return positions_2d, covariance_2d, z


def render_gaussians(
    positions_2d: torch.Tensor,
    covariances_2d: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    depths: torch.Tensor,
    image_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Render 2D Gaussians to an image.
    
    Args:
        positions_2d: 2D positions [N, 2]
        covariances_2d: 2D covariance matrices [N, 2, 2]
        colors: RGB colors [N, 3]
        opacities: Opacity values [N, 1]
        depths: Depth values [N]
        image_size: (height, width)
        
    Returns:
        Rendered image [height, width, 3]
    """
    height, width = image_size
    device = positions_2d.device
    
    # Create pixel grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    pixel_coords = torch.stack([x_coords, y_coords], dim=-1).float()  # [H, W, 2]
    
    # Initialize output image
    output_image = torch.zeros(height, width, 3, device=device)
    total_alpha = torch.zeros(height, width, device=device)
    
    # Sort Gaussians by depth (back to front)
    depth_order = torch.argsort(depths, descending=True)
    
    # Render each Gaussian
    for idx in depth_order:
        pos = positions_2d[idx]  # [2]
        cov = covariances_2d[idx]  # [2, 2]
        color = colors[idx]  # [3]
        opacity = opacities[idx]  # [1]
        
        # Compute squared Mahalanobis distance
        diff = pixel_coords - pos.view(1, 1, 2)  # [H, W, 2]
        
        # Compute covariance inverse (with regularization)
        cov_reg = cov + torch.eye(2, device=device) * 1e-6
        try:
            cov_inv = torch.inverse(cov_reg)
        except:
            continue  # Skip degenerate Gaussians
        
        # Compute Gaussian weights
        quad_form = torch.sum(diff * torch.matmul(diff, cov_inv), dim=-1)  # [H, W]
        weights = torch.exp(-0.5 * quad_form)  # [H, W]
        
        # Apply opacity
        alpha = weights * opacity.item()
        
        # Alpha blending
        output_image += alpha.unsqueeze(-1) * (1 - total_alpha).unsqueeze(-1) * color.view(1, 1, 3)
        total_alpha += alpha * (1 - total_alpha)
        
        # Early termination if fully opaque
        if torch.all(total_alpha > 0.99):
            break
    
    return output_image


class GaussianSplattingRenderer(BaseNeuralRenderer):
    """
    3D Gaussian Splatting renderer implementation.
    
    This renderer uses 3D Gaussians to represent the scene and provides
    high-quality real-time rendering capabilities.
    """

    def __init__(self, config: RenderConfig):
        super().__init__(config)
        self.dataset: Optional[ImageDataset] = None
        self.initial_num_gaussians = 100000

    def build_model(self) -> nn.Module:
        """Build Gaussian Splatting model."""
        return GaussianModel(num_gaussians=self.initial_num_gaussians)

    async def load_data(self, data_spec: Dict[str, Any]) -> None:
        """Load image dataset for training."""
        if data_spec["type"] == "images":
            self.dataset = ImageDataset(data_spec["path"])
            await self.dataset.load_async()
            
            self.training_data = {
                "images": self.dataset.images,
                "poses": self.dataset.poses,
                "intrinsics": self.dataset.intrinsics,
                "bounds": self.dataset.bounds
            }
            
            # Initialize Gaussians from point cloud if available
            if hasattr(self.dataset, 'point_cloud') and self.dataset.point_cloud is not None:
                self._initialize_from_point_cloud()
            
            logger.info(f"Loaded {len(self.dataset)} images for Gaussian Splatting training")
        else:
            raise ValueError(f"Unsupported data type: {data_spec['type']}")

    def _initialize_from_point_cloud(self) -> None:
        """Initialize Gaussians from point cloud."""
        if self.model is None:
            return
        
        points = torch.from_numpy(self.dataset.point_cloud[:, :3]).float()
        colors = torch.from_numpy(self.dataset.point_cloud[:, 3:6]).float() / 255.0
        
        # Sample points if we have too many
        if len(points) > self.initial_num_gaussians:
            indices = torch.randperm(len(points))[:self.initial_num_gaussians]
            points = points[indices]
            colors = colors[indices]
        
        # Initialize model parameters
        with torch.no_grad():
            self.model.positions.data[:len(points)] = points
            self.model.features.data[:len(points)] = colors
            
        logger.info(f"Initialized {len(points)} Gaussians from point cloud")

    def forward(self, rays: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Render view using Gaussian Splatting.
        
        Args:
            rays: Ray tensor [N, 8] - for GS, we use camera info instead
            
        Returns:
            Rendering outputs
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        # Extract camera info from kwargs or use default
        camera = kwargs.get('camera')
        image_size = kwargs.get('image_size', (self.config.resolution, self.config.resolution))
        
        if camera is None:
            # Create default camera from ray info (simplified)
            camera = Camera(
                width=image_size[1],
                height=image_size[0],
                fx=image_size[1] / 2,
                fy=image_size[0] / 2,
                cx=image_size[1] / 2,
                cy=image_size[0] / 2
            )
        
        # Get Gaussian parameters
        gaussians = self.model.get_gaussians()
        
        # Build 3D covariance matrices
        rotation_matrices = quaternion_to_rotation_matrix(gaussians['rotations'])
        covariances_3d = build_covariance_3d(gaussians['scales'], rotation_matrices)
        
        # Project to 2D
        positions_2d, covariances_2d, depths = project_gaussians_2d(
            gaussians['positions'],
            covariances_3d,
            camera,
            image_size
        )
        
        # Render
        rendered_image = render_gaussians(
            positions_2d,
            covariances_2d,
            gaussians['colors'],
            gaussians['opacities'],
            depths,
            image_size
        )
        
        # For compatibility with ray-based interface, reshape to batch format
        if rays.dim() == 2:
            # Flatten image for per-pixel output
            batch_size = rays.shape[0]
            height, width = image_size
            
            # Sample pixels corresponding to the rays
            # This is simplified - in practice would use actual ray-pixel correspondence
            pixel_indices = torch.randint(0, height * width, (batch_size,), device=rays.device)
            flat_image = rendered_image.reshape(-1, 3)
            sampled_colors = flat_image[pixel_indices]
            
            return {
                'rgb': sampled_colors,
                'depth': depths.mean().expand(batch_size),
                'full_image': rendered_image
            }
        
        return {
            'rgb': rendered_image,
            'depth': depths,
            'full_image': rendered_image
        }

    async def render_view(
        self,
        camera: Camera,
        resolution: Optional[Tuple[int, int]] = None
    ) -> RenderingResult:
        """
        Render a view from a specific camera.
        
        Args:
            camera: Camera specification
            resolution: Optional resolution override
            
        Returns:
            Rendered result
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        self.model.eval()
        
        # Use config resolution if not specified
        if resolution is None:
            resolution = (self.config.resolution, self.config.resolution)
        
        with self.timer.measure("gaussian_splatting_render"):
            with torch.no_grad():
                # Create dummy rays for interface compatibility
                dummy_rays = torch.zeros(1, 8, device=self.device)
                
                # Render using forward pass
                outputs = self.forward(
                    dummy_rays,
                    camera=camera,
                    image_size=resolution
                )
                
                rendered_image = outputs['full_image'].cpu().numpy()
        
        return RenderingResult(
            image=rendered_image,
            depth=None,  # Depth is per-Gaussian, not per-pixel in standard GS
            metadata={
                "camera": camera,
                "resolution": resolution,
                "render_time": self.timer.get_last_duration(),
                "num_gaussians": self.model.num_gaussians
            }
        )

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute Gaussian Splatting loss."""
        # Main photometric loss
        rgb_loss = F.l1_loss(outputs['rgb'], targets['rgb'])  # L1 loss is common for GS
        
        total_loss = rgb_loss
        
        # Optional SSIM loss for better perceptual quality
        if 'full_image' in outputs and 'full_image' in targets:
            # Simplified SSIM loss placeholder
            ssim_loss = F.mse_loss(outputs['full_image'], targets['full_image'])
            total_loss += 0.2 * ssim_loss
        
        return total_loss

    def _sample_batch(self) -> Dict[str, torch.Tensor]:
        """Sample training batch from dataset."""
        if self.dataset is None:
            return super()._sample_batch()
        
        batch_size = self.config.batch_size
        
        # Sample random image
        img_idx = torch.randint(0, len(self.dataset), (1,)).item()
        image = torch.from_numpy(self.dataset.images[img_idx]).to(self.device)
        pose = torch.from_numpy(self.dataset.poses[img_idx]).to(self.device)
        intrinsics = self.dataset.intrinsics[img_idx]
        
        # For Gaussian Splatting, we often train on full images rather than rays
        # Here we sample pixels for compatibility with the ray interface
        h, w = image.shape[:2]
        
        # Sample random pixels
        pixel_indices = torch.randint(0, h * w, (batch_size,), device=self.device)
        selected_rgb = image.reshape(-1, 3)[pixel_indices]
        
        # Create dummy rays for interface compatibility
        dummy_rays = torch.randn(batch_size, 8, device=self.device)
        
        return {
            "rays": dummy_rays,
            "targets": {
                "rgb": selected_rgb,
                "full_image": image
            }
        }

    def densification_step(self) -> None:
        """
        Perform densification step to add/remove Gaussians.
        
        This is a key component of Gaussian Splatting optimization.
        """
        if self.model is None:
            return
        
        gaussians = self.model.get_gaussians()
        
        # Compute gradient magnitudes (simplified)
        if hasattr(self.model.positions, 'grad') and self.model.positions.grad is not None:
            grad_magnitude = torch.norm(self.model.positions.grad, dim=-1)
            
            # Densify high-gradient regions
            densify_mask = grad_magnitude > 0.01  # Threshold
            densify_positions = gaussians['positions'][densify_mask]
            
            if len(densify_positions) > 0:
                # Add noise to create new Gaussians
                noise = torch.randn_like(densify_positions) * 0.01
                new_positions = densify_positions + noise
                self.model.densify_gaussians(new_positions)
                
                logger.debug(f"Densified {len(new_positions)} Gaussians")
        
        # Prune low-opacity Gaussians
        opacity_mask = gaussians['opacities'].squeeze(-1) > 0.01
        if opacity_mask.sum() < len(opacity_mask):
            self.model.prune_gaussians(opacity_mask)
            logger.debug(f"Pruned {(~opacity_mask).sum()} low-opacity Gaussians")

    async def _training_step(self) -> Tuple[float, Dict[str, float]]:
        """Perform a single training step with densification."""
        # Standard training step
        loss, metrics = await super()._training_step()
        
        # Perform densification every N iterations
        if self.current_iteration % 100 == 0:
            self.densification_step()
            metrics['num_gaussians'] = self.model.num_gaussians
        
        return loss, metrics 