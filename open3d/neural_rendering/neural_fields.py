"""
Neural field representations for 3D reconstruction.

This module provides neural network implementations for representing
various 3D fields including signed distance functions, density fields,
and feature fields.
"""

from typing import Dict, List, Optional, Tuple, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLPLayer(nn.Module):
    """Basic MLP layer with optional skip connections."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_layers: Optional[List[int]] = None,
        activation: str = "relu",
        final_activation: Optional[str] = None
    ):
        super().__init__()
        
        if skip_layers is None:
            skip_layers = []
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip_layers = skip_layers
        
        # Choose activation function
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "elu":
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        
        # Build layers
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                layer_input_dim = input_dim
            elif i in skip_layers:
                layer_input_dim = hidden_dim + input_dim
            else:
                layer_input_dim = hidden_dim
            
            if i == num_layers - 1:
                layer_output_dim = output_dim
            else:
                layer_output_dim = hidden_dim
            
            self.layers.append(nn.Linear(layer_input_dim, layer_output_dim))
        
        # Final activation
        if final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        elif final_activation == "tanh":
            self.final_activation = nn.Tanh()
        else:
            self.final_activation = None
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        input_x = x
        
        for i, layer in enumerate(self.layers):
            if i in self.skip_layers and i > 0:
                x = torch.cat([x, input_x], dim=-1)
            
            x = layer(x)
            
            if i < len(self.layers) - 1:
                x = self.activation(x)
        
        if self.final_activation:
            x = self.final_activation(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for neural fields."""
    
    def __init__(
        self,
        input_dim: int,
        num_freqs: int,
        max_freq_log2: Optional[int] = None,
        include_input: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        self.include_input = include_input
        
        if max_freq_log2 is None:
            max_freq_log2 = num_freqs - 1
        
        # Create frequency bands
        freq_bands = 2. ** torch.linspace(0., max_freq_log2, steps=num_freqs)
        self.register_buffer('freq_bands', freq_bands)
        
        # Calculate output dimension
        self.output_dim = 0
        if include_input:
            self.output_dim += input_dim
        self.output_dim += input_dim * num_freqs * 2  # sin and cos
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding."""
        outputs = []
        
        if self.include_input:
            outputs.append(inputs)
        
        for freq in self.freq_bands:
            for func in [torch.sin, torch.cos]:
                outputs.append(func(inputs * freq))
        
        return torch.cat(outputs, dim=-1)


class SDFNetwork(nn.Module):
    """
    Neural network for Signed Distance Function (SDF) representation.
    
    This network learns to map 3D coordinates to signed distance values,
    commonly used for representing 3D shapes and surfaces.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_layers: Optional[List[int]] = None,
        num_freq_bands: int = 10,
        use_positional_encoding: bool = True,
        geometric_init: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.use_positional_encoding = use_positional_encoding
        
        if skip_layers is None:
            skip_layers = [4]
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(
                input_dim=input_dim,
                num_freqs=num_freq_bands
            )
            mlp_input_dim = self.pos_encoding.output_dim
        else:
            self.pos_encoding = None
            mlp_input_dim = input_dim
        
        # SDF network
        self.sdf_net = MLPLayer(
            input_dim=mlp_input_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            skip_layers=skip_layers,
            activation="relu"
        )
        
        # Geometric initialization
        if geometric_init:
            self._geometric_init()
    
    def _geometric_init(self):
        """Apply geometric initialization for SDF networks."""
        # Initialize the final layer to represent a sphere
        with torch.no_grad():
            final_layer = self.sdf_net.layers[-1]
            final_layer.weight.fill_(0.0)
            final_layer.bias.fill_(0.0)
            
            # Initialize intermediate layers with smaller weights
            for layer in self.sdf_net.layers[:-1]:
                nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(layer.out_features))
                nn.init.zeros_(layer.bias)
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SDF network.
        
        Args:
            points: 3D points [N, 3]
            
        Returns:
            SDF values [N, 1]
        """
        if self.pos_encoding:
            encoded_points = self.pos_encoding(points)
        else:
            encoded_points = points
        
        sdf = self.sdf_net(encoded_points)
        return sdf
    
    def get_sdf_loss(
        self,
        points: torch.Tensor,
        sdf_gt: torch.Tensor,
        truncation: float = 0.1
    ) -> torch.Tensor:
        """Compute SDF loss with truncation."""
        pred_sdf = self.forward(points)
        
        # Truncated L1 loss
        diff = torch.abs(pred_sdf - sdf_gt)
        loss = torch.where(diff < truncation, diff, truncation)
        
        return loss.mean()


class DensityField(nn.Module):
    """
    Neural density field for volume rendering.
    
    This network learns to map 3D coordinates to density values,
    used in neural radiance fields and other volumetric representations.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_layers: Optional[List[int]] = None,
        num_freq_bands: int = 10,
        use_positional_encoding: bool = True,
        density_activation: str = "relu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.use_positional_encoding = use_positional_encoding
        
        if skip_layers is None:
            skip_layers = [4]
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(
                input_dim=input_dim,
                num_freqs=num_freq_bands
            )
            mlp_input_dim = self.pos_encoding.output_dim
        else:
            self.pos_encoding = None
            mlp_input_dim = input_dim
        
        # Density network
        self.density_net = MLPLayer(
            input_dim=mlp_input_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            skip_layers=skip_layers,
            activation="relu"
        )
        
        # Density activation
        if density_activation == "relu":
            self.density_activation = F.relu
        elif density_activation == "softplus":
            self.density_activation = F.softplus
        elif density_activation == "exp":
            self.density_activation = torch.exp
        else:
            self.density_activation = F.relu
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through density field.
        
        Args:
            points: 3D points [N, 3]
            
        Returns:
            Density values [N, 1]
        """
        if self.pos_encoding:
            encoded_points = self.pos_encoding(points)
        else:
            encoded_points = points
        
        density_logits = self.density_net(encoded_points)
        density = self.density_activation(density_logits)
        
        return density
    
    def get_density_regularization(
        self,
        points: torch.Tensor,
        weight: float = 0.01
    ) -> torch.Tensor:
        """Compute density regularization to encourage sparsity."""
        density = self.forward(points)
        return weight * torch.mean(density)


class FeatureNetwork(nn.Module):
    """
    Neural feature field for representing appearance and material properties.
    
    This network learns to map 3D coordinates (and optionally viewing directions)
    to feature vectors that can represent colors, materials, or other properties.
    """
    
    def __init__(
        self,
        position_dim: int = 3,
        direction_dim: int = 3,
        feature_dim: int = 256,
        output_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        pos_freq_bands: int = 10,
        dir_freq_bands: int = 4,
        use_view_dirs: bool = True
    ):
        super().__init__()
        
        self.position_dim = position_dim
        self.direction_dim = direction_dim
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.use_view_dirs = use_view_dirs
        
        # Position encoding
        self.pos_encoding = PositionalEncoding(
            input_dim=position_dim,
            num_freqs=pos_freq_bands
        )
        
        # Direction encoding (if using view directions)
        if use_view_dirs:
            self.dir_encoding = PositionalEncoding(
                input_dim=direction_dim,
                num_freqs=dir_freq_bands
            )
        else:
            self.dir_encoding = None
        
        # Feature extraction network
        self.feature_net = MLPLayer(
            input_dim=self.pos_encoding.output_dim,
            output_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation="relu"
        )
        
        # Output network
        if use_view_dirs:
            output_input_dim = feature_dim + self.dir_encoding.output_dim
        else:
            output_input_dim = feature_dim
        
        self.output_net = MLPLayer(
            input_dim=output_input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim // 2,
            num_layers=2,
            activation="relu",
            final_activation="sigmoid"
        )
    
    def forward(
        self,
        points: torch.Tensor,
        directions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through feature network.
        
        Args:
            points: 3D points [N, 3]
            directions: View directions [N, 3] (optional)
            
        Returns:
            Feature vectors [N, output_dim]
        """
        # Encode positions
        encoded_points = self.pos_encoding(points)
        
        # Extract features
        features = self.feature_net(encoded_points)
        
        # Add view direction if provided
        if self.use_view_dirs and directions is not None:
            encoded_dirs = self.dir_encoding(directions)
            combined_features = torch.cat([features, encoded_dirs], dim=-1)
        else:
            combined_features = features
        
        # Generate output
        output = self.output_net(combined_features)
        
        return output


class HashEncoding(nn.Module):
    """
    Hash-based positional encoding for efficient neural fields.
    
    This implements a simplified version of the hash encoding used in
    Instant Neural Graphics Primitives (Instant-NGP).
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        num_levels: int = 16,
        feature_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 2048
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_levels = num_levels
        self.feature_per_level = feature_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        
        # Calculate resolutions for each level
        self.resolutions = []
        for i in range(num_levels):
            resolution = int(np.floor(base_resolution * np.exp(i * np.log(finest_resolution / base_resolution) / (num_levels - 1))))
            self.resolutions.append(resolution)
        
        # Create hash tables for each level
        hashmap_size = 2 ** log2_hashmap_size
        self.hash_tables = nn.ModuleList([
            nn.Embedding(hashmap_size, feature_per_level)
            for _ in range(num_levels)
        ])
        
        # Initialize hash tables
        for table in self.hash_tables:
            nn.init.uniform_(table.weight, -1e-4, 1e-4)
        
        self.output_dim = num_levels * feature_per_level
    
    def _hash_function(self, coords: torch.Tensor) -> torch.Tensor:
        """Simple hash function for coordinates."""
        # Simplified hash - in practice would use more sophisticated hashing
        primes = torch.tensor([1, 2654435761, 805459861], device=coords.device, dtype=torch.long)
        coords_int = coords.long()
        
        hashed = coords_int[..., 0] * primes[0]
        for i in range(1, coords.shape[-1]):
            hashed ^= coords_int[..., i] * primes[i % len(primes)]
        
        return hashed % (2 ** self.log2_hashmap_size)
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Apply hash encoding to points."""
        batch_size = points.shape[0]
        device = points.device
        
        # Normalize points to [0, 1]
        points_normalized = (points + 1) / 2  # Assuming input in [-1, 1]
        points_normalized = torch.clamp(points_normalized, 0, 1)
        
        encoded_features = []
        
        for level, resolution in enumerate(self.resolutions):
            # Scale points to current resolution
            scaled_points = points_normalized * (resolution - 1)
            
            # Get grid coordinates
            grid_coords = torch.floor(scaled_points).long()
            
            # Hash coordinates
            hashed_indices = self._hash_function(grid_coords)
            
            # Lookup features
            features = self.hash_tables[level](hashed_indices)
            encoded_features.append(features)
        
        return torch.cat(encoded_features, dim=-1) 