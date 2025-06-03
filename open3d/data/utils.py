"""
Data utility functions for 3D reconstruction and computer vision.

This module provides various utility functions for data processing,
camera operations, and geometric calculations.
"""

from typing import Tuple, Optional, List, Union, Dict, Any
import numpy as np
import cv2
from pathlib import Path

from ..core.types import Camera, Vector3


class DataUtils:
    """General data utility functions."""
    
    @staticmethod
    def load_image(path: Union[str, Path]) -> np.ndarray:
        """Load image from file."""
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not load image from {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def save_image(image: np.ndarray, path: Union[str, Path]) -> None:
        """Save image to file."""
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), image_bgr)
    
    @staticmethod
    def compute_image_bounds(images: List[np.ndarray]) -> Dict[str, Any]:
        """Compute bounds statistics for a list of images."""
        if not images:
            return {}
        
        heights = [img.shape[0] for img in images]
        widths = [img.shape[1] for img in images]
        
        return {
            "min_height": min(heights),
            "max_height": max(heights),
            "min_width": min(widths),
            "max_width": max(widths),
            "mean_height": np.mean(heights),
            "mean_width": np.mean(widths)
        }


class CameraUtils:
    """Camera-related utility functions."""
    
    @staticmethod
    def create_camera_from_intrinsics(
        intrinsics: Dict[str, float],
        width: int,
        height: int,
        position: Optional[Vector3] = None,
        rotation: Optional[Vector3] = None
    ) -> Camera:
        """Create Camera object from intrinsics dictionary."""
        return Camera(
            width=width,
            height=height,
            fx=intrinsics["fx"],
            fy=intrinsics["fy"],
            cx=intrinsics["cx"],
            cy=intrinsics["cy"],
            position=position or Vector3(0, 0, 0),
            rotation=rotation or Vector3(0, 0, 0)
        )
    
    @staticmethod
    def compute_focal_from_fov(fov_degrees: float, image_width: int) -> float:
        """Compute focal length from field of view."""
        fov_radians = np.radians(fov_degrees)
        return 0.5 * image_width / np.tan(0.5 * fov_radians)
    
    @staticmethod
    def compute_fov_from_focal(focal: float, image_width: int) -> float:
        """Compute field of view from focal length."""
        fov_radians = 2 * np.arctan(0.5 * image_width / focal)
        return np.degrees(fov_radians)
    
    @staticmethod
    def pixel_to_ray(
        pixel_x: float,
        pixel_y: float,
        camera: Camera
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert pixel coordinates to ray origin and direction."""
        # Convert to normalized device coordinates
        x = (pixel_x - camera.cx) / camera.fx
        y = -(pixel_y - camera.cy) / camera.fy  # Flip Y
        
        # Ray direction in camera space
        direction = np.array([x, y, -1.0])
        direction = direction / np.linalg.norm(direction)
        
        # Ray origin (camera position)
        origin = np.array([camera.position.x, camera.position.y, camera.position.z])
        
        return origin, direction
    
    @staticmethod
    def project_point_to_pixel(
        point_3d: np.ndarray,
        camera: Camera
    ) -> Tuple[float, float]:
        """Project 3D point to pixel coordinates."""
        # Simplified projection (assumes point is in camera coordinate system)
        x, y, z = point_3d
        
        if z <= 0:
            return float('inf'), float('inf')
        
        pixel_x = (x / z) * camera.fx + camera.cx
        pixel_y = -(y / z) * camera.fy + camera.cy  # Flip Y
        
        return pixel_x, pixel_y


class GeometryUtils:
    """Geometric utility functions."""
    
    @staticmethod
    def compute_bounding_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute axis-aligned bounding box of points."""
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        return min_bounds, max_bounds
    
    @staticmethod
    def compute_centroid(points: np.ndarray) -> np.ndarray:
        """Compute centroid of points."""
        return np.mean(points, axis=0)
    
    @staticmethod
    def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Normalize points to unit cube centered at origin."""
        centroid = GeometryUtils.compute_centroid(points)
        centered_points = points - centroid
        
        # Compute scale to fit in unit cube
        max_extent = np.max(np.abs(centered_points))
        if max_extent > 0:
            scale = 1.0 / max_extent
            normalized_points = centered_points * scale
        else:
            scale = 1.0
            normalized_points = centered_points
        
        normalization_info = {
            "centroid": centroid,
            "scale": scale
        }
        
        return normalized_points, normalization_info
    
    @staticmethod
    def denormalize_points(
        normalized_points: np.ndarray,
        normalization_info: Dict[str, Any]
    ) -> np.ndarray:
        """Denormalize points using stored normalization info."""
        points = normalized_points / normalization_info["scale"]
        points = points + normalization_info["centroid"]
        return points
    
    @staticmethod
    def compute_point_cloud_normals(
        points: np.ndarray,
        k: int = 20
    ) -> np.ndarray:
        """Compute point cloud normals using local neighborhoods."""
        # Simplified normal computation
        # In practice would use proper nearest neighbor search
        normals = np.zeros_like(points)
        
        for i, point in enumerate(points):
            # Find k nearest neighbors (simplified)
            distances = np.linalg.norm(points - point, axis=1)
            neighbor_indices = np.argsort(distances)[1:k+1]  # Skip self
            
            if len(neighbor_indices) >= 3:
                neighbors = points[neighbor_indices]
                
                # Compute normal using PCA
                centered = neighbors - np.mean(neighbors, axis=0)
                _, _, vh = np.linalg.svd(centered)
                normal = vh[-1]  # Last row is normal direction
                normals[i] = normal
        
        return normals
    
    @staticmethod
    def compute_mesh_area(vertices: np.ndarray, faces: np.ndarray) -> float:
        """Compute total surface area of a mesh."""
        total_area = 0.0
        
        for face in faces:
            if len(face) >= 3:
                # Compute triangle area
                v0, v1, v2 = vertices[face[:3]]
                edge1 = v1 - v0
                edge2 = v2 - v0
                cross = np.cross(edge1, edge2)
                area = 0.5 * np.linalg.norm(cross)
                total_area += area
        
        return total_area
    
    @staticmethod
    def compute_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
        """Compute volume of a closed mesh."""
        volume = 0.0
        
        for face in faces:
            if len(face) >= 3:
                # Use divergence theorem
                v0, v1, v2 = vertices[face[:3]]
                volume += np.dot(v0, np.cross(v1, v2)) / 6.0
        
        return abs(volume)
    
    @staticmethod
    def sample_points_on_mesh(
        vertices: np.ndarray,
        faces: np.ndarray,
        num_points: int
    ) -> np.ndarray:
        """Sample points uniformly on mesh surface."""
        # Compute face areas
        face_areas = []
        for face in faces:
            if len(face) >= 3:
                v0, v1, v2 = vertices[face[:3]]
                edge1 = v1 - v0
                edge2 = v2 - v0
                area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                face_areas.append(area)
            else:
                face_areas.append(0.0)
        
        face_areas = np.array(face_areas)
        face_probs = face_areas / np.sum(face_areas)
        
        # Sample faces according to area
        sampled_faces = np.random.choice(
            len(faces), 
            size=num_points, 
            p=face_probs
        )
        
        # Sample points on each selected face
        sampled_points = []
        for face_idx in sampled_faces:
            face = faces[face_idx]
            if len(face) >= 3:
                v0, v1, v2 = vertices[face[:3]]
                
                # Sample random barycentric coordinates
                r1, r2 = np.random.random(2)
                if r1 + r2 > 1:
                    r1, r2 = 1 - r1, 1 - r2
                
                point = (1 - r1 - r2) * v0 + r1 * v1 + r2 * v2
                sampled_points.append(point)
        
        return np.array(sampled_points) 