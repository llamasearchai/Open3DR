"""
Dataset classes for loading and managing training data.

This module provides dataset classes for various data types used in
3D reconstruction and neural rendering.
"""

import asyncio
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import json

import numpy as np
import cv2
from loguru import logger
import torch
from torch.utils.data import Dataset

from ..core.types import Camera, Vector3
from ..core.utils import DataLoader, Validator


class BaseDataset(Dataset):
    """Base dataset class with common functionality."""

    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)
        self.data_loader = DataLoader()
        self.validator = Validator()
        
        # Validate data path
        self.validator.validate_directory_exists(self.data_path)
        
        # Dataset state
        self.is_loaded = False
        self.metadata: Dict[str, Any] = {}

    def __len__(self) -> int:
        """Return dataset size."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item."""
        raise NotImplementedError

    async def load_async(self) -> None:
        """Asynchronously load dataset."""
        if self.is_loaded:
            return
        
        await self._load_data()
        self.is_loaded = True
        logger.info(f"Loaded {self.__class__.__name__} with {len(self)} items")

    async def _load_data(self) -> None:
        """Load dataset data (to be implemented by subclasses)."""
        raise NotImplementedError


class ImageDataset(BaseDataset):
    """
    Dataset for loading image sequences with camera poses.
    
    Supports COLMAP, NeRF synthetic, and Blender formats.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        format: str = "auto",
        scale_factor: float = 1.0,
        load_masks: bool = False
    ):
        super().__init__(data_path)
        
        self.format = format
        self.scale_factor = scale_factor
        self.load_masks = load_masks
        
        # Data containers
        self.images: List[np.ndarray] = []
        self.poses: List[np.ndarray] = []
        self.intrinsics: List[Dict[str, float]] = []
        self.bounds: Optional[np.ndarray] = None
        self.masks: List[np.ndarray] = []
        self.point_cloud: Optional[np.ndarray] = None
        
        # Image metadata
        self.image_paths: List[Path] = []
        self.image_names: List[str] = []

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get image, pose, and intrinsics for given index."""
        item = {
            "image": self.images[idx],
            "pose": self.poses[idx],
            "intrinsics": self.intrinsics[idx],
            "index": idx,
            "image_path": str(self.image_paths[idx]),
            "image_name": self.image_names[idx]
        }
        
        if self.bounds is not None:
            item["bounds"] = self.bounds
            
        if self.masks and idx < len(self.masks):
            item["mask"] = self.masks[idx]
            
        return item

    async def _load_data(self) -> None:
        """Load image dataset based on detected or specified format."""
        # Auto-detect format if not specified
        if self.format == "auto":
            self.format = self._detect_format()
        
        if self.format == "colmap":
            await self._load_colmap()
        elif self.format == "nerf_synthetic":
            await self._load_nerf_synthetic()
        elif self.format == "blender":
            await self._load_blender()
        elif self.format == "images":
            await self._load_images_only()
        else:
            raise ValueError(f"Unsupported format: {self.format}")
        
        # Apply scale factor if needed
        if self.scale_factor != 1.0:
            self._apply_scale_factor()
        
        # Load masks if requested
        if self.load_masks:
            await self._load_masks()

    def _detect_format(self) -> str:
        """Auto-detect dataset format based on directory contents."""
        # Check for COLMAP files
        if (self.data_path / "sparse").exists() or (self.data_path / "cameras.txt").exists():
            return "colmap"
        
        # Check for NeRF synthetic format
        if (self.data_path / "transforms_train.json").exists():
            return "nerf_synthetic"
        
        # Check for Blender format
        if (self.data_path / "transforms.json").exists():
            return "blender"
        
        # Default to images only
        return "images"

    async def _load_colmap(self) -> None:
        """Load COLMAP format dataset."""
        from .formats import ColmapReader
        
        reader = ColmapReader(self.data_path)
        data = await reader.load_async()
        
        self.images = data["images"]
        self.poses = data["poses"]
        self.intrinsics = data["intrinsics"]
        self.bounds = data.get("bounds")
        self.point_cloud = data.get("point_cloud")
        self.image_paths = data["image_paths"]
        self.image_names = data["image_names"]

    async def _load_nerf_synthetic(self) -> None:
        """Load NeRF synthetic format dataset."""
        from .formats import NerfSyntheticReader
        
        reader = NerfSyntheticReader(self.data_path)
        data = await reader.load_async()
        
        self.images = data["images"]
        self.poses = data["poses"]
        self.intrinsics = data["intrinsics"]
        self.bounds = data.get("bounds")
        self.image_paths = data["image_paths"]
        self.image_names = data["image_names"]

    async def _load_blender(self) -> None:
        """Load Blender format dataset."""
        from .formats import BlenderReader
        
        reader = BlenderReader(self.data_path)
        data = await reader.load_async()
        
        self.images = data["images"]
        self.poses = data["poses"]
        self.intrinsics = data["intrinsics"]
        self.bounds = data.get("bounds")
        self.image_paths = data["image_paths"]
        self.image_names = data["image_names"]

    async def _load_images_only(self) -> None:
        """Load images without pose information."""
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(self.data_path.glob(f"*{ext}")))
            image_files.extend(list(self.data_path.glob(f"*{ext.upper()}")))
        
        image_files.sort()
        
        if not image_files:
            raise ValueError(f"No images found in {self.data_path}")
        
        # Load images
        for image_path in image_files:
            image = cv2.imread(str(image_path))
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.images.append(image)
                self.image_paths.append(image_path)
                self.image_names.append(image_path.name)
                
                # Create default intrinsics
                h, w = image.shape[:2]
                intrinsics = {
                    "fx": w / 2,
                    "fy": h / 2,
                    "cx": w / 2,
                    "cy": h / 2
                }
                self.intrinsics.append(intrinsics)
                
                # Create identity pose
                pose = np.eye(4)
                self.poses.append(pose)

    async def _load_masks(self) -> None:
        """Load mask images if available."""
        mask_dir = self.data_path / "masks"
        if not mask_dir.exists():
            logger.warning("Masks requested but mask directory not found")
            return
        
        for image_name in self.image_names:
            # Try different mask file formats
            mask_name = Path(image_name).stem
            mask_paths = [
                mask_dir / f"{mask_name}.png",
                mask_dir / f"{mask_name}.jpg",
                mask_dir / f"{mask_name}_mask.png"
            ]
            
            mask_loaded = False
            for mask_path in mask_paths:
                if mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        self.masks.append(mask)
                        mask_loaded = True
                        break
            
            if not mask_loaded:
                # Create default mask (all ones)
                h, w = self.images[len(self.masks)].shape[:2]
                self.masks.append(np.ones((h, w), dtype=np.uint8) * 255)

    def _apply_scale_factor(self) -> None:
        """Apply scale factor to images and intrinsics."""
        if self.scale_factor == 1.0:
            return
        
        scaled_images = []
        scaled_intrinsics = []
        
        for i, (image, intrinsics) in enumerate(zip(self.images, self.intrinsics)):
            # Scale image
            new_h = int(image.shape[0] * self.scale_factor)
            new_w = int(image.shape[1] * self.scale_factor)
            scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scaled_images.append(scaled_image)
            
            # Scale intrinsics
            scaled_intrinsics.append({
                "fx": intrinsics["fx"] * self.scale_factor,
                "fy": intrinsics["fy"] * self.scale_factor,
                "cx": intrinsics["cx"] * self.scale_factor,
                "cy": intrinsics["cy"] * self.scale_factor
            })
        
        self.images = scaled_images
        self.intrinsics = scaled_intrinsics
        
        # Scale masks if present
        if self.masks:
            scaled_masks = []
            for mask in self.masks:
                new_h = int(mask.shape[0] * self.scale_factor)
                new_w = int(mask.shape[1] * self.scale_factor)
                scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                scaled_masks.append(scaled_mask)
            self.masks = scaled_masks

    def get_bounds(self) -> Optional[np.ndarray]:
        """Get scene bounds if available."""
        return self.bounds

    def get_camera_info(self, idx: int) -> Camera:
        """Get Camera object for given index."""
        intrinsics = self.intrinsics[idx]
        pose = self.poses[idx]
        
        # Extract position from pose
        position = Vector3(pose[0, 3], pose[1, 3], pose[2, 3])
        
        # Extract rotation (simplified)
        rotation = Vector3(0, 0, 0)  # Would need proper rotation extraction
        
        h, w = self.images[idx].shape[:2]
        
        return Camera(
            width=w,
            height=h,
            fx=intrinsics["fx"],
            fy=intrinsics["fy"],
            cx=intrinsics["cx"],
            cy=intrinsics["cy"],
            position=position,
            rotation=rotation
        )


class PointCloudDataset(BaseDataset):
    """Dataset for loading and processing point clouds."""

    def __init__(
        self,
        data_path: Union[str, Path],
        max_points: Optional[int] = None,
        load_colors: bool = True,
        load_normals: bool = True
    ):
        super().__init__(data_path)
        
        self.max_points = max_points
        self.load_colors = load_colors
        self.load_normals = load_normals
        
        # Data containers
        self.points: List[np.ndarray] = []
        self.colors: List[np.ndarray] = []
        self.normals: List[np.ndarray] = []
        self.point_cloud_paths: List[Path] = []

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get point cloud data for given index."""
        item = {
            "points": self.points[idx],
            "index": idx,
            "path": str(self.point_cloud_paths[idx])
        }
        
        if self.colors and idx < len(self.colors):
            item["colors"] = self.colors[idx]
            
        if self.normals and idx < len(self.normals):
            item["normals"] = self.normals[idx]
            
        return item

    async def _load_data(self) -> None:
        """Load point cloud files."""
        # Find point cloud files
        pc_extensions = {'.ply', '.pcd', '.xyz', '.pts'}
        pc_files = []
        
        for ext in pc_extensions:
            pc_files.extend(list(self.data_path.glob(f"*{ext}")))
        
        pc_files.sort()
        
        if not pc_files:
            raise ValueError(f"No point cloud files found in {self.data_path}")
        
        # Load point clouds
        for pc_path in pc_files:
            try:
                if pc_path.suffix.lower() == '.ply':
                    data = self._load_ply(pc_path)
                elif pc_path.suffix.lower() == '.pcd':
                    data = self._load_pcd(pc_path)
                else:
                    data = self._load_xyz(pc_path)
                
                # Subsample if needed
                if self.max_points and len(data["points"]) > self.max_points:
                    indices = np.random.choice(len(data["points"]), self.max_points, replace=False)
                    data["points"] = data["points"][indices]
                    if "colors" in data:
                        data["colors"] = data["colors"][indices]
                    if "normals" in data:
                        data["normals"] = data["normals"][indices]
                
                self.points.append(data["points"])
                self.point_cloud_paths.append(pc_path)
                
                if self.load_colors and "colors" in data:
                    self.colors.append(data["colors"])
                    
                if self.load_normals and "normals" in data:
                    self.normals.append(data["normals"])
                    
            except Exception as e:
                logger.warning(f"Failed to load point cloud {pc_path}: {e}")

    def _load_ply(self, file_path: Path) -> Dict[str, np.ndarray]:
        """Load PLY file."""
        from .formats import PLYReader
        
        reader = PLYReader()
        return reader.load(file_path)

    def _load_pcd(self, file_path: Path) -> Dict[str, np.ndarray]:
        """Load PCD file (simplified)."""
        # Simplified PCD loader - would use proper library in practice
        points = []
        colors = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            # Find data start
            data_start = 0
            for i, line in enumerate(lines):
                if line.strip() == "DATA ascii":
                    data_start = i + 1
                    break
            
            # Parse points
            for line in lines[data_start:]:
                parts = line.strip().split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    points.append([x, y, z])
                    
                    if len(parts) >= 6:  # RGB
                        r, g, b = float(parts[3]), float(parts[4]), float(parts[5])
                        colors.append([r, g, b])
        
        data = {"points": np.array(points)}
        if colors:
            data["colors"] = np.array(colors)
            
        return data

    def _load_xyz(self, file_path: Path) -> Dict[str, np.ndarray]:
        """Load XYZ file."""
        points = []
        colors = []
        
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    points.append([x, y, z])
                    
                    if len(parts) >= 6:
                        r, g, b = float(parts[3]), float(parts[4]), float(parts[5])
                        colors.append([r, g, b])
        
        data = {"points": np.array(points)}
        if colors:
            data["colors"] = np.array(colors)
            
        return data


class MeshDataset(BaseDataset):
    """Dataset for loading and processing 3D meshes."""

    def __init__(
        self,
        data_path: Union[str, Path],
        load_textures: bool = True,
        load_materials: bool = True
    ):
        super().__init__(data_path)
        
        self.load_textures = load_textures
        self.load_materials = load_materials
        
        # Data containers
        self.vertices: List[np.ndarray] = []
        self.faces: List[np.ndarray] = []
        self.textures: List[np.ndarray] = []
        self.normals: List[np.ndarray] = []
        self.mesh_paths: List[Path] = []

    def __len__(self) -> int:
        return len(self.vertices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get mesh data for given index."""
        item = {
            "vertices": self.vertices[idx],
            "faces": self.faces[idx],
            "index": idx,
            "path": str(self.mesh_paths[idx])
        }
        
        if self.textures and idx < len(self.textures):
            item["textures"] = self.textures[idx]
            
        if self.normals and idx < len(self.normals):
            item["normals"] = self.normals[idx]
            
        return item

    async def _load_data(self) -> None:
        """Load mesh files."""
        # Find mesh files
        mesh_extensions = {'.obj', '.ply', '.fbx', '.dae', '.3ds'}
        mesh_files = []
        
        for ext in mesh_extensions:
            mesh_files.extend(list(self.data_path.glob(f"*{ext}")))
        
        mesh_files.sort()
        
        if not mesh_files:
            raise ValueError(f"No mesh files found in {self.data_path}")
        
        # Load meshes
        for mesh_path in mesh_files:
            try:
                if mesh_path.suffix.lower() == '.obj':
                    data = self._load_obj(mesh_path)
                elif mesh_path.suffix.lower() == '.ply':
                    data = self._load_ply_mesh(mesh_path)
                else:
                    logger.warning(f"Unsupported mesh format: {mesh_path.suffix}")
                    continue
                
                self.vertices.append(data["vertices"])
                self.faces.append(data["faces"])
                self.mesh_paths.append(mesh_path)
                
                if "normals" in data:
                    self.normals.append(data["normals"])
                    
            except Exception as e:
                logger.warning(f"Failed to load mesh {mesh_path}: {e}")

    def _load_obj(self, file_path: Path) -> Dict[str, np.ndarray]:
        """Load OBJ file."""
        from .formats import OBJReader
        
        reader = OBJReader()
        return reader.load(file_path)

    def _load_ply_mesh(self, file_path: Path) -> Dict[str, np.ndarray]:
        """Load PLY mesh file."""
        from .formats import PLYReader
        
        reader = PLYReader()
        return reader.load(file_path)


class SensorDataset(BaseDataset):
    """Dataset for loading sensor data (camera, LiDAR, radar, IMU)."""

    def __init__(
        self,
        data_path: Union[str, Path],
        sensor_types: List[str] = None
    ):
        super().__init__(data_path)
        
        if sensor_types is None:
            sensor_types = ["camera", "lidar", "radar", "imu"]
        self.sensor_types = sensor_types
        
        # Data containers
        self.sensor_data: Dict[str, List[Dict[str, Any]]] = {
            sensor_type: [] for sensor_type in sensor_types
        }
        self.timestamps: List[float] = []

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sensor data for given timestamp index."""
        item = {
            "timestamp": self.timestamps[idx],
            "index": idx
        }
        
        for sensor_type in self.sensor_types:
            if idx < len(self.sensor_data[sensor_type]):
                item[sensor_type] = self.sensor_data[sensor_type][idx]
                
        return item

    async def _load_data(self) -> None:
        """Load sensor data files."""
        # Look for sensor data files
        for sensor_type in self.sensor_types:
            sensor_dir = self.data_path / sensor_type
            if sensor_dir.exists():
                await self._load_sensor_type(sensor_type, sensor_dir)
        
        # Load timestamps
        timestamp_file = self.data_path / "timestamps.txt"
        if timestamp_file.exists():
            with open(timestamp_file, 'r') as f:
                self.timestamps = [float(line.strip()) for line in f]
        else:
            # Generate timestamps based on data length
            max_length = max(len(data) for data in self.sensor_data.values())
            self.timestamps = list(range(max_length))

    async def _load_sensor_type(self, sensor_type: str, sensor_dir: Path) -> None:
        """Load data for specific sensor type."""
        if sensor_type == "camera":
            await self._load_camera_data(sensor_dir)
        elif sensor_type == "lidar":
            await self._load_lidar_data(sensor_dir)
        elif sensor_type == "radar":
            await self._load_radar_data(sensor_dir)
        elif sensor_type == "imu":
            await self._load_imu_data(sensor_dir)

    async def _load_camera_data(self, camera_dir: Path) -> None:
        """Load camera sensor data."""
        # Load images
        image_files = sorted(list(camera_dir.glob("*.jpg")) + list(camera_dir.glob("*.png")))
        
        for image_file in image_files:
            image = cv2.imread(str(image_file))
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.sensor_data["camera"].append({
                    "image": image,
                    "path": str(image_file),
                    "name": image_file.name
                })

    async def _load_lidar_data(self, lidar_dir: Path) -> None:
        """Load LiDAR sensor data."""
        # Load point cloud files
        pc_files = sorted(list(lidar_dir.glob("*.ply")) + list(lidar_dir.glob("*.pcd")))
        
        for pc_file in pc_files:
            try:
                if pc_file.suffix.lower() == '.ply':
                    from .formats import PLYReader
                    reader = PLYReader()
                    data = reader.load(pc_file)
                else:
                    # Load PCD or other format
                    data = {"points": np.random.rand(1000, 3)}  # Placeholder
                
                self.sensor_data["lidar"].append({
                    "points": data["points"],
                    "path": str(pc_file),
                    "name": pc_file.name
                })
            except Exception as e:
                logger.warning(f"Failed to load LiDAR data {pc_file}: {e}")

    async def _load_radar_data(self, radar_dir: Path) -> None:
        """Load radar sensor data."""
        # Placeholder for radar data loading
        # Would load radar-specific formats
        pass

    async def _load_imu_data(self, imu_dir: Path) -> None:
        """Load IMU sensor data."""
        # Look for IMU data files
        imu_file = imu_dir / "imu.txt"
        if imu_file.exists():
            with open(imu_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        # Parse acceleration and angular velocity
                        accel = [float(parts[0]), float(parts[1]), float(parts[2])]
                        gyro = [float(parts[3]), float(parts[4]), float(parts[5])]
                        
                        self.sensor_data["imu"].append({
                            "acceleration": np.array(accel),
                            "angular_velocity": np.array(gyro)
                        }) 