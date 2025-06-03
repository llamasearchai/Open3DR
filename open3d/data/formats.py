"""
Data format readers for various 3D data formats.

This module provides readers for common 3D reconstruction and computer vision
data formats including COLMAP, NeRF synthetic, Blender, and mesh formats.
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import struct

import numpy as np
import cv2
from loguru import logger


class BaseReader:
    """Base class for data format readers."""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path) if data_path else None

    async def load_async(self) -> Dict[str, Any]:
        """Load data asynchronously."""
        return await asyncio.to_thread(self.load)

    def load(self) -> Dict[str, Any]:
        """Load data synchronously."""
        raise NotImplementedError


class ColmapReader(BaseReader):
    """Reader for COLMAP data format."""

    def load(self) -> Dict[str, Any]:
        """Load COLMAP reconstruction data."""
        if not self.data_path or not self.data_path.exists():
            raise ValueError("Invalid COLMAP data path")

        # Look for sparse reconstruction
        sparse_dir = self.data_path / "sparse"
        if not sparse_dir.exists():
            sparse_dir = self.data_path / "sparse" / "0"
        
        if not sparse_dir.exists():
            # Try text format files
            cameras_file = self.data_path / "cameras.txt"
            images_file = self.data_path / "images.txt"
            points_file = self.data_path / "points3D.txt"
            
            if cameras_file.exists() and images_file.exists():
                return self._load_text_format()
            else:
                raise FileNotFoundError("No COLMAP data found")
        
        # Load binary format
        cameras_file = sparse_dir / "cameras.bin"
        images_file = sparse_dir / "images.bin"
        points_file = sparse_dir / "points3D.bin"
        
        if cameras_file.exists() and images_file.exists():
            return self._load_binary_format(sparse_dir)
        else:
            # Try text format in sparse directory
            cameras_file = sparse_dir / "cameras.txt"
            images_file = sparse_dir / "images.txt"
            points_file = sparse_dir / "points3D.txt"
            
            if cameras_file.exists() and images_file.exists():
                return self._load_text_format(sparse_dir)
            else:
                raise FileNotFoundError("No valid COLMAP format found")

    def _load_binary_format(self, sparse_dir: Path) -> Dict[str, Any]:
        """Load COLMAP binary format."""
        # Load cameras
        cameras = self._read_cameras_binary(sparse_dir / "cameras.bin")
        
        # Load images
        images_data = self._read_images_binary(sparse_dir / "images.bin")
        
        # Load point cloud if available
        points_file = sparse_dir / "points3D.bin"
        point_cloud = None
        if points_file.exists():
            point_cloud = self._read_points3d_binary(points_file)
        
        # Load actual image files
        images = []
        poses = []
        intrinsics = []
        image_paths = []
        image_names = []
        
        for image_id, image_info in images_data.items():
            image_path = self.data_path / "images" / image_info["name"]
            if image_path.exists():
                image = cv2.imread(str(image_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    
                    # Convert pose
                    pose = self._quat_to_pose(image_info["quat"], image_info["tvec"])
                    poses.append(pose)
                    
                    # Get camera intrinsics
                    camera_id = image_info["camera_id"]
                    camera_info = cameras[camera_id]
                    intrinsics.append({
                        "fx": camera_info["params"][0],
                        "fy": camera_info["params"][1],
                        "cx": camera_info["params"][2],
                        "cy": camera_info["params"][3]
                    })
                    
                    image_paths.append(image_path)
                    image_names.append(image_info["name"])
        
        return {
            "images": images,
            "poses": poses,
            "intrinsics": intrinsics,
            "point_cloud": point_cloud,
            "image_paths": image_paths,
            "image_names": image_names,
            "bounds": self._compute_bounds(poses) if poses else None
        }

    def _load_text_format(self, sparse_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Load COLMAP text format."""
        if sparse_dir is None:
            sparse_dir = self.data_path
        
        # Load cameras
        cameras = self._read_cameras_text(sparse_dir / "cameras.txt")
        
        # Load images
        images_data = self._read_images_text(sparse_dir / "images.txt")
        
        # Load point cloud if available
        points_file = sparse_dir / "points3D.txt"
        point_cloud = None
        if points_file.exists():
            point_cloud = self._read_points3d_text(points_file)
        
        # Process similar to binary format
        images = []
        poses = []
        intrinsics = []
        image_paths = []
        image_names = []
        
        for image_id, image_info in images_data.items():
            image_path = self.data_path / "images" / image_info["name"]
            if image_path.exists():
                image = cv2.imread(str(image_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    
                    pose = self._quat_to_pose(image_info["quat"], image_info["tvec"])
                    poses.append(pose)
                    
                    camera_id = image_info["camera_id"]
                    camera_info = cameras[camera_id]
                    intrinsics.append({
                        "fx": camera_info["params"][0],
                        "fy": camera_info["params"][1], 
                        "cx": camera_info["params"][2],
                        "cy": camera_info["params"][3]
                    })
                    
                    image_paths.append(image_path)
                    image_names.append(image_info["name"])
        
        return {
            "images": images,
            "poses": poses,
            "intrinsics": intrinsics,
            "point_cloud": point_cloud,
            "image_paths": image_paths,
            "image_names": image_names,
            "bounds": self._compute_bounds(poses) if poses else None
        }

    def _read_cameras_binary(self, file_path: Path) -> Dict[int, Dict[str, Any]]:
        """Read cameras from binary file."""
        cameras = {}
        with open(file_path, "rb") as f:
            num_cameras = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_cameras):
                camera_id = struct.unpack("<I", f.read(4))[0]
                model_id = struct.unpack("<I", f.read(4))[0]
                width = struct.unpack("<Q", f.read(8))[0]
                height = struct.unpack("<Q", f.read(8))[0]
                
                # Read parameters (assuming PINHOLE model)
                num_params = 4  # fx, fy, cx, cy
                params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
                
                cameras[camera_id] = {
                    "model_id": model_id,
                    "width": width,
                    "height": height,
                    "params": params
                }
        
        return cameras

    def _read_cameras_text(self, file_path: Path) -> Dict[int, Dict[str, Any]]:
        """Read cameras from text file."""
        cameras = {}
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                parts = line.split()
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(x) for x in parts[4:]]
                
                cameras[camera_id] = {
                    "model": model,
                    "width": width,
                    "height": height,
                    "params": params
                }
        
        return cameras

    def _read_images_binary(self, file_path: Path) -> Dict[int, Dict[str, Any]]:
        """Read images from binary file."""
        images = {}
        with open(file_path, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_images):
                image_id = struct.unpack("<I", f.read(4))[0]
                quat = struct.unpack("<4d", f.read(32))
                tvec = struct.unpack("<3d", f.read(24))
                camera_id = struct.unpack("<I", f.read(4))[0]
                
                # Read image name
                name_length = 0
                name_bytes = b""
                while True:
                    char = f.read(1)
                    if char == b"\x00":
                        break
                    name_bytes += char
                
                name = name_bytes.decode("utf-8")
                
                # Skip 2D points for now
                num_points2d = struct.unpack("<Q", f.read(8))[0]
                f.read(24 * num_points2d)  # Skip point data
                
                images[image_id] = {
                    "quat": quat,
                    "tvec": tvec,
                    "camera_id": camera_id,
                    "name": name
                }
        
        return images

    def _read_images_text(self, file_path: Path) -> Dict[int, Dict[str, Any]]:
        """Read images from text file."""
        images = {}
        with open(file_path, "r") as f:
            lines = f.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line or line.startswith("#"):
                    i += 1
                    continue
                
                parts = line.split()
                image_id = int(parts[0])
                quat = [float(x) for x in parts[1:5]]  # qw, qx, qy, qz
                tvec = [float(x) for x in parts[5:8]]
                camera_id = int(parts[8])
                name = parts[9]
                
                images[image_id] = {
                    "quat": quat,
                    "tvec": tvec,
                    "camera_id": camera_id,
                    "name": name
                }
                
                i += 2  # Skip point line

        return images

    def _read_points3d_binary(self, file_path: Path) -> np.ndarray:
        """Read 3D points from binary file."""
        points = []
        with open(file_path, "rb") as f:
            num_points = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_points):
                point_id = struct.unpack("<Q", f.read(8))[0]
                xyz = struct.unpack("<3d", f.read(24))
                rgb = struct.unpack("<3B", f.read(3))
                error = struct.unpack("<d", f.read(8))[0]
                
                # Skip track information
                track_length = struct.unpack("<Q", f.read(8))[0]
                f.read(8 * track_length)  # Skip track data
                
                points.append([xyz[0], xyz[1], xyz[2], rgb[0], rgb[1], rgb[2]])
        
        return np.array(points) if points else None

    def _read_points3d_text(self, file_path: Path) -> np.ndarray:
        """Read 3D points from text file."""
        points = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                parts = line.split()
                point_id = int(parts[0])
                xyz = [float(x) for x in parts[1:4]]
                rgb = [int(x) for x in parts[4:7]]
                
                points.append([xyz[0], xyz[1], xyz[2], rgb[0], rgb[1], rgb[2]])
        
        return np.array(points) if points else None

    def _quat_to_pose(self, quat: List[float], tvec: List[float]) -> np.ndarray:
        """Convert quaternion and translation to pose matrix."""
        # Quaternion to rotation matrix
        w, x, y, z = quat
        
        # Normalize quaternion
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Convert to rotation matrix
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        # Create pose matrix
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = tvec
        
        return pose

    def _compute_bounds(self, poses: List[np.ndarray]) -> np.ndarray:
        """Compute scene bounds from camera poses."""
        positions = np.array([pose[:3, 3] for pose in poses])
        
        min_bounds = np.min(positions, axis=0)
        max_bounds = np.max(positions, axis=0)
        
        # Add some padding
        center = (min_bounds + max_bounds) / 2
        extent = max_bounds - min_bounds
        extent = np.max(extent) * 1.2  # 20% padding
        
        return np.array([
            center - extent/2,
            center + extent/2
        ])


class NerfSyntheticReader(BaseReader):
    """Reader for NeRF synthetic dataset format."""

    def load(self) -> Dict[str, Any]:
        """Load NeRF synthetic dataset."""
        if not self.data_path or not self.data_path.exists():
            raise ValueError("Invalid NeRF synthetic data path")

        # Load train split (could also load val/test)
        transforms_file = self.data_path / "transforms_train.json"
        if not transforms_file.exists():
            transforms_file = self.data_path / "transforms.json"
        
        if not transforms_file.exists():
            raise FileNotFoundError("No transforms file found")

        with open(transforms_file, "r") as f:
            data = json.load(f)

        # Extract camera info
        camera_angle_x = data.get("camera_angle_x", 0.8575560308385229)
        
        images = []
        poses = []
        intrinsics = []
        image_paths = []
        image_names = []
        
        for frame in data["frames"]:
            image_path = self.data_path / frame["file_path"]
            
            # Try different extensions
            if not image_path.exists():
                for ext in [".png", ".jpg", ".jpeg"]:
                    test_path = image_path.with_suffix(ext)
                    if test_path.exists():
                        image_path = test_path
                        break
            
            if image_path.exists():
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if image is not None:
                    # Handle RGBA images
                    if image.shape[2] == 4:
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                    else:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    images.append(image)
                    
                    # Convert transform matrix
                    transform = np.array(frame["transform_matrix"])
                    poses.append(transform)
                    
                    # Compute intrinsics from camera angle
                    h, w = image.shape[:2]
                    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)
                    
                    intrinsics.append({
                        "fx": focal,
                        "fy": focal,
                        "cx": w / 2,
                        "cy": h / 2
                    })
                    
                    image_paths.append(image_path)
                    image_names.append(image_path.name)

        return {
            "images": images,
            "poses": poses,
            "intrinsics": intrinsics,
            "image_paths": image_paths,
            "image_names": image_names,
            "bounds": self._compute_bounds(poses) if poses else None
        }

    def _compute_bounds(self, poses: List[np.ndarray]) -> np.ndarray:
        """Compute scene bounds from poses."""
        positions = np.array([pose[:3, 3] for pose in poses])
        
        min_bounds = np.min(positions, axis=0)
        max_bounds = np.max(positions, axis=0)
        
        center = (min_bounds + max_bounds) / 2
        extent = np.max(max_bounds - min_bounds) * 1.5
        
        return np.array([
            center - extent/2,
            center + extent/2
        ])


class BlenderReader(BaseReader):
    """Reader for Blender dataset format."""

    def load(self) -> Dict[str, Any]:
        """Load Blender dataset."""
        # Blender format is similar to NeRF synthetic
        # but may have different file structure
        
        transforms_file = self.data_path / "transforms.json"
        if not transforms_file.exists():
            raise FileNotFoundError("No transforms.json found")

        with open(transforms_file, "r") as f:
            data = json.load(f)

        camera_angle_x = data.get("camera_angle_x", 0.8575560308385229)
        
        images = []
        poses = []
        intrinsics = []
        image_paths = []
        image_names = []
        
        for frame in data["frames"]:
            # Blender may use different path format
            file_path = frame["file_path"]
            if file_path.startswith("./"):
                file_path = file_path[2:]
            
            image_path = self.data_path / file_path
            
            if image_path.exists():
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if image is not None:
                    if image.shape[2] == 4:
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                    else:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    images.append(image)
                    
                    transform = np.array(frame["transform_matrix"])
                    poses.append(transform)
                    
                    h, w = image.shape[:2]
                    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)
                    
                    intrinsics.append({
                        "fx": focal,
                        "fy": focal,
                        "cx": w / 2,
                        "cy": h / 2
                    })
                    
                    image_paths.append(image_path)
                    image_names.append(image_path.name)

        return {
            "images": images,
            "poses": poses,
            "intrinsics": intrinsics,
            "image_paths": image_paths,
            "image_names": image_names,
            "bounds": self._compute_bounds(poses) if poses else None
        }

    def _compute_bounds(self, poses: List[np.ndarray]) -> np.ndarray:
        """Compute scene bounds."""
        positions = np.array([pose[:3, 3] for pose in poses])
        
        min_bounds = np.min(positions, axis=0)
        max_bounds = np.max(positions, axis=0)
        
        center = (min_bounds + max_bounds) / 2
        extent = np.max(max_bounds - min_bounds) * 1.5
        
        return np.array([
            center - extent/2,
            center + extent/2
        ])


class PLYReader(BaseReader):
    """Reader for PLY point cloud and mesh files."""

    def load(self, file_path: Optional[Path] = None) -> Dict[str, np.ndarray]:
        """Load PLY file."""
        if file_path:
            ply_path = Path(file_path)
        else:
            ply_path = self.data_path
        
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY file not found: {ply_path}")

        # Simple PLY reader (would use a proper library in production)
        vertices = []
        faces = []
        colors = []
        normals = []
        
        with open(ply_path, 'r') as f:
            # Read header
            line = f.readline().strip()
            if line != "ply":
                raise ValueError("Invalid PLY file")
            
            format_line = f.readline().strip()
            if "ascii" not in format_line:
                raise ValueError("Only ASCII PLY files supported")
            
            vertex_count = 0
            face_count = 0
            has_colors = False
            has_normals = False
            
            # Parse header
            while True:
                line = f.readline().strip()
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                elif line.startswith("element face"):
                    face_count = int(line.split()[-1])
                elif line.startswith("property") and "red" in line:
                    has_colors = True
                elif line.startswith("property") and "nx" in line:
                    has_normals = True
                elif line == "end_header":
                    break
            
            # Read vertices
            for _ in range(vertex_count):
                parts = f.readline().strip().split()
                vertex = [float(parts[0]), float(parts[1]), float(parts[2])]
                vertices.append(vertex)
                
                if has_colors and len(parts) >= 6:
                    color = [int(parts[3]), int(parts[4]), int(parts[5])]
                    colors.append(color)
                
                if has_normals:
                    normal_start = 3 + (3 if has_colors else 0)
                    if len(parts) >= normal_start + 3:
                        normal = [float(parts[normal_start]), 
                                float(parts[normal_start + 1]), 
                                float(parts[normal_start + 2])]
                        normals.append(normal)
            
            # Read faces
            for _ in range(face_count):
                parts = f.readline().strip().split()
                if len(parts) >= 4:  # Triangle or quad
                    face_vertices = int(parts[0])
                    face = [int(parts[i + 1]) for i in range(face_vertices)]
                    faces.append(face)

        result = {"points" if not faces else "vertices": np.array(vertices)}
        
        if colors:
            result["colors"] = np.array(colors)
        if normals:
            result["normals"] = np.array(normals)
        if faces:
            result["faces"] = np.array(faces)
        
        return result


class OBJReader(BaseReader):
    """Reader for OBJ mesh files."""

    def load(self, file_path: Optional[Path] = None) -> Dict[str, np.ndarray]:
        """Load OBJ file."""
        if file_path:
            obj_path = Path(file_path)
        else:
            obj_path = self.data_path
        
        if not obj_path.exists():
            raise FileNotFoundError(f"OBJ file not found: {obj_path}")

        vertices = []
        faces = []
        normals = []
        texture_coords = []
        
        with open(obj_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                if parts[0] == 'v':  # Vertex
                    vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                    vertices.append(vertex)
                
                elif parts[0] == 'vn':  # Vertex normal
                    normal = [float(parts[1]), float(parts[2]), float(parts[3])]
                    normals.append(normal)
                
                elif parts[0] == 'vt':  # Texture coordinate
                    tc = [float(parts[1]), float(parts[2])]
                    texture_coords.append(tc)
                
                elif parts[0] == 'f':  # Face
                    face = []
                    for vertex_data in parts[1:]:
                        # Handle different face formats (v, v/vt, v/vt/vn, v//vn)
                        indices = vertex_data.split('/')
                        vertex_idx = int(indices[0]) - 1  # OBJ uses 1-based indexing
                        face.append(vertex_idx)
                    faces.append(face)

        result = {"vertices": np.array(vertices)}
        
        if faces:
            result["faces"] = np.array(faces)
        if normals:
            result["normals"] = np.array(normals)
        if texture_coords:
            result["texture_coords"] = np.array(texture_coords)
        
        return result


class FBXReader(BaseReader):
    """Reader for FBX mesh files."""

    def load(self, file_path: Optional[Path] = None) -> Dict[str, np.ndarray]:
        """Load FBX file."""
        # FBX is a complex binary format
        # This is a placeholder - would need a proper FBX library
        raise NotImplementedError("FBX reader not implemented - would require FBX SDK or similar") 