"""
Utility classes and functions for Open3D reconstruction platform.

This module provides common utilities including timing, logging, monitoring,
file management, data loading, and validation functionality.
"""

import time
import psutil
import os
import sys
from typing import Dict, Any, Optional, List, Union, Iterator, Callable
from pathlib import Path
import threading
from contextlib import contextmanager
import hashlib
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from loguru import logger
import torch
from pydantic import BaseModel, ValidationError


@dataclass
class TimingResult:
    """Result of a timing operation."""
    name: str
    duration: float
    start_time: float
    end_time: float
    metadata: Dict[str, Any]


class Timer:
    """
    High-precision timer for performance monitoring.
    
    Supports context managers, decorators, and manual timing.
    """

    def __init__(self):
        self.timings: Dict[str, List[TimingResult]] = {}
        self._active_timers: Dict[str, float] = {}
        self._lock = threading.Lock()

    @contextmanager
    def measure(self, name: str, **metadata):
        """
        Context manager for timing code blocks.
        
        Args:
            name: Name of the timing operation
            **metadata: Additional metadata to store
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            result = TimingResult(
                name=name,
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                metadata=metadata
            )
            
            with self._lock:
                if name not in self.timings:
                    self.timings[name] = []
                self.timings[name].append(result)

    def start(self, name: str) -> None:
        """Start a named timer."""
        with self._lock:
            self._active_timers[name] = time.perf_counter()

    def stop(self, name: str, **metadata) -> float:
        """
        Stop a named timer and return duration.
        
        Args:
            name: Name of the timer
            **metadata: Additional metadata
            
        Returns:
            Duration in seconds
        """
        end_time = time.perf_counter()
        
        with self._lock:
            if name not in self._active_timers:
                raise ValueError(f"Timer '{name}' was not started")
            
            start_time = self._active_timers.pop(name)
            duration = end_time - start_time
            
            result = TimingResult(
                name=name,
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                metadata=metadata
            )
            
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(result)
            
            return duration

    def get_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a named timer.
        
        Args:
            name: Timer name
            
        Returns:
            Statistics dictionary
        """
        if name not in self.timings:
            return {}
        
        durations = [t.duration for t in self.timings[name]]
        
        return {
            "count": len(durations),
            "total": sum(durations),
            "mean": np.mean(durations),
            "median": np.median(durations),
            "std": np.std(durations),
            "min": min(durations),
            "max": max(durations),
        }

    def get_last_duration(self, name: Optional[str] = None) -> float:
        """Get duration of last timing operation."""
        if name:
            if name in self.timings and self.timings[name]:
                return self.timings[name][-1].duration
            return 0.0
        
        # Return last timing overall
        all_timings = []
        for timing_list in self.timings.values():
            all_timings.extend(timing_list)
        
        if all_timings:
            return max(all_timings, key=lambda t: t.end_time).duration
        return 0.0

    def clear(self, name: Optional[str] = None) -> None:
        """Clear timing history."""
        with self._lock:
            if name:
                self.timings.pop(name, None)
            else:
                self.timings.clear()
                self._active_timers.clear()

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all timings."""
        return {name: self.get_stats(name) for name in self.timings}


class MemoryMonitor:
    """System memory monitoring utility."""

    def __init__(self):
        self.process = psutil.Process()

    def get_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            "process_rss": memory_info.rss / 1024 / 1024,  # MB
            "process_vms": memory_info.vms / 1024 / 1024,  # MB
            "system_total": system_memory.total / 1024 / 1024 / 1024,  # GB
            "system_available": system_memory.available / 1024 / 1024 / 1024,  # GB
            "system_percent": system_memory.percent,
        }

    def get_peak_usage(self) -> float:
        """Get peak memory usage of current process in MB."""
        return self.process.memory_info().peak_wss / 1024 / 1024 if hasattr(self.process.memory_info(), 'peak_wss') else 0.0

    @contextmanager
    def monitor(self, name: str = "operation"):
        """Context manager to monitor memory usage."""
        start_memory = self.get_usage()
        start_time = time.time()
        
        try:
            yield
        finally:
            end_memory = self.get_usage()
            end_time = time.time()
            
            logger.info(f"Memory usage for {name}:")
            logger.info(f"  Duration: {end_time - start_time:.2f}s")
            logger.info(f"  Memory delta: {end_memory['process_rss'] - start_memory['process_rss']:.2f} MB")
            logger.info(f"  Peak memory: {end_memory['process_rss']:.2f} MB")


class GPUMonitor:
    """GPU monitoring utility."""

    def __init__(self):
        self.has_gpu = torch.cuda.is_available()

    def get_usage(self) -> Optional[Dict[str, Any]]:
        """Get current GPU usage."""
        if not self.has_gpu:
            return None
        
        device_count = torch.cuda.device_count()
        gpu_info = {}
        
        for device_id in range(device_count):
            props = torch.cuda.get_device_properties(device_id)
            memory_allocated = torch.cuda.memory_allocated(device_id)
            memory_reserved = torch.cuda.memory_reserved(device_id)
            memory_total = props.total_memory
            
            gpu_info[f"gpu_{device_id}"] = {
                "name": props.name,
                "memory_allocated": memory_allocated / 1024 / 1024,  # MB
                "memory_reserved": memory_reserved / 1024 / 1024,  # MB
                "memory_total": memory_total / 1024 / 1024,  # MB
                "memory_percent": (memory_allocated / memory_total) * 100,
                "compute_capability": f"{props.major}.{props.minor}",
                "multiprocessor_count": props.multi_processor_count,
            }
        
        return gpu_info

    def clear_cache(self) -> None:
        """Clear GPU cache."""
        if self.has_gpu:
            torch.cuda.empty_cache()

    @contextmanager
    def monitor(self, device: Optional[int] = None):
        """Context manager to monitor GPU usage."""
        if not self.has_gpu:
            yield
            return
        
        device = device or torch.cuda.current_device()
        start_memory = torch.cuda.memory_allocated(device)
        
        try:
            yield
        finally:
            end_memory = torch.cuda.memory_allocated(device)
            peak_memory = torch.cuda.max_memory_allocated(device)
            
            logger.info(f"GPU memory usage (device {device}):")
            logger.info(f"  Memory delta: {(end_memory - start_memory) / 1024 / 1024:.2f} MB")
            logger.info(f"  Peak memory: {peak_memory / 1024 / 1024:.2f} MB")
            
            torch.cuda.reset_peak_memory_stats(device)


class FileManager:
    """File and directory management utility."""

    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """Ensure directory exists, create if necessary."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def safe_filename(filename: str) -> str:
        """Create a safe filename by removing/replacing problematic characters."""
        import re
        # Remove or replace problematic characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove leading/trailing spaces and dots
        filename = filename.strip(' .')
        # Ensure it's not too long
        if len(filename) > 255:
            filename = filename[:255]
        return filename

    @staticmethod
    def get_file_hash(file_path: Union[str, Path], algorithm: str = "md5") -> str:
        """Get hash of file contents."""
        file_path = Path(file_path)
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()

    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get detailed file information."""
        file_path = Path(file_path)
        stat = file_path.stat()
        
        return {
            "path": str(file_path),
            "name": file_path.name,
            "size": stat.st_size,
            "size_mb": stat.st_size / 1024 / 1024,
            "created": datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "accessed": datetime.fromtimestamp(stat.st_atime),
            "is_file": file_path.is_file(),
            "is_dir": file_path.is_dir(),
            "suffix": file_path.suffix,
            "stem": file_path.stem,
        }

    @staticmethod
    def list_files(
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = False
    ) -> List[Path]:
        """List files in directory with optional pattern matching."""
        directory = Path(directory)
        
        if recursive:
            return list(directory.rglob(pattern))
        else:
            return list(directory.glob(pattern))

    @staticmethod
    def clean_old_files(
        directory: Union[str, Path],
        max_age_days: int = 7,
        pattern: str = "*",
        dry_run: bool = False
    ) -> List[Path]:
        """Clean old files from directory."""
        directory = Path(directory)
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        
        old_files = []
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                old_files.append(file_path)
                if not dry_run:
                    file_path.unlink()
                    logger.info(f"Deleted old file: {file_path}")
        
        if dry_run:
            logger.info(f"Would delete {len(old_files)} old files")
        else:
            logger.info(f"Deleted {len(old_files)} old files")
        
        return old_files


class DataLoader:
    """Generic data loading utility with caching and validation."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_json(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load JSON file with error handling."""
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            raise
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise

    def save_json(self, data: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """Save data as JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load_pickle(self, file_path: Union[str, Path]) -> Any:
        """Load pickle file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def save_pickle(self, data: Any, file_path: Union[str, Path]) -> None:
        """Save data as pickle file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def load_numpy(self, file_path: Union[str, Path]) -> np.ndarray:
        """Load numpy array."""
        return np.load(file_path)

    def save_numpy(self, data: np.ndarray, file_path: Union[str, Path]) -> None:
        """Save numpy array."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(file_path, data)

    def load_with_cache(
        self,
        key: str,
        loader_func: Callable[[], Any],
        cache_timeout: Optional[float] = None
    ) -> Any:
        """Load data with caching support."""
        if not self.cache_dir:
            return loader_func()
        
        cache_file = self.cache_dir / f"{key}.cache"
        
        # Check if cache exists and is valid
        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            
            if cache_timeout is None or cache_age < cache_timeout:
                try:
                    return self.load_pickle(cache_file)
                except Exception as e:
                    logger.warning(f"Failed to load cache {cache_file}: {e}")
        
        # Load fresh data and cache it
        data = loader_func()
        try:
            self.save_pickle(data, cache_file)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_file}: {e}")
        
        return data


class Validator:
    """Data validation utility using Pydantic models."""

    @staticmethod
    def validate_data(data: Dict[str, Any], model_class: type) -> BaseModel:
        """
        Validate data against Pydantic model.
        
        Args:
            data: Data to validate
            model_class: Pydantic model class
            
        Returns:
            Validated model instance
        """
        try:
            return model_class(**data)
        except ValidationError as e:
            logger.error(f"Validation failed: {e}")
            raise

    @staticmethod
    def validate_file_exists(file_path: Union[str, Path]) -> Path:
        """Validate that file exists."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path

    @staticmethod
    def validate_directory_exists(dir_path: Union[str, Path]) -> Path:
        """Validate that directory exists."""
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise NotADirectoryError(f"Directory not found: {dir_path}")
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {dir_path}")
        return dir_path

    @staticmethod
    def validate_image_format(file_path: Union[str, Path]) -> bool:
        """Validate image file format."""
        file_path = Path(file_path)
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        return file_path.suffix.lower() in valid_extensions

    @staticmethod
    def validate_config_dict(config: Dict[str, Any], required_keys: List[str]) -> bool:
        """Validate configuration dictionary has required keys."""
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        return True

    @staticmethod
    def validate_device(device: str) -> str:
        """Validate PyTorch device specification."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        
        if device in ["cpu", "cuda"]:
            return device
        
        if device.startswith("cuda:"):
            device_id = int(device.split(":")[1])
            if device_id >= torch.cuda.device_count():
                raise ValueError(f"CUDA device {device_id} not available")
            return device
        
        raise ValueError(f"Invalid device specification: {device}")


# Utility functions
def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    return {
        "platform": sys.platform,
        "python_version": sys.version,
        "cpu_count": os.cpu_count(),
        "memory_total": psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "timestamp": datetime.now().isoformat(),
    }


def format_bytes(bytes_value: int) -> str:
    """Format bytes into human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


class ProgressTracker:
    """Simple progress tracking utility."""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()

    def update(self, increment: int = 1) -> None:
        """Update progress."""
        self.current += increment
        self._log_progress()

    def _log_progress(self) -> None:
        """Log current progress."""
        if self.total > 0:
            percent = (self.current / self.total) * 100
            elapsed = time.time() - self.start_time
            
            if self.current > 0:
                eta = (elapsed / self.current) * (self.total - self.current)
                logger.info(f"{self.description}: {percent:.1f}% ({self.current}/{self.total}) - ETA: {format_duration(eta)}")

    def finish(self) -> None:
        """Mark progress as finished."""
        elapsed = time.time() - self.start_time
        logger.info(f"{self.description} completed in {format_duration(elapsed)}") 