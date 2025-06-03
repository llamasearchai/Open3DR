"""
Data loaders for efficient data loading and processing.

This module provides various data loading utilities for handling different
data types, formats, and sources in an efficient manner.
"""

import asyncio
from typing import Dict, List, Optional, Union, Any, AsyncIterator, Iterator
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
from loguru import logger
import torch
from torch.utils.data import DataLoader as TorchDataLoader


class DataLoader:
    """
    Generic data loader for various data types.
    
    Provides utilities for loading files, caching, and batch processing.
    """

    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self.cache: Dict[str, Any] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)

    def load_file(self, file_path: Union[str, Path]) -> Any:
        """
        Load a file with caching.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Loaded data
        """
        file_path = Path(file_path)
        cache_key = str(file_path)
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Load file
        try:
            if file_path.suffix.lower() in ['.json']:
                import json
                with open(file_path, 'r') as f:
                    data = json.load(f)
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                import yaml
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f)
            elif file_path.suffix.lower() in ['.txt']:
                with open(file_path, 'r') as f:
                    data = f.read()
            elif file_path.suffix.lower() in ['.npy']:
                data = np.load(file_path)
            elif file_path.suffix.lower() in ['.npz']:
                data = np.load(file_path)
            else:
                # Binary load
                with open(file_path, 'rb') as f:
                    data = f.read()
            
            # Cache result
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = data
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            raise

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self.cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "cached_items": list(self.cache.keys())
        }


class AsyncDataLoader:
    """
    Asynchronous data loader for non-blocking data loading.
    
    Provides async methods for loading data without blocking the main thread.
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def load_file_async(self, file_path: Union[str, Path]) -> Any:
        """
        Asynchronously load a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Loaded data
        """
        loop = asyncio.get_event_loop()
        
        def _load_file():
            loader = DataLoader()
            return loader.load_file(file_path)
        
        return await loop.run_in_executor(self.executor, _load_file)

    async def load_files_async(
        self,
        file_paths: List[Union[str, Path]]
    ) -> List[Any]:
        """
        Load multiple files asynchronously.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            List of loaded data
        """
        tasks = [self.load_file_async(path) for path in file_paths]
        return await asyncio.gather(*tasks)

    async def load_directory_async(
        self,
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = False
    ) -> Dict[str, Any]:
        """
        Load all files in a directory asynchronously.
        
        Args:
            directory: Directory path
            pattern: File pattern to match
            recursive: Whether to search recursively
            
        Returns:
            Dictionary mapping file paths to loaded data
        """
        directory = Path(directory)
        
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))
        
        # Filter out directories
        files = [f for f in files if f.is_file()]
        
        # Load files
        data_list = await self.load_files_async(files)
        
        # Create mapping
        result = {}
        for file_path, data in zip(files, data_list):
            result[str(file_path)] = data
        
        return result


class BatchDataLoader:
    """
    Batch data loader for processing data in batches.
    
    Provides efficient batch processing capabilities for large datasets.
    """

    def __init__(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = False
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last

    def create_torch_loader(
        self,
        dataset: torch.utils.data.Dataset,
        **kwargs
    ) -> TorchDataLoader:
        """
        Create a PyTorch DataLoader.
        
        Args:
            dataset: PyTorch dataset
            **kwargs: Additional DataLoader arguments
            
        Returns:
            PyTorch DataLoader
        """
        loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'num_workers': self.num_workers,
            'drop_last': self.drop_last,
            **kwargs
        }
        
        return TorchDataLoader(dataset, **loader_kwargs)

    def batch_iterator(
        self,
        data: Union[List[Any], np.ndarray],
        transform_fn: Optional[callable] = None
    ) -> Iterator[Any]:
        """
        Create a batch iterator for data.
        
        Args:
            data: Input data
            transform_fn: Optional transform function
            
        Yields:
            Batches of data
        """
        if isinstance(data, np.ndarray):
            data = data.tolist()
        
        # Shuffle if requested
        if self.shuffle:
            indices = np.random.permutation(len(data))
            data = [data[i] for i in indices]
        
        # Create batches
        for i in range(0, len(data), self.batch_size):
            if self.drop_last and i + self.batch_size > len(data):
                break
            
            batch = data[i:i + self.batch_size]
            
            # Apply transform if provided
            if transform_fn:
                batch = [transform_fn(item) for item in batch]
            
            yield batch

    async def async_batch_iterator(
        self,
        data: Union[List[Any], np.ndarray],
        transform_fn: Optional[callable] = None
    ) -> AsyncIterator[Any]:
        """
        Create an async batch iterator for data.
        
        Args:
            data: Input data
            transform_fn: Optional transform function
            
        Yields:
            Batches of data
        """
        for batch in self.batch_iterator(data, transform_fn):
            yield batch
            # Allow other coroutines to run
            await asyncio.sleep(0.001)


class StreamingDataLoader:
    """
    Streaming data loader for handling large datasets that don't fit in memory.
    
    Loads data on-demand from disk or network sources.
    """

    def __init__(
        self,
        data_source: Union[str, Path, callable],
        buffer_size: int = 100,
        prefetch_factor: int = 2
    ):
        self.data_source = data_source
        self.buffer_size = buffer_size
        self.prefetch_factor = prefetch_factor
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.prefetch_thread = None
        self.stop_prefetch = False

    def start_prefetch(self) -> None:
        """Start prefetching data in background."""
        if self.prefetch_thread is None or not self.prefetch_thread.is_alive():
            self.stop_prefetch = False
            self.prefetch_thread = threading.Thread(target=self._prefetch_worker)
            self.prefetch_thread.start()

    def stop_prefetch_thread(self) -> None:
        """Stop prefetching thread."""
        self.stop_prefetch = True
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join()

    def _prefetch_worker(self) -> None:
        """Background worker for prefetching data."""
        while not self.stop_prefetch:
            with self.buffer_lock:
                if len(self.buffer) < self.buffer_size * self.prefetch_factor:
                    try:
                        # Load next data item
                        if callable(self.data_source):
                            data = self.data_source()
                        else:
                            # Load from file or directory
                            data = self._load_next_item()
                        
                        if data is not None:
                            self.buffer.append(data)
                    except Exception as e:
                        logger.error(f"Error in prefetch worker: {e}")
                        break
            
            time.sleep(0.01)  # Small delay to prevent busy waiting

    def _load_next_item(self) -> Optional[Any]:
        """Load next item from data source."""
        # Placeholder implementation
        # In practice, this would implement specific loading logic
        # based on the data source type (files, database, API, etc.)
        return None

    def get_batch(self, batch_size: int) -> List[Any]:
        """
        Get a batch of data.
        
        Args:
            batch_size: Size of the batch
            
        Returns:
            Batch of data
        """
        batch = []
        
        with self.buffer_lock:
            for _ in range(min(batch_size, len(self.buffer))):
                if self.buffer:
                    batch.append(self.buffer.pop(0))
        
        return batch

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_prefetch_thread()


class CachedDataLoader:
    """
    Data loader with intelligent caching and memory management.
    
    Automatically manages cache based on memory usage and access patterns.
    """

    def __init__(
        self,
        max_memory_mb: int = 1000,
        cache_strategy: str = "lru"  # lru, lfu, random
    ):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache_strategy = cache_strategy
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}
        self.memory_usage = 0

    def get_item_size(self, item: Any) -> int:
        """Estimate memory size of an item."""
        if isinstance(item, np.ndarray):
            return item.nbytes
        elif isinstance(item, torch.Tensor):
            return item.numel() * item.element_size()
        elif isinstance(item, (list, tuple)):
            return sum(self.get_item_size(x) for x in item)
        elif isinstance(item, dict):
            return sum(self.get_item_size(k) + self.get_item_size(v) 
                      for k, v in item.items())
        elif isinstance(item, str):
            return len(item.encode('utf-8'))
        else:
            # Rough estimate
            return 1000

    def _evict_items(self, required_space: int) -> None:
        """Evict items from cache to free up space."""
        if self.cache_strategy == "lru":
            # Remove least recently used items
            sorted_items = sorted(
                self.access_times.items(),
                key=lambda x: x[1]
            )
        elif self.cache_strategy == "lfu":
            # Remove least frequently used items
            sorted_items = sorted(
                self.access_counts.items(),
                key=lambda x: x[1]
            )
        else:  # random
            import random
            sorted_items = list(self.cache.keys())
            random.shuffle(sorted_items)
            sorted_items = [(k, 0) for k in sorted_items]
        
        freed_space = 0
        for key, _ in sorted_items:
            if freed_space >= required_space:
                break
            
            if key in self.cache:
                item_size = self.get_item_size(self.cache[key])
                del self.cache[key]
                del self.access_times[key]
                del self.access_counts[key]
                self.memory_usage -= item_size
                freed_space += item_size

    def load_and_cache(self, key: str, loader_fn: callable) -> Any:
        """
        Load item and cache it.
        
        Args:
            key: Cache key
            loader_fn: Function to load the item
            
        Returns:
            Loaded item
        """
        # Check if already cached
        if key in self.cache:
            self.access_times[key] = time.time()
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            return self.cache[key]
        
        # Load item
        item = loader_fn()
        item_size = self.get_item_size(item)
        
        # Check if we need to evict items
        if self.memory_usage + item_size > self.max_memory_bytes:
            self._evict_items(item_size)
        
        # Cache item if it fits
        if item_size <= self.max_memory_bytes:
            self.cache[key] = item
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
            self.memory_usage += item_size
        
        return item

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache),
            "memory_usage_mb": self.memory_usage / (1024 * 1024),
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
            "cache_strategy": self.cache_strategy,
            "hit_rate": self._calculate_hit_rate()
        }

    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if not self.access_counts:
            return 0.0
        
        total_accesses = sum(self.access_counts.values())
        cache_hits = sum(1 for count in self.access_counts.values() if count > 1)
        
        return cache_hits / total_accesses if total_accesses > 0 else 0.0 