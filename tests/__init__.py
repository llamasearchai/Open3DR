"""
Test suite for Open3D reconstruction platform.

This package contains comprehensive tests for all modules including:
- Unit tests for core functionality
- Integration tests for API endpoints
- Performance benchmarks
- GPU-accelerated tests
- Mock data generators
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_DATA_DIR = project_root / "tests" / "data"
TEST_OUTPUT_DIR = project_root / "tests" / "output"
MOCK_API_KEY = "test-api-key-12345"

# Create test directories if they don't exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

# Test environment variables
os.environ.setdefault("OPENAI_API_KEY", MOCK_API_KEY)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")  # Test DB
os.environ.setdefault("TEST_MODE", "true")

__all__ = [
    "TEST_DATA_DIR",
    "TEST_OUTPUT_DIR", 
    "MOCK_API_KEY",
] 