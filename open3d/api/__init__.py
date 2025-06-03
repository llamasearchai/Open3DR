"""
FastAPI application for Open3D reconstruction platform.

This module provides REST API endpoints for neural rendering, sensor simulation,
and AI agent integration.
"""

from .app import create_app, get_app
from .routers import (
    reconstruction_router,
    simulation_router,
    agents_router,
    sensors_router,
    health_router,
)
from .middleware import setup_middleware
from .dependencies import get_database, get_redis, get_openai_client
from .models import (
    ReconstructionRequest,
    ReconstructionResponse,
    SimulationRequest,
    SimulationResponse,
    AgentRequest,
    AgentResponse,
)

__all__ = [
    "create_app",
    "get_app",
    # Routers
    "reconstruction_router",
    "simulation_router", 
    "agents_router",
    "sensors_router",
    "health_router",
    # Middleware
    "setup_middleware",
    # Dependencies
    "get_database",
    "get_redis",
    "get_openai_client",
    # Models
    "ReconstructionRequest",
    "ReconstructionResponse",
    "SimulationRequest",
    "SimulationResponse",
    "AgentRequest",
    "AgentResponse",
] 