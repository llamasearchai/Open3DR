"""
Core Engine for Open3D reconstruction platform.

The Engine class is the main orchestrator that manages all subsystems
including neural rendering, sensor simulation, AI agents, and data processing.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import time
from contextlib import asynccontextmanager

from loguru import logger
import torch
import numpy as np

from .config import Config
from .types import Vector3, Transform, RenderConfig, SensorConfig, SimulationConfig
from .utils import Timer, MemoryMonitor, GPUMonitor
from ..neural_rendering import NeRFReconstructor, GaussianSplattingRenderer
from ..sensors import SensorManager
from ..simulation import SimulationManager
from ..agents import AgentOrchestrator


class Engine:
    """
    Main engine for Open3D reconstruction platform.
    
    This class coordinates all major subsystems and provides a unified
    interface for 3D reconstruction, sensor simulation, and AI automation.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the Open3D Engine.
        
        Args:
            config: Configuration object, defaults to loading from config files
        """
        self.config = config or Config.load_default()
        
        # Core components
        self.timer = Timer()
        self.memory_monitor = MemoryMonitor()
        self.gpu_monitor = GPUMonitor()
        
        # Subsystem managers
        self.sensor_manager: Optional[SensorManager] = None
        self.simulation_manager: Optional[SimulationManager] = None
        self.agent_orchestrator: Optional[AgentOrchestrator] = None
        
        # Active reconstructions and simulations
        self.active_reconstructions: Dict[str, Any] = {}
        self.active_simulations: Dict[str, Any] = {}
        
        # Engine state
        self.is_initialized = False
        self.is_running = False
        
        logger.info("Open3D Engine initialized")

    async def initialize(self) -> None:
        """Initialize all subsystems."""
        if self.is_initialized:
            logger.warning("Engine already initialized")
            return
        
        logger.info("Initializing Open3D Engine subsystems...")
        
        with self.timer.measure("engine_initialization"):
            # Initialize GPU if available
            await self._initialize_gpu()
            
            # Initialize subsystem managers
            self.sensor_manager = SensorManager(self.config)
            self.simulation_manager = SimulationManager(self.config)
            self.agent_orchestrator = AgentOrchestrator(self.config)
            
            # Initialize each subsystem
            await self.sensor_manager.initialize()
            await self.simulation_manager.initialize()
            await self.agent_orchestrator.initialize()
            
            self.is_initialized = True
            
        logger.info(f"Engine initialization completed in {self.timer.get_last_duration():.2f}s")

    async def shutdown(self) -> None:
        """Shutdown all subsystems gracefully."""
        if not self.is_initialized:
            return
        
        logger.info("Shutting down Open3D Engine...")
        
        # Stop all active reconstructions
        for reconstruction_id in list(self.active_reconstructions.keys()):
            await self.stop_reconstruction(reconstruction_id)
        
        # Stop all active simulations
        for simulation_id in list(self.active_simulations.keys()):
            await self.stop_simulation(simulation_id)
        
        # Shutdown subsystems
        if self.agent_orchestrator:
            await self.agent_orchestrator.shutdown()
        if self.simulation_manager:
            await self.simulation_manager.shutdown()
        if self.sensor_manager:
            await self.sensor_manager.shutdown()
        
        self.is_initialized = False
        self.is_running = False
        
        logger.info("Engine shutdown completed")

    @asynccontextmanager
    async def running(self):
        """Context manager for engine lifecycle."""
        await self.initialize()
        self.is_running = True
        try:
            yield self
        finally:
            await self.shutdown()

    async def create_reconstruction(
        self,
        config: RenderConfig,
        input_data: Dict[str, Any],
        reconstruction_id: Optional[str] = None
    ) -> str:
        """
        Create a new 3D reconstruction task.
        
        Args:
            config: Rendering configuration
            input_data: Input data specification
            reconstruction_id: Optional custom ID
            
        Returns:
            Reconstruction ID
        """
        if not self.is_initialized:
            await self.initialize()
        
        reconstruction_id = reconstruction_id or self._generate_id("recon")
        
        logger.info(f"Creating reconstruction {reconstruction_id} with method {config.model_type}")
        
        # Create appropriate reconstructor
        if config.model_type == "gaussian_splatting":
            reconstructor = GaussianSplattingRenderer(config)
        else:
            reconstructor = NeRFReconstructor(config)
        
        # Store reconstruction info
        self.active_reconstructions[reconstruction_id] = {
            "reconstructor": reconstructor,
            "config": config,
            "input_data": input_data,
            "status": "created",
            "created_at": time.time(),
            "progress": 0.0,
            "metrics": {},
        }
        
        return reconstruction_id

    async def start_reconstruction(self, reconstruction_id: str) -> bool:
        """
        Start a reconstruction task.
        
        Args:
            reconstruction_id: ID of reconstruction to start
            
        Returns:
            True if started successfully
        """
        if reconstruction_id not in self.active_reconstructions:
            logger.error(f"Reconstruction {reconstruction_id} not found")
            return False
        
        reconstruction = self.active_reconstructions[reconstruction_id]
        
        if reconstruction["status"] != "created":
            logger.warning(f"Reconstruction {reconstruction_id} already started")
            return False
        
        logger.info(f"Starting reconstruction {reconstruction_id}")
        
        # Update status
        reconstruction["status"] = "running"
        reconstruction["started_at"] = time.time()
        
        # Start reconstruction in background
        asyncio.create_task(self._run_reconstruction(reconstruction_id))
        
        return True

    async def _run_reconstruction(self, reconstruction_id: str) -> None:
        """Run reconstruction in background."""
        reconstruction = self.active_reconstructions[reconstruction_id]
        reconstructor = reconstruction["reconstructor"]
        
        try:
            # Load input data
            await reconstructor.load_data(reconstruction["input_data"])
            
            # Train model with progress tracking
            async for progress in reconstructor.train_async():
                reconstruction["progress"] = progress.get("progress", 0.0)
                reconstruction["metrics"] = progress.get("metrics", {})
                
                # Log progress periodically
                if int(progress.get("iteration", 0)) % 1000 == 0:
                    logger.info(f"Reconstruction {reconstruction_id} progress: {progress['progress']:.1f}%")
            
            # Mark as completed
            reconstruction["status"] = "completed"
            reconstruction["completed_at"] = time.time()
            reconstruction["output_path"] = reconstructor.save_results()
            
            logger.info(f"Reconstruction {reconstruction_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Reconstruction {reconstruction_id} failed: {e}")
            reconstruction["status"] = "failed"
            reconstruction["error"] = str(e)

    async def stop_reconstruction(self, reconstruction_id: str) -> bool:
        """
        Stop a running reconstruction.
        
        Args:
            reconstruction_id: ID of reconstruction to stop
            
        Returns:
            True if stopped successfully
        """
        if reconstruction_id not in self.active_reconstructions:
            return False
        
        reconstruction = self.active_reconstructions[reconstruction_id]
        
        if reconstruction["status"] == "running":
            reconstruction["status"] = "stopped"
            reconstruction["stopped_at"] = time.time()
            
            # Stop the reconstructor
            if "reconstructor" in reconstruction:
                await reconstruction["reconstructor"].stop()
            
            logger.info(f"Reconstruction {reconstruction_id} stopped")
        
        return True

    async def create_simulation(
        self,
        config: SimulationConfig,
        scenario: str,
        sensors: List[SensorConfig],
        simulation_id: Optional[str] = None
    ) -> str:
        """
        Create a new simulation.
        
        Args:
            config: Simulation configuration
            scenario: Scenario name
            sensors: List of sensor configurations
            simulation_id: Optional custom ID
            
        Returns:
            Simulation ID
        """
        if not self.is_initialized:
            await self.initialize()
        
        simulation_id = simulation_id or self._generate_id("sim")
        
        logger.info(f"Creating simulation {simulation_id} with scenario {scenario}")
        
        # Create simulation through manager
        simulation = await self.simulation_manager.create_simulation(
            config, scenario, sensors
        )
        
        self.active_simulations[simulation_id] = {
            "simulation": simulation,
            "config": config,
            "scenario": scenario,
            "sensors": sensors,
            "status": "created",
            "created_at": time.time(),
        }
        
        return simulation_id

    async def start_simulation(self, simulation_id: str) -> bool:
        """Start a simulation."""
        if simulation_id not in self.active_simulations:
            return False
        
        simulation_info = self.active_simulations[simulation_id]
        simulation = simulation_info["simulation"]
        
        await simulation.start()
        simulation_info["status"] = "running"
        simulation_info["started_at"] = time.time()
        
        logger.info(f"Simulation {simulation_id} started")
        return True

    async def stop_simulation(self, simulation_id: str) -> bool:
        """Stop a running simulation."""
        if simulation_id not in self.active_simulations:
            return False
        
        simulation_info = self.active_simulations[simulation_id]
        simulation = simulation_info["simulation"]
        
        await simulation.stop()
        simulation_info["status"] = "stopped"
        simulation_info["stopped_at"] = time.time()
        
        logger.info(f"Simulation {simulation_id} stopped")
        return True

    async def process_agent_command(
        self,
        agent_type: str,
        command: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process a command through AI agents.
        
        Args:
            agent_type: Type of agent to use
            command: Natural language command
            context: Additional context
            
        Returns:
            Agent response
        """
        if not self.agent_orchestrator:
            raise RuntimeError("Agent orchestrator not initialized")
        
        return await self.agent_orchestrator.process_command(
            agent_type, command, context
        )

    def get_status(self) -> Dict[str, Any]:
        """Get engine status and statistics."""
        return {
            "initialized": self.is_initialized,
            "running": self.is_running,
            "active_reconstructions": len(self.active_reconstructions),
            "active_simulations": len(self.active_simulations),
            "memory_usage": self.memory_monitor.get_usage(),
            "gpu_usage": self.gpu_monitor.get_usage() if torch.cuda.is_available() else None,
            "uptime": time.time() - getattr(self, "start_time", time.time()),
        }

    def get_reconstruction_status(self, reconstruction_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific reconstruction."""
        if reconstruction_id not in self.active_reconstructions:
            return None
        
        reconstruction = self.active_reconstructions[reconstruction_id]
        return {
            "reconstruction_id": reconstruction_id,
            "status": reconstruction["status"],
            "progress": reconstruction.get("progress", 0.0),
            "metrics": reconstruction.get("metrics", {}),
            "created_at": reconstruction["created_at"],
            "config": reconstruction["config"].dict(),
        }

    def get_simulation_status(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific simulation."""
        if simulation_id not in self.active_simulations:
            return None
        
        simulation_info = self.active_simulations[simulation_id]
        return {
            "simulation_id": simulation_id,
            "status": simulation_info["status"],
            "scenario": simulation_info["scenario"],
            "created_at": simulation_info["created_at"],
            "config": simulation_info["config"].dict(),
        }

    async def _initialize_gpu(self) -> None:
        """Initialize GPU support if available."""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            logger.info(f"GPU support available: {device_count} device(s)")
            logger.info(f"Current device: {current_device} ({device_name})")
            
            # Warm up GPU
            torch.cuda.empty_cache()
            _ = torch.zeros(1).cuda()
            
        else:
            logger.warning("No GPU support available, using CPU")

    def _generate_id(self, prefix: str = "task") -> str:
        """Generate unique ID for tasks."""
        import uuid
        return f"{prefix}_{int(time.time())}_{str(uuid.uuid4())[:8]}"

    def __repr__(self) -> str:
        return f"Engine(initialized={self.is_initialized}, running={self.is_running})" 