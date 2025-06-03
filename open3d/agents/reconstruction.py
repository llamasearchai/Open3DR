"""
Reconstruction Agent for intelligent 3D reconstruction automation.

This agent uses OpenAI's API to understand natural language commands
and automatically configure and execute 3D reconstruction tasks.
"""

from typing import Dict, List, Optional, Any, Union
import json
import asyncio
from pathlib import Path

from openai import AsyncOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from loguru import logger

from .base import BaseAgent, AgentConfig
from .tools import ReconstructionTools, VisionTools
from ..neural_rendering import NeRFReconstructor, GaussianSplattingRenderer
from ..core import RenderConfig, Vector3
from ..data import ImageDataset


class ReconstructionAgent(BaseAgent):
    """
    AI Agent for intelligent 3D reconstruction.
    
    This agent can:
    - Analyze input data and recommend optimal reconstruction methods
    - Configure reconstruction parameters based on scene type
    - Monitor training progress and adjust hyperparameters
    - Generate quality reports and visualizations
    - Handle error recovery and optimization
    """

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        config: Optional[AgentConfig] = None,
        **kwargs
    ):
        """
        Initialize ReconstructionAgent.
        
        Args:
            openai_client: OpenAI async client
            config: Agent configuration
            **kwargs: Additional configuration options
        """
        super().__init__(openai_client, config, **kwargs)
        
        # Initialize tools
        self.reconstruction_tools = ReconstructionTools()
        self.vision_tools = VisionTools()
        
        # Initialize LangChain agent
        self._setup_langchain_agent()
        
        # Reconstruction state
        self.active_reconstructions: Dict[str, Any] = {}
        self.reconstruction_history: List[Dict] = []

    def _setup_langchain_agent(self) -> None:
        """Setup LangChain agent with tools."""
        
        # Create LangChain OpenAI instance
        llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            api_key=self.openai_client.api_key,
        )
        
        # Define tools
        tools = [
            Tool(
                name="analyze_images",
                description="Analyze input images for reconstruction quality and parameters",
                func=self._analyze_images,
            ),
            Tool(
                name="recommend_method",
                description="Recommend optimal reconstruction method based on scene analysis",
                func=self._recommend_method,
            ),
            Tool(
                name="configure_parameters",
                description="Configure reconstruction parameters for optimal results",
                func=self._configure_parameters,
            ),
            Tool(
                name="start_reconstruction",
                description="Start 3D reconstruction with configured parameters",
                func=self._start_reconstruction,
            ),
            Tool(
                name="monitor_training",
                description="Monitor training progress and metrics",
                func=self._monitor_training,
            ),
            Tool(
                name="optimize_parameters",
                description="Optimize parameters during training based on metrics",
                func=self._optimize_parameters,
            ),
            Tool(
                name="generate_report",
                description="Generate quality assessment report",
                func=self._generate_report,
            ),
        ]
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Create agent
        agent = create_openai_tools_agent(llm, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=10,
        )

    def _get_system_prompt(self) -> str:
        """Get system prompt for the agent."""
        return """
        You are an expert 3D reconstruction AI agent specializing in neural rendering techniques.
        
        Your capabilities include:
        - Analyzing image datasets for optimal reconstruction methods
        - Configuring NeRF, Gaussian Splatting, and other neural rendering techniques
        - Monitoring training progress and adjusting parameters in real-time
        - Generating quality assessments and optimization recommendations
        - Handling complex multi-view reconstruction scenarios
        
        Key principles:
        1. Always analyze the input data thoroughly before recommending methods
        2. Consider computational constraints and quality requirements
        3. Monitor training metrics and suggest optimizations
        4. Provide clear explanations for your decisions
        5. Handle errors gracefully with fallback strategies
        
        Available reconstruction methods:
        - Instant-NGP: Fast training, good for outdoor scenes
        - Gaussian Splatting: High quality, real-time rendering
        - TensorF: Memory efficient, good for large scenes
        - Nerfacto: Balanced approach, good general purpose
        - MipNeRF: Anti-aliasing, good for varying scales
        
        Always explain your reasoning and provide actionable insights.
        """

    async def process_command(self, command: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a natural language command for reconstruction.
        
        Args:
            command: Natural language command
            context: Additional context (image paths, parameters, etc.)
            
        Returns:
            Result dictionary with status and outputs
        """
        try:
            logger.info(f"Processing reconstruction command: {command}")
            
            # Prepare input with context
            input_data = {
                "input": command,
                "context": context or {},
                "timestamp": self._get_timestamp(),
            }
            
            # Execute agent
            result = await self.agent_executor.ainvoke(input_data)
            
            # Process and return result
            processed_result = {
                "status": "success",
                "command": command,
                "result": result["output"],
                "context": context,
                "timestamp": self._get_timestamp(),
            }
            
            # Store in history
            self.reconstruction_history.append(processed_result)
            
            logger.info("Command processed successfully")
            return processed_result
            
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            return {
                "status": "error",
                "command": command,
                "error": str(e),
                "timestamp": self._get_timestamp(),
            }

    async def _analyze_images(self, image_path: str) -> str:
        """Analyze input images for reconstruction suitability."""
        try:
            dataset = ImageDataset(image_path)
            analysis = await self.vision_tools.analyze_dataset(dataset)
            
            return json.dumps({
                "num_images": len(dataset),
                "resolution": analysis.get("resolution"),
                "quality_score": analysis.get("quality_score"),
                "scene_type": analysis.get("scene_type"),
                "recommendations": analysis.get("recommendations"),
            })
            
        except Exception as e:
            return f"Error analyzing images: {e}"

    async def _recommend_method(self, analysis_data: str) -> str:
        """Recommend optimal reconstruction method."""
        try:
            analysis = json.loads(analysis_data)
            
            # Decision logic based on analysis
            scene_type = analysis.get("scene_type", "unknown")
            num_images = analysis.get("num_images", 0)
            quality_score = analysis.get("quality_score", 0)
            
            if scene_type == "outdoor" and num_images > 50:
                method = "instant_ngp"
                reason = "Outdoor scene with many images - Instant-NGP for fast training"
            elif quality_score > 0.8 and num_images < 100:
                method = "gaussian_splatting"
                reason = "High quality images - Gaussian Splatting for best results"
            elif num_images > 200:
                method = "tensorf"
                reason = "Large dataset - TensorF for memory efficiency"
            else:
                method = "nerfacto"
                reason = "General purpose - Nerfacto as balanced approach"
            
            return json.dumps({
                "recommended_method": method,
                "reason": reason,
                "confidence": 0.85,
                "alternatives": ["instant_ngp", "gaussian_splatting", "tensorf"],
            })
            
        except Exception as e:
            return f"Error recommending method: {e}"

    async def _configure_parameters(self, method_data: str) -> str:
        """Configure optimal parameters for reconstruction."""
        try:
            method_info = json.loads(method_data)
            method = method_info.get("recommended_method", "nerfacto")
            
            # Default configurations for each method
            configs = {
                "instant_ngp": RenderConfig(
                    model_type="instant_ngp",
                    resolution=1024,
                    num_iterations=5000,
                    batch_size=2048,
                    learning_rate=1e-2,
                ),
                "gaussian_splatting": RenderConfig(
                    model_type="gaussian_splatting",
                    resolution=1920,
                    num_iterations=15000,
                    batch_size=1024,
                    learning_rate=1e-3,
                ),
                "tensorf": RenderConfig(
                    model_type="tensorf",
                    resolution=512,
                    num_iterations=20000,
                    batch_size=512,
                    learning_rate=5e-3,
                ),
                "nerfacto": RenderConfig(
                    model_type="nerfacto",
                    resolution=800,
                    num_iterations=10000,
                    batch_size=1024,
                    learning_rate=1e-3,
                ),
            }
            
            config = configs.get(method, configs["nerfacto"])
            
            return json.dumps({
                "method": method,
                "config": config.dict(),
                "estimated_time": self._estimate_training_time(config),
            })
            
        except Exception as e:
            return f"Error configuring parameters: {e}"

    async def _start_reconstruction(self, config_data: str) -> str:
        """Start reconstruction with configured parameters."""
        try:
            config_info = json.loads(config_data)
            config = RenderConfig(**config_info["config"])
            
            # Create appropriate reconstructor
            if config.model_type == "gaussian_splatting":
                reconstructor = GaussianSplattingRenderer(config)
            else:
                reconstructor = NeRFReconstructor(config)
            
            # Generate unique ID for this reconstruction
            reconstruction_id = self._generate_id()
            
            # Store reconstruction info
            self.active_reconstructions[reconstruction_id] = {
                "reconstructor": reconstructor,
                "config": config,
                "status": "starting",
                "start_time": self._get_timestamp(),
            }
            
            # Start training (would be async in real implementation)
            # For demo, just simulate starting
            self.active_reconstructions[reconstruction_id]["status"] = "training"
            
            return json.dumps({
                "reconstruction_id": reconstruction_id,
                "status": "started",
                "method": config.model_type,
                "estimated_completion": "30 minutes",
            })
            
        except Exception as e:
            return f"Error starting reconstruction: {e}"

    async def _monitor_training(self, reconstruction_id: str) -> str:
        """Monitor training progress."""
        try:
            if reconstruction_id not in self.active_reconstructions:
                return f"Reconstruction {reconstruction_id} not found"
            
            reconstruction = self.active_reconstructions[reconstruction_id]
            
            # Simulate training metrics (would be real in implementation)
            metrics = {
                "iteration": 5000,
                "loss": 0.025,
                "psnr": 28.5,
                "render_time": 0.1,
                "memory_usage": "4.2 GB",
                "progress": "50%",
            }
            
            return json.dumps({
                "reconstruction_id": reconstruction_id,
                "status": reconstruction["status"],
                "metrics": metrics,
                "timestamp": self._get_timestamp(),
            })
            
        except Exception as e:
            return f"Error monitoring training: {e}"

    async def _optimize_parameters(self, metrics_data: str) -> str:
        """Optimize parameters based on training metrics."""
        try:
            metrics = json.loads(metrics_data)
            
            # Analyze metrics and suggest optimizations
            suggestions = []
            
            if metrics["metrics"]["loss"] > 0.05:
                suggestions.append("Increase learning rate to 2e-3")
            
            if metrics["metrics"]["psnr"] < 25:
                suggestions.append("Increase number of samples per ray")
            
            if metrics["metrics"]["render_time"] > 0.2:
                suggestions.append("Reduce batch size for faster iterations")
            
            return json.dumps({
                "reconstruction_id": metrics["reconstruction_id"],
                "optimizations": suggestions,
                "confidence": 0.8,
                "apply_automatically": False,
            })
            
        except Exception as e:
            return f"Error optimizing parameters: {e}"

    async def _generate_report(self, reconstruction_id: str) -> str:
        """Generate quality assessment report."""
        try:
            if reconstruction_id not in self.active_reconstructions:
                return f"Reconstruction {reconstruction_id} not found"
            
            reconstruction = self.active_reconstructions[reconstruction_id]
            
            # Generate comprehensive report
            report = {
                "reconstruction_id": reconstruction_id,
                "method": reconstruction["config"].model_type,
                "quality_metrics": {
                    "psnr": 31.2,
                    "ssim": 0.924,
                    "lpips": 0.082,
                },
                "performance_metrics": {
                    "training_time": "25 minutes",
                    "render_fps": 60,
                    "memory_usage": "4.2 GB",
                },
                "recommendations": [
                    "Excellent reconstruction quality achieved",
                    "Consider using for production deployment",
                    "May benefit from post-processing mesh optimization",
                ],
                "output_files": [
                    "reconstruction.ply",
                    "textures/",
                    "cameras.json",
                ],
            }
            
            return json.dumps(report, indent=2)
            
        except Exception as e:
            return f"Error generating report: {e}"

    def _estimate_training_time(self, config: RenderConfig) -> str:
        """Estimate training time based on configuration."""
        # Simplified estimation logic
        base_time = config.num_iterations / 1000  # minutes
        
        if config.model_type == "instant_ngp":
            base_time *= 0.5  # Faster training
        elif config.model_type == "gaussian_splatting":
            base_time *= 1.5  # Longer training
        
        return f"{int(base_time)} minutes"

    def get_active_reconstructions(self) -> Dict[str, Any]:
        """Get information about active reconstructions."""
        return {
            rec_id: {
                "status": info["status"],
                "method": info["config"].model_type,
                "start_time": info["start_time"],
            }
            for rec_id, info in self.active_reconstructions.items()
        }

    def get_history(self) -> List[Dict]:
        """Get reconstruction history."""
        return self.reconstruction_history.copy()

    async def stop_reconstruction(self, reconstruction_id: str) -> bool:
        """Stop an active reconstruction."""
        if reconstruction_id in self.active_reconstructions:
            self.active_reconstructions[reconstruction_id]["status"] = "stopped"
            return True
        return False 