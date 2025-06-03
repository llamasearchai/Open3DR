"""
Reconstruction API endpoints for neural rendering and 3D reconstruction.

This module provides REST endpoints for managing reconstruction jobs,
training neural networks, and rendering novel views.
"""

from typing import Dict, List, Optional, Any, Union
import asyncio
from pathlib import Path
import uuid
import json

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form, Query, Depends
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger
import torch

from ...core.types import RenderConfig, DeviceType, RenderingBackend
from ...core.engine import Engine
from ...neural_rendering import NeRFReconstructor, GaussianSplattingRenderer, TrainingProgress
from ...data import ImageDataset

router = APIRouter(prefix="/reconstruction", tags=["reconstruction"])

# Global reconstruction jobs storage
reconstruction_jobs: Dict[str, Dict[str, Any]] = {}
active_engines: Dict[str, Engine] = {}


# Pydantic models for API
class ReconstructionConfig(BaseModel):
    """Configuration for reconstruction job."""
    method: str = Field(..., description="Reconstruction method (nerf, gaussian_splatting, instant_ngp)")
    resolution: int = Field(512, description="Rendering resolution")
    num_iterations: int = Field(10000, description="Number of training iterations")
    batch_size: int = Field(1024, description="Training batch size")
    learning_rate: float = Field(0.0005, description="Learning rate")
    device: DeviceType = Field(DeviceType.AUTO, description="Computing device")
    num_samples: int = Field(64, description="Number of samples per ray")
    use_mixed_precision: bool = Field(True, description="Use mixed precision training")
    
    class Config:
        use_enum_values = True


class ReconstructionJob(BaseModel):
    """Reconstruction job information."""
    id: str
    status: str
    method: str
    config: ReconstructionConfig
    progress: float = 0.0
    current_iteration: int = 0
    total_iterations: int
    loss: Optional[float] = None
    metrics: Dict[str, float] = {}
    created_at: str
    updated_at: str
    output_path: Optional[str] = None
    error_message: Optional[str] = None


class RenderRequest(BaseModel):
    """Request for rendering a view."""
    camera_position: List[float] = Field(..., description="Camera position [x, y, z]")
    camera_rotation: List[float] = Field([0, 0, 0], description="Camera rotation [rx, ry, rz]")
    resolution: Optional[List[int]] = Field(None, description="Render resolution [width, height]")
    output_format: str = Field("png", description="Output format (png, jpg)")


class DataUploadResponse(BaseModel):
    """Response for data upload."""
    dataset_id: str
    message: str
    num_images: int
    format: str


@router.post("/jobs", response_model=ReconstructionJob)
async def create_reconstruction_job(
    config: ReconstructionConfig,
    dataset_id: str = Query(..., description="Dataset ID to use for reconstruction"),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> ReconstructionJob:
    """
    Create a new reconstruction job.
    
    This endpoint creates a new 3D reconstruction job using the specified
    method and configuration. The job runs in the background.
    """
    job_id = str(uuid.uuid4())
    
    # Validate dataset exists
    # In practice, would check if dataset exists in storage
    
    # Create job
    job = ReconstructionJob(
        id=job_id,
        status="created",
        method=config.method,
        config=config,
        total_iterations=config.num_iterations,
        created_at=datetime.utcnow().isoformat(),
        updated_at=datetime.utcnow().isoformat()
    )
    
    reconstruction_jobs[job_id] = job.dict()
    
    # Start reconstruction in background
    background_tasks.add_task(run_reconstruction, job_id, dataset_id, config)
    
    logger.info(f"Created reconstruction job {job_id} with method {config.method}")
    
    return job


@router.get("/jobs", response_model=List[ReconstructionJob])
async def list_reconstruction_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    method: Optional[str] = Query(None, description="Filter by method")
) -> List[ReconstructionJob]:
    """
    List all reconstruction jobs.
    
    Returns a list of all reconstruction jobs, optionally filtered by
    status or method.
    """
    jobs = []
    
    for job_data in reconstruction_jobs.values():
        # Apply filters
        if status and job_data["status"] != status:
            continue
        if method and job_data["method"] != method:
            continue
            
        jobs.append(ReconstructionJob(**job_data))
    
    return jobs


@router.get("/jobs/{job_id}", response_model=ReconstructionJob)
async def get_reconstruction_job(job_id: str) -> ReconstructionJob:
    """
    Get reconstruction job details.
    
    Returns detailed information about a specific reconstruction job
    including current progress and metrics.
    """
    if job_id not in reconstruction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = reconstruction_jobs[job_id]
    return ReconstructionJob(**job_data)


@router.delete("/jobs/{job_id}")
async def delete_reconstruction_job(job_id: str) -> Dict[str, str]:
    """
    Delete a reconstruction job.
    
    Stops the job if it's running and removes it from the system.
    """
    if job_id not in reconstruction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Stop job if running
    if job_id in active_engines:
        engine = active_engines[job_id]
        await engine.stop()
        del active_engines[job_id]
    
    # Remove job
    del reconstruction_jobs[job_id]
    
    logger.info(f"Deleted reconstruction job {job_id}")
    
    return {"message": f"Job {job_id} deleted successfully"}


@router.post("/jobs/{job_id}/stop")
async def stop_reconstruction_job(job_id: str) -> Dict[str, str]:
    """
    Stop a running reconstruction job.
    
    Gracefully stops the training process and updates the job status.
    """
    if job_id not in reconstruction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_id in active_engines:
        engine = active_engines[job_id]
        await engine.stop()
        
        # Update job status
        reconstruction_jobs[job_id]["status"] = "stopped"
        reconstruction_jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Stopped reconstruction job {job_id}")
        
        return {"message": f"Job {job_id} stopped successfully"}
    else:
        return {"message": f"Job {job_id} is not running"}


@router.get("/jobs/{job_id}/progress")
async def get_job_progress(job_id: str) -> Dict[str, Any]:
    """
    Get real-time progress of a reconstruction job.
    
    Returns current training progress including loss, metrics, and ETA.
    """
    if job_id not in reconstruction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = reconstruction_jobs[job_id]
    
    progress_info = {
        "job_id": job_id,
        "status": job_data["status"],
        "progress": job_data["progress"],
        "current_iteration": job_data["current_iteration"],
        "total_iterations": job_data["total_iterations"],
        "loss": job_data.get("loss"),
        "metrics": job_data.get("metrics", {}),
        "updated_at": job_data["updated_at"]
    }
    
    # Add ETA if job is running
    if job_data["status"] == "running" and job_data["current_iteration"] > 0:
        elapsed_time = (datetime.utcnow() - datetime.fromisoformat(job_data["created_at"])).total_seconds()
        iterations_per_second = job_data["current_iteration"] / elapsed_time
        remaining_iterations = job_data["total_iterations"] - job_data["current_iteration"]
        eta_seconds = remaining_iterations / iterations_per_second if iterations_per_second > 0 else 0
        progress_info["eta_seconds"] = eta_seconds
    
    return progress_info


@router.post("/jobs/{job_id}/render")
async def render_view(
    job_id: str,
    render_request: RenderRequest
) -> FileResponse:
    """
    Render a novel view from a trained model.
    
    Uses the trained neural rendering model to generate a new view
    from the specified camera pose.
    """
    if job_id not in reconstruction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = reconstruction_jobs[job_id]
    
    if job_data["status"] not in ["completed", "training"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot render from job with status: {job_data['status']}"
        )
    
    if job_id not in active_engines:
        raise HTTPException(status_code=400, detail="No active engine for this job")
    
    engine = active_engines[job_id]
    
    try:
        # Create camera from request
        from ...core.types import Camera, Vector3
        
        camera = Camera(
            width=render_request.resolution[0] if render_request.resolution else 512,
            height=render_request.resolution[1] if render_request.resolution else 512,
            fx=256.0,  # Simplified intrinsics
            fy=256.0,
            cx=256.0,
            cy=256.0,
            position=Vector3(*render_request.camera_position),
            rotation=Vector3(*render_request.camera_rotation)
        )
        
        # Render view
        result = await engine.render_view(camera)
        
        # Save rendered image
        output_dir = Path("output") / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        render_id = str(uuid.uuid4())
        output_path = output_dir / f"render_{render_id}.{render_request.output_format}"
        
        # Save image (simplified)
        import cv2
        import numpy as np
        
        image = (result.image * 255).astype(np.uint8)
        if render_request.output_format.lower() == "jpg":
            cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGRA))
        
        return FileResponse(
            path=output_path,
            filename=f"render_{render_id}.{render_request.output_format}",
            media_type=f"image/{render_request.output_format}"
        )
        
    except Exception as e:
        logger.error(f"Failed to render view for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Rendering failed: {str(e)}")


@router.get("/jobs/{job_id}/download")
async def download_results(job_id: str) -> FileResponse:
    """
    Download trained model and results.
    
    Returns a ZIP file containing the trained model, configuration,
    and sample renders.
    """
    if job_id not in reconstruction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = reconstruction_jobs[job_id]
    
    if job_data["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot download from job with status: {job_data['status']}"
        )
    
    output_path = job_data.get("output_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Results not found")
    
    # Create ZIP file (simplified)
    import zipfile
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
        zip_path = tmp_file.name
    
    with zipfile.ZipFile(zip_path, 'w') as zip_file:
        result_dir = Path(output_path)
        for file_path in result_dir.rglob("*"):
            if file_path.is_file():
                zip_file.write(file_path, file_path.relative_to(result_dir))
    
    return FileResponse(
        path=zip_path,
        filename=f"reconstruction_{job_id}.zip",
        media_type="application/zip"
    )


@router.post("/upload-dataset", response_model=DataUploadResponse)
async def upload_dataset(
    files: List[UploadFile] = File(...),
    dataset_name: str = Form(...),
    format: str = Form("auto")
) -> DataUploadResponse:
    """
    Upload dataset for reconstruction.
    
    Accepts multiple image files and optional camera poses/intrinsics
    to create a dataset for neural rendering training.
    """
    dataset_id = str(uuid.uuid4())
    dataset_dir = Path("datasets") / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded files
    image_count = 0
    for file in files:
        if file.content_type and file.content_type.startswith("image/"):
            file_path = dataset_dir / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            image_count += 1
    
    # Create dataset metadata
    metadata = {
        "id": dataset_id,
        "name": dataset_name,
        "format": format,
        "num_images": image_count,
        "created_at": datetime.utcnow().isoformat()
    }
    
    with open(dataset_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Uploaded dataset {dataset_id} with {image_count} images")
    
    return DataUploadResponse(
        dataset_id=dataset_id,
        message=f"Successfully uploaded {image_count} images",
        num_images=image_count,
        format=format
    )


@router.get("/methods")
async def get_reconstruction_methods() -> Dict[str, Any]:
    """
    Get available reconstruction methods.
    
    Returns information about supported neural rendering techniques
    and their capabilities.
    """
    methods = {
        "nerf": {
            "name": "Neural Radiance Fields",
            "description": "Original NeRF implementation for high-quality novel view synthesis",
            "speed": "slow",
            "quality": "high",
            "memory_usage": "high"
        },
        "instant_ngp": {
            "name": "Instant Neural Graphics Primitives", 
            "description": "Fast NeRF variant with hash encoding for real-time training",
            "speed": "fast",
            "quality": "high",
            "memory_usage": "medium"
        },
        "gaussian_splatting": {
            "name": "3D Gaussian Splatting",
            "description": "Real-time neural rendering with 3D Gaussians",
            "speed": "very_fast",
            "quality": "high",
            "memory_usage": "low"
        },
        "mipnerf": {
            "name": "Mip-NeRF",
            "description": "Anti-aliased NeRF for better handling of scale",
            "speed": "slow",
            "quality": "very_high",
            "memory_usage": "high"
        },
        "tensorf": {
            "name": "TensoRF",
            "description": "Tensor decomposition for efficient NeRF",
            "speed": "medium",
            "quality": "high",
            "memory_usage": "medium"
        },
        "nerfacto": {
            "name": "Nerfacto",
            "description": "Balanced NeRF approach with good speed/quality tradeoff",
            "speed": "medium",
            "quality": "high",
            "memory_usage": "medium"
        }
    }
    
    return {
        "methods": methods,
        "default": "instant_ngp",
        "recommended_for_beginners": "nerfacto"
    }


async def run_reconstruction(job_id: str, dataset_id: str, config: ReconstructionConfig):
    """Run reconstruction job in background."""
    try:
        # Update job status
        reconstruction_jobs[job_id]["status"] = "initializing"
        reconstruction_jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
        
        # Create engine
        render_config = RenderConfig(
            model_type=config.method,
            resolution=config.resolution,
            num_iterations=config.num_iterations,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            device=config.device,
            num_samples=config.num_samples,
            use_mixed_precision=config.use_mixed_precision
        )
        
        engine = Engine(render_config)
        active_engines[job_id] = engine
        
        # Load dataset
        dataset_path = Path("datasets") / dataset_id
        await engine.load_data({"type": "images", "path": str(dataset_path)})
        
        # Start training
        reconstruction_jobs[job_id]["status"] = "running"
        reconstruction_jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
        
        async for progress in engine.train_async():
            # Update job progress
            reconstruction_jobs[job_id].update({
                "progress": progress.progress,
                "current_iteration": progress.iteration,
                "loss": progress.loss,
                "metrics": progress.metrics,
                "updated_at": datetime.utcnow().isoformat()
            })
        
        # Save results
        output_path = engine.save_results()
        
        # Complete job
        reconstruction_jobs[job_id].update({
            "status": "completed",
            "progress": 100.0,
            "output_path": output_path,
            "updated_at": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Completed reconstruction job {job_id}")
        
    except Exception as e:
        # Handle errors
        reconstruction_jobs[job_id].update({
            "status": "failed",
            "error_message": str(e),
            "updated_at": datetime.utcnow().isoformat()
        })
        
        logger.error(f"Reconstruction job {job_id} failed: {e}")
        
        # Clean up
        if job_id in active_engines:
            del active_engines[job_id]


from datetime import datetime 