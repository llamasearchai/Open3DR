"""
Medical API routers for Open3DReconstruction Medical AI Platform.

This module contains all the medical API route handlers organized by medical functionality.
"""

from .medical_diagnostics import router as medical_diagnostics_router
from .medical_imaging import router as medical_imaging_router  
from .drug_discovery import router as drug_discovery_router
from .clinical_decision import router as clinical_decision_router
from .medical_research import router as medical_research_router
from .patient_monitoring import router as patient_monitoring_router
from .medical_websocket import router as medical_websocket_router

# Legacy routers (kept for backward compatibility)
from .reconstruction import router as reconstruction_router

__all__ = [
    "medical_diagnostics_router",
    "medical_imaging_router",
    "drug_discovery_router", 
    "clinical_decision_router",
    "medical_research_router",
    "patient_monitoring_router",
    "medical_websocket_router",
    "reconstruction_router",  # Legacy
] 