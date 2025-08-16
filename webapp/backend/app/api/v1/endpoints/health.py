"""
Health check endpoints for TAROT CKD Risk Prediction API
"""

import time
import psutil
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, Request
import structlog

from app.core.config import settings
from app.schemas.prediction_simple import HealthResponse
from app.models.model_manager import ModelManager


logger = structlog.get_logger(__name__)
router = APIRouter()


def get_model_manager(request: Request) -> Optional[ModelManager]:
    """Dependency to get the model manager from app state (optional)"""
    return getattr(request.app.state, 'model_manager', None)


@router.get("/", response_model=HealthResponse)
async def health_check(
    request: Request,
    model_manager: Optional[ModelManager] = Depends(get_model_manager)
) -> HealthResponse:
    """
    Basic health check endpoint
    
    Returns the current status of the API including:
    - Service status
    - Model loading status
    - Basic system information
    - API version
    """
    models_loaded = 0
    if model_manager:
        models_loaded = len(model_manager.models)
    
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version=settings.VERSION,
        models_loaded=models_loaded,
        system_info=None  # Basic health check doesn't include detailed system info
    )


@router.get("/detailed", response_model=HealthResponse)
async def detailed_health_check(
    request: Request,
    model_manager: Optional[ModelManager] = Depends(get_model_manager)
) -> HealthResponse:
    """
    Detailed health check with system information
    
    Returns comprehensive status including:
    - Model status and types
    - Memory usage
    - CPU information
    - System resources
    """
    # Get system information
    system_info = get_system_info()
    
    # Get model information
    models_loaded = 0
    model_status = {}
    
    if model_manager and model_manager.is_loaded:
        status = model_manager.get_status()
        models_loaded = status['model_count']
        model_status = {
            'total_models': status['model_count'],
            'model_types': status['model_types'],
            'preprocessor_loaded': status['preprocessor_loaded'],
            'baseline_hazards': status['baseline_hazard_count'],
            'memory_usage_mb': status['memory_usage_mb']
        }
    
    system_info['models'] = model_status
    
    return HealthResponse(
        status="healthy" if models_loaded > 0 else "degraded",
        timestamp=time.time(),
        version=settings.VERSION,
        models_loaded=models_loaded,
        system_info=system_info
    )


@router.get("/models")
async def model_status(
    request: Request,
    model_manager: Optional[ModelManager] = Depends(get_model_manager)
) -> Dict[str, Any]:
    """
    Get detailed model status information
    
    Returns information about:
    - Loaded models and their types
    - Model metadata
    - Memory usage
    - Loading status
    """
    if not model_manager:
        return {
            "status": "not_initialized",
            "message": "Model manager not available"
        }
    
    if not model_manager.is_loaded:
        return {
            "status": "not_loaded",
            "message": "Models are not loaded yet"
        }
    
    status = model_manager.get_status()
    
    # Add detailed model information
    model_details = []
    for model_name, metadata in model_manager.model_metadata.items():
        model_details.append({
            "name": model_name,
            "type": metadata.get('model_type', 'unknown'),
            "network": metadata.get('network_type', 'unknown'),
            "event": metadata.get('event_type', 'unknown'),
            "file_size_mb": metadata.get('file_size_mb', 0)
        })
    
    return {
        "status": "loaded",
        "summary": status,
        "models": model_details,
        "total_size_mb": sum(m.get('file_size_mb', 0) for m in model_manager.model_metadata.values()),
        "load_time": "Available on startup"
    }


@router.get("/readiness")
async def readiness_check(
    request: Request,
    model_manager: Optional[ModelManager] = Depends(get_model_manager)
) -> Dict[str, Any]:
    """
    Kubernetes-style readiness probe
    
    Returns 200 if the service is ready to accept traffic,
    503 if models are still loading or failed to load.
    """
    if not model_manager or not model_manager.is_loaded:
        return {
            "ready": False,
            "reason": "Models not loaded",
            "timestamp": time.time()
        }
    
    # Check if minimum number of models are loaded
    min_models = 10  # Minimum required for basic functionality
    if len(model_manager.models) < min_models:
        return {
            "ready": False,
            "reason": f"Insufficient models loaded ({len(model_manager.models)} < {min_models})",
            "timestamp": time.time()
        }
    
    return {
        "ready": True,
        "models_loaded": len(model_manager.models),
        "timestamp": time.time()
    }


@router.get("/liveness")
async def liveness_check() -> Dict[str, Any]:
    """
    Kubernetes-style liveness probe
    
    Returns 200 if the service is alive and functioning,
    regardless of model loading status.
    """
    return {
        "alive": True,
        "timestamp": time.time(),
        "version": settings.VERSION
    }


def get_system_info() -> Dict[str, Any]:
    """Get current system information"""
    try:
        # Memory information
        memory = psutil.virtual_memory()
        
        # CPU information
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Disk information
        disk = psutil.disk_usage('/')
        
        return {
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_percent": memory.percent
            },
            "cpu": {
                "usage_percent": cpu_percent,
                "core_count": cpu_count
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "used_percent": round((disk.used / disk.total) * 100, 1)
            },
            "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
        }
        
    except Exception as e:
        logger.warning("Failed to get system info", error=str(e))
        return {
            "error": "System information unavailable",
            "message": str(e)
        }