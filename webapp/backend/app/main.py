"""
TAROT CKD Risk Prediction Web Application
Main FastAPI application entry point
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import structlog

from app.core.config import settings
from app.core.logging import setup_logging
from app.models.model_manager import ModelManager
from app.api.v1.api import api_router

# Setup structured logging
setup_logging()
logger = structlog.get_logger(__name__)

# Global model manager instance
model_manager: ModelManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events - startup and shutdown
    """
    # Startup
    logger.info("Starting TAROT CKD Risk Prediction API")
    
    try:
        # Initialize model manager
        global model_manager
        model_manager = ModelManager()
        
        # Load all models
        logger.info("Loading all 36 ensemble models...")
        start_time = time.time()
        await model_manager.load_models()
        load_time = time.time() - start_time
        
        status = model_manager.get_status()
        logger.info(
            "Models loaded successfully", 
            load_time_seconds=load_time,
            loaded_models=status['loaded_models'],
            total_models=status['total_models']
        )
        
        # Store model manager in app state
        app.state.model_manager = model_manager
        
    except Exception as e:
        logger.error("Failed to initialize application", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down TAROT CKD Risk Prediction API")
    if model_manager:
        await model_manager.cleanup()


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="""
    TAROT CKD Risk Prediction API
    
    A comprehensive API for predicting chronic kidney disease (CKD) progression risk using 
    ensemble deep learning models. Provides risk predictions for dialysis initiation and 
    all-cause mortality over 1-5 year horizons.
    
    ## Features
    
    * **Real-time Risk Prediction**: Ensemble of 36+ deep learning models
    * **Clinical Validation**: eGFR-based screening and comprehensive input validation  
    * **Unit Conversion**: Automatic conversion between laboratory units
    * **SHAP Analysis**: Feature importance for clinical interpretation
    * **Privacy-First**: Zero data logging with session-based processing
    
    ## Clinical Usage
    
    **Target Users**: Healthcare professionals and CKD patients
    
    **Input Requirements**: Age ≥18, eGFR 10-60 mL/min/1.73m²
    
    **Clinical Benchmarks**:
    - 5-year risk 3-5%: Nephrology referral
    - 2-year risk >10%: Multidisciplinary care  
    - 2-year risk >40%: KRT preparation
    """,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json" if settings.DEBUG else None,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Privacy headers - no caching for API responses
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    
    return response


@app.middleware("http")
async def add_request_logging(request: Request, call_next):
    """Log all requests for monitoring"""
    start_time = time.time()
    
    # Log request
    logger.info(
        "Request received",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host,
        user_agent=request.headers.get("user-agent", "unknown")
    )
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(
        "Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time
    )
    
    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    logger.warning("Validation error", error=str(exc), url=str(request.url))
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation Error",
            "message": str(exc),
            "type": "validation_error"
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(
        "HTTP exception", 
        status_code=exc.status_code,
        error=exc.detail,
        url=str(request.url)
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail,
            "type": "http_error"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(
        "Unexpected error", 
        error=str(exc),
        error_type=type(exc).__name__,
        url=str(request.url),
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again.",
            "type": "internal_error"
        }
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.VERSION,
        "models_loaded": getattr(app.state, 'model_manager', None).get_status()['loaded_models'] if hasattr(app.state, 'model_manager') and app.state.model_manager else 0
    }


# Include API routes
app.include_router(api_router, prefix=settings.API_V1_STR)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "description": "TAROT CKD Risk Prediction API",
        "docs_url": "/docs",
        "health_url": "/health",
        "api_v1_url": f"{settings.API_V1_STR}/predict",
        "models_status": getattr(app.state, 'model_manager', None).get_status() if hasattr(app.state, 'model_manager') and app.state.model_manager else {"loaded_models": 0, "total_models": 36}
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        access_log=True
    )