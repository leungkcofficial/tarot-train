"""
API v1 router for TAROT CKD Risk Prediction
"""

from fastapi import APIRouter

from app.api.v1.endpoints import predict, health, static_pages

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(predict.router, prefix="/predict", tags=["prediction"])
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(static_pages.router, prefix="/info", tags=["information"])