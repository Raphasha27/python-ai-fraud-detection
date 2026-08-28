from fastapi import APIRouter, HTTPException
from typing import Optional

router = APIRouter()


@router.get("/")
async def root():
    return {"message": "AI Fraud Detection API"}


@router.get("/info")
async def api_info():
    return {
        "name": "AI Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "metrics": "/metrics",
            "transactions": "/transactions",
        },
    }
