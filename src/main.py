from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uvicorn

from .config import get_settings
from .ml.model import FraudDetectionModel
from .ml.features import FeatureEngineer
from .db.connection import Database

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Real-time transaction fraud detection using machine learning",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = FraudDetectionModel()
feature_engineer = FeatureEngineer()
db = Database()


class Transaction(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_id: str = Field(..., min_length=1, description="Merchant identifier")
    merchant_category: str = Field(..., description="Merchant category")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    card_number: str = Field(..., min_length=13, max_length=19, description="Card number")
    location_lat: float = Field(..., ge=-90, le=90, description="Latitude")
    location_lon: float = Field(..., ge=-180, le=180, description="Longitude")
    is_international: bool = Field(default=False, description="International transaction")


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud: bool
    confidence: float = Field(..., ge=0, le=1)
    risk_level: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
    version: str


class MetricsResponse(BaseModel):
    total_predictions: int
    fraud_detected: int
    average_probability: float
    model_accuracy: float


@app.on_event("startup")
async def startup():
    try:
        model.load(settings.model_path)
    except FileNotFoundError:
        print(f"Warning: Model not found at {settings.model_path}. Run scripts/train_model.py")
    await db.connect()


@app.on_event("shutdown")
async def shutdown():
    await db.disconnect()


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=model.is_loaded,
        database_connected=db.is_connected,
        version=settings.app_version,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: Transaction):
    import time
    start_time = time.time()
    
    try:
        features = feature_engineer.extract(transaction)
        prediction = model.predict(features)
        
        risk_level = "low"
        if prediction["fraud_probability"] > 0.7:
            risk_level = "critical"
        elif prediction["fraud_probability"] > 0.5:
            risk_level = "high"
        elif prediction["fraud_probability"] > 0.3:
            risk_level = "medium"
        
        transaction_id = await db.save_transaction(transaction, prediction)
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            transaction_id=transaction_id,
            fraud_probability=prediction["fraud_probability"],
            is_fraud=prediction["is_fraud"],
            confidence=prediction["confidence"],
            risk_level=risk_level,
            processing_time_ms=round(processing_time, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    try:
        metrics = await db.get_model_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@app.get("/transactions")
async def list_transactions(limit: int = 100, offset: int = 0):
    try:
        transactions = await db.list_transactions(limit=limit, offset=offset)
        return transactions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list transactions: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
