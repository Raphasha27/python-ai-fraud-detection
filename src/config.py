from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "AI Fraud Detection API"
    app_version: str = "1.0.0"
    environment: str = "development"
    
    database_url: str = "postgresql://postgres:postgres@localhost:5432/fraud_detection"
    redis_url: str = "redis://localhost:6379"
    
    model_path: str = "models/fraud_model.pkl"
    
    cors_origins: list[str] = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
