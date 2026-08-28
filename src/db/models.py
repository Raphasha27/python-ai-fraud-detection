from sqlalchemy import Column, String, Float, Boolean, DateTime, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    amount = Column(Float, nullable=False)
    merchant_id = Column(String(50), nullable=False)
    merchant_category = Column(String(50), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    card_number = Column(String(19), nullable=False)
    location_lat = Column(Float, nullable=False)
    location_lon = Column(Float, nullable=False)
    is_international = Column(Boolean, default=False)

    fraud_probability = Column(Float)
    is_fraud = Column(Boolean)
    confidence = Column(Float)
    risk_level = Column(String(20))

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "amount": self.amount,
            "merchant_id": self.merchant_id,
            "merchant_category": self.merchant_category,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "card_number": f"****{self.card_number[-4:]}" if self.card_number else None,
            "location_lat": self.location_lat,
            "location_lon": self.location_lon,
            "is_international": self.is_international,
            "fraud_probability": self.fraud_probability,
            "is_fraud": self.is_fraud,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
