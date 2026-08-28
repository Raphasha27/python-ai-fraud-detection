from .model import FraudDetectionModel
from .features import FeatureEngineer


class FraudPredictor:
    def __init__(self, model_path: str = "models/fraud_model.pkl"):
        self.model = FraudDetectionModel()
        self.feature_engineer = FeatureEngineer()
        self.model.load(model_path)

    def predict_transaction(self, transaction) -> dict:
        features = self.feature_engineer.extract(transaction)
        prediction = self.model.predict(features)

        risk_level = self._calculate_risk_level(prediction["fraud_probability"])

        return {
            **prediction,
            "risk_level": risk_level,
        }

    def _calculate_risk_level(self, probability: float) -> str:
        if probability > 0.7:
            return "critical"
        elif probability > 0.5:
            return "high"
        elif probability > 0.3:
            return "medium"
        else:
            return "low"

    def predict_batch(self, transactions: list) -> list[dict]:
        return [self.predict_transaction(t) for t in transactions]
