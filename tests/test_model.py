import pytest
import pandas as pd
import numpy as np
from src.ml.model import FraudDetectionModel
from src.ml.features import FeatureEngineer
from src.ml.train import generate_synthetic_data


@pytest.fixture
def model():
    return FraudDetectionModel()


@pytest.fixture
def feature_engineer():
    return FeatureEngineer()


@pytest.fixture
def trained_model():
    model = FraudDetectionModel()
    X, y = generate_synthetic_data(n_samples=1000)
    model.train(X, y)
    return model


@pytest.fixture
def sample_features():
    return {
        "amount": 150.0,
        "hour": 14,
        "day_of_week": 2,
        "is_international": 0,
        "merchant_category_encoded": 5,
        "transaction_count_24h": 3,
        "avg_amount_7d": 120.0,
        "distance_from_home": 5.2,
        "time_since_last_transaction": 3600,
        "card_age_days": 365,
        "amount_to_avg_ratio": 1.25,
        "is_weekend": 0,
    }


class TestFraudDetectionModel:
    def test_model_initialization(self, model):
        assert model.model is not None
        assert model.scaler is not None
        assert not model.is_loaded

    def test_model_training(self):
        model = FraudDetectionModel()
        X, y = generate_synthetic_data(n_samples=500)
        metrics = model.train(X, y)

        assert "classification_report" in metrics
        assert "confusion_matrix" in metrics
        assert "auc_roc" in metrics
        assert 0 <= metrics["auc_roc"] <= 1

    def test_model_prediction(self, trained_model, sample_features):
        prediction = trained_model.predict(sample_features)

        assert "fraud_probability" in prediction
        assert "is_fraud" in prediction
        assert "confidence" in prediction
        assert 0 <= prediction["fraud_probability"] <= 1
        assert isinstance(prediction["is_fraud"], bool)
        assert 0 <= prediction["confidence"] <= 1

    def test_model_save_and_load(self, trained_model, tmp_path):
        model_path = tmp_path / "test_model.pkl"
        trained_model.save(str(model_path))

        new_model = FraudDetectionModel()
        new_model.load(str(model_path))

        assert new_model.is_loaded

    def test_feature_importance(self, trained_model):
        importance = trained_model.get_feature_importance()
        assert len(importance) == 12
        assert all(v >= 0 for v in importance.values())


class TestFeatureEngineer:
    def test_extract_features(self, feature_engineer):
        from datetime import datetime

        class MockTransaction:
            amount = 150.0
            merchant_id = "M12345"
            merchant_category = "electronics"
            timestamp = datetime(2026, 8, 27, 14, 30, 0)
            card_number = "4111111111111111"
            location_lat = -26.2041
            location_lon = 28.0473
            is_international = False

        features = feature_engineer.extract(MockTransaction())

        assert "amount" in features
        assert "hour" in features
        assert "day_of_week" in features
        assert features["hour"] == 14
        assert features["day_of_week"] == 2

    def test_distance_calculation(self, feature_engineer):
        distance = feature_engineer.calculate_distance(
            -26.2041, 28.0473, -33.9249, 18.4241
        )
        assert distance > 0
        assert distance < 2000
