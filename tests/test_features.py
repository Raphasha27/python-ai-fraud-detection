import pytest
import pandas as pd
import numpy as np
from src.ml.features import FeatureEngineer
from src.data.pipeline import DataPipeline
from src.data.generator import generate_transactions


@pytest.fixture
def feature_engineer():
    return FeatureEngineer()


@pytest.fixture
def data_pipeline():
    return DataPipeline()


class TestFeatureEngineer:
    def test_merchant_category_encoding(self, feature_engineer):
        assert feature_engineer.merchant_categories["electronics"] == 0
        assert feature_engineer.merchant_categories["groceries"] == 1
        assert feature_engineer.merchant_categories["other"] == 8

    def test_distance_calculation_same_point(self, feature_engineer):
        distance = feature_engineer.calculate_distance(0, 0, 0, 0)
        assert distance == 0

    def test_distance_calculation_known_distance(self, feature_engineer):
        distance = feature_engineer.calculate_distance(0, 0, 0, 1)
        assert 110 < distance < 112


class TestDataPipeline:
    def test_process_raw_data(self, data_pipeline):
        df = pd.DataFrame(
            {
                "amount": [100, 200, 300],
                "timestamp": pd.to_datetime(["2026-08-27 10:00:00"] * 3),
                "card_number": ["CARD1", "CARD1", "CARD2"],
            }
        )

        result = data_pipeline.process_raw_data(df)

        assert "hour" in result.columns
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns
        assert "amount_log" in result.columns

    def test_detect_anomalies(self, data_pipeline):
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "amount": np.concatenate([np.random.normal(100, 10, 100), [10000]]),
            }
        )

        result = data_pipeline.detect_anomalies(df)

        assert "amount_zscore" in result.columns
        assert "is_amount_anomaly" in result.columns
        assert result["is_amount_anomaly"].sum() >= 1


class TestDataGenerator:
    def test_generate_transactions(self):
        df = generate_transactions(n=100, fraud_ratio=0.1)

        assert len(df) == 100
        assert "amount" in df.columns
        assert "is_fraud" in df.columns
        assert df["is_fraud"].sum() == 10

    def test_generate_transactions_default_ratio(self):
        df = generate_transactions(n=1000)

        assert len(df) == 1000
        fraud_ratio = df["is_fraud"].mean()
        assert 0.03 < fraud_ratio < 0.07
