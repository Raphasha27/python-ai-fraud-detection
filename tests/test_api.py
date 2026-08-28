import pytest
from fastapi.testclient import TestClient
from src.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "version" in data


class TestPredictEndpoint:
    def test_predict_fraud(self, client):
        transaction = {
            "amount": 150.00,
            "merchant_id": "M12345",
            "merchant_category": "electronics",
            "timestamp": "2026-08-27T14:30:00Z",
            "card_number": "4111111111111111",
            "location_lat": -26.2041,
            "location_lon": 28.0473,
            "is_international": False,
        }

        response = client.post("/predict", json=transaction)
        assert response.status_code == 200
        data = response.json()
        assert "transaction_id" in data
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "confidence" in data
        assert "risk_level" in data
        assert "processing_time_ms" in data

    def test_predict_invalid_amount(self, client):
        transaction = {
            "amount": -100,
            "merchant_id": "M12345",
            "merchant_category": "electronics",
            "timestamp": "2026-08-27T14:30:00Z",
            "card_number": "4111111111111111",
            "location_lat": -26.2041,
            "location_lon": 28.0473,
            "is_international": False,
        }

        response = client.post("/predict", json=transaction)
        assert response.status_code == 422

    def test_predict_missing_fields(self, client):
        transaction = {"amount": 150.00}

        response = client.post("/predict", json=transaction)
        assert response.status_code == 422


class TestMetricsEndpoint:
    def test_get_metrics(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "fraud_detected" in data
        assert "average_probability" in data


class TestTransactionsEndpoint:
    def test_list_transactions(self, client):
        response = client.get("/transactions")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
