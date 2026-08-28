import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


def generate_transactions(n: int = 1000, fraud_ratio: float = 0.05) -> pd.DataFrame:
    np.random.seed(42)

    n_fraud = int(n * fraud_ratio)
    n_legit = n - n_fraud

    base_time = datetime(2026, 8, 27, 8, 0, 0)

    legitimate = _generate_legitimate_transactions(n_legit, base_time)
    fraud = _generate_fraud_transactions(n_fraud, base_time)

    df_legit = pd.DataFrame(legitimate)
    df_fraud = pd.DataFrame(fraud)

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df["is_fraud"] = [0] * n_legit + [1] * n_fraud

    return df.sample(frac=1).reset_index(drop=True)


def _generate_legitimate_transactions(n: int, base_time: datetime) -> dict:
    return {
        "amount": np.random.lognormal(mean=3.5, sigma=1.0, size=n),
        "merchant_id": [f"M{np.random.randint(1000, 9999)}" for _ in range(n)],
        "merchant_category": np.random.choice(
            ["groceries", "restaurants", "retail", "utilities"], size=n
        ),
        "timestamp": [base_time + timedelta(hours=np.random.randint(6, 23)) for _ in range(n)],
        "card_number": [f"411111{np.random.randint(1000000000, 9999999999)}" for _ in range(n)],
        "location_lat": np.random.uniform(-34.0, -25.0, size=n),
        "location_lon": np.random.uniform(18.0, 30.0, size=n),
        "is_international": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
    }


def _generate_fraud_transactions(n: int, base_time: datetime) -> dict:
    return {
        "amount": np.random.lognormal(mean=5.5, sigma=1.5, size=n),
        "merchant_id": [f"M{np.random.randint(1000, 9999)}" for _ in range(n)],
        "merchant_category": np.random.choice(
            ["electronics", "travel", "entertainment"], size=n
        ),
        "timestamp": [base_time + timedelta(hours=np.random.randint(0, 24)) for _ in range(n)],
        "card_number": [f"411111{np.random.randint(1000000000, 9999999999)}" for _ in range(n)],
        "location_lat": np.random.uniform(-90, 90, size=n),
        "location_lon": np.random.uniform(-180, 180, size=n),
        "is_international": np.random.choice([0, 1], size=n, p=[0.3, 0.7]),
    }
