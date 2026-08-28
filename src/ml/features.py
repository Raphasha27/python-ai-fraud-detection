import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional


class FeatureEngineer:
    def __init__(self):
        self.merchant_categories = {
            "electronics": 0,
            "groceries": 1,
            "restaurants": 2,
            "entertainment": 3,
            "travel": 4,
            "healthcare": 5,
            "utilities": 6,
            "retail": 7,
            "other": 8,
        }

    def extract(self, transaction) -> dict:
        timestamp = transaction.timestamp
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0

        merchant_category_encoded = self.merchant_categories.get(
            transaction.merchant_category.lower(), 8
        )

        amount = transaction.amount
        amount_to_avg_ratio = amount / 100.0

        features = {
            "amount": float(amount),
            "hour": int(hour),
            "day_of_week": int(day_of_week),
            "is_international": int(transaction.is_international),
            "merchant_category_encoded": int(merchant_category_encoded),
            "transaction_count_24h": 1,
            "avg_amount_7d": float(amount),
            "distance_from_home": 0.0,
            "time_since_last_transaction": 0,
            "card_age_days": 365,
            "amount_to_avg_ratio": float(amount_to_avg_ratio),
            "is_weekend": int(is_weekend),
        }

        return features

    def extract_batch(self, transactions: list) -> pd.DataFrame:
        features_list = [self.extract(t) for t in transactions]
        return pd.DataFrame(features_list)

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c
