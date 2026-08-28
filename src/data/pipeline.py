import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, timedelta


class DataPipeline:
    def __init__(self):
        self.feature_stats = {}

    def process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        if "amount" in df.columns:
            df["amount_log"] = np.log1p(df["amount"])

        return df

    def aggregate_features(self, df: pd.DataFrame, window_hours: int = 24) -> pd.DataFrame:
        df = df.copy()

        if "timestamp" in df.columns and "card_number" in df.columns:
            df = df.sort_values(["card_number", "timestamp"])

            for window in [24, 168, 720]:
                df[f"tx_count_{window}h"] = (
                    df.groupby("card_number")["timestamp"]
                    .rolling(f"{window}h")
                    .count()
                    .reset_index(0, drop=True)
                )

            df["avg_amount_7d"] = (
                df.groupby("card_number")["amount"]
                .rolling("168h")
                .mean()
                .reset_index(0, drop=True)
            )

        return df

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "amount" in df.columns:
            mean_amount = df["amount"].mean()
            std_amount = df["amount"].std()
            df["amount_zscore"] = (df["amount"] - mean_amount) / std_amount
            df["is_amount_anomaly"] = (df["amount_zscore"].abs() > 3).astype(int)

        return df
