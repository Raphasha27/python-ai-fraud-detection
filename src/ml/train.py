import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from .model import FraudDetectionModel
from .features import FeatureEngineer


def generate_synthetic_data(n_samples: int = 10000) -> tuple[pd.DataFrame, pd.Series]:
    np.random.seed(42)

    n_fraud = int(n_samples * 0.05)
    n_legit = n_samples - n_fraud

    legitimate = {
        "amount": np.random.lognormal(mean=3.5, sigma=1.0, size=n_legit),
        "hour": np.random.choice(range(6, 23), size=n_legit),
        "day_of_week": np.random.choice(range(7), size=n_legit),
        "is_international": np.random.choice([0, 1], size=n_legit, p=[0.9, 0.1]),
        "merchant_category_encoded": np.random.randint(0, 9, size=n_legit),
        "transaction_count_24h": np.random.poisson(lam=2, size=n_legit) + 1,
        "avg_amount_7d": np.random.lognormal(mean=3.5, sigma=0.8, size=n_legit),
        "distance_from_home": np.random.exponential(scale=5, size=n_legit),
        "time_since_last_transaction": np.random.exponential(scale=3600, size=n_legit),
        "card_age_days": np.random.randint(30, 1825, size=n_legit),
        "amount_to_avg_ratio": np.random.normal(1.0, 0.3, size=n_legit),
        "is_weekend": np.random.choice([0, 1], size=n_legit, p=[0.7, 0.3]),
    }

    fraud = {
        "amount": np.random.lognormal(mean=5.0, sigma=1.5, size=n_fraud),
        "hour": np.random.choice(range(0, 24), size=n_fraud),
        "day_of_week": np.random.choice(range(7), size=n_fraud),
        "is_international": np.random.choice([0, 1], size=n_fraud, p=[0.3, 0.7]),
        "merchant_category_encoded": np.random.randint(0, 9, size=n_fraud),
        "transaction_count_24h": np.random.poisson(lam=8, size=n_fraud) + 1,
        "avg_amount_7d": np.random.lognormal(mean=3.5, sigma=0.8, size=n_fraud),
        "distance_from_home": np.random.exponential(scale=50, size=n_fraud),
        "time_since_last_transaction": np.random.exponential(scale=300, size=n_fraud),
        "card_age_days": np.random.randint(1, 365, size=n_fraud),
        "amount_to_avg_ratio": np.random.normal(3.0, 1.5, size=n_fraud),
        "is_weekend": np.random.choice([0, 1], size=n_fraud, p=[0.5, 0.5]),
    }

    df_legit = pd.DataFrame(legitimate)
    df_fraud = pd.DataFrame(fraud)

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    y = pd.Series([0] * n_legit + [1] * n_fraud)

    shuffle_idx = np.random.permutation(len(df))
    df = df.iloc[shuffle_idx].reset_index(drop=True)
    y = y.iloc[shuffle_idx].reset_index(drop=True)

    return df, y


def train_model(output_path: str = "models/fraud_model.pkl") -> dict:
    print("Generating synthetic training data...")
    X, y = generate_synthetic_data(n_samples=10000)

    print(f"Dataset shape: {X.shape}")
    print(f"Fraud ratio: {y.mean():.2%}")

    model = FraudDetectionModel()

    print("Training model...")
    metrics = model.train(X, y)

    print("\nTraining Results:")
    print(metrics["classification_report"])
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"CV Score: {metrics['cv_scores_mean']:.4f} (+/- {metrics['cv_scores_std']:.4f})")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"\nModel saved to {output_path}")

    return metrics


if __name__ == "__main__":
    train_model()
