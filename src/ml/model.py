import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Optional


class FraudDetectionModel:
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.is_loaded = False
        self.feature_names = [
            "amount",
            "hour",
            "day_of_week",
            "is_international",
            "merchant_category_encoded",
            "transaction_count_24h",
            "avg_amount_7d",
            "distance_from_home",
            "time_since_last_transaction",
            "card_age_days",
            "amount_to_avg_ratio",
            "is_weekend",
        ]
        self._training_metrics: dict = {}

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)

        self._training_metrics = {
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "cv_scores_mean": float(cv_scores.mean()),
            "cv_scores_std": float(cv_scores.std()),
            "auc_roc": float(roc_auc_score(y_test, y_proba)),
        }

        return self._training_metrics

    def predict(self, features: dict) -> dict:
        if not self.is_loaded and not hasattr(self.model, "estimators_"):
            raise ValueError("Model not trained or loaded")

        X = pd.DataFrame([features])[self.feature_names]
        X_scaled = self.scaler.transform(X)

        probability = self.model.predict_proba(X_scaled)[0]
        prediction = self.model.predict(X_scaled)[0]

        return {
            "fraud_probability": float(probability[1]),
            "is_fraud": bool(prediction),
            "confidence": float(max(probability)),
        }

    def predict_batch(self, X: pd.DataFrame) -> list[dict]:
        X_scaled = self.scaler.transform(X[self.feature_names])
        probabilities = self.model.predict_proba(X_scaled)
        predictions = self.model.predict(X_scaled)

        results = []
        for prob, pred in zip(probabilities, predictions):
            results.append(
                {
                    "fraud_probability": float(prob[1]),
                    "is_fraud": bool(pred),
                    "confidence": float(max(prob)),
                }
            )
        return results

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "training_metrics": self._training_metrics,
            },
            path,
        )

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self._training_metrics = data.get("training_metrics", {})
        self.is_loaded = True

    def get_feature_importance(self) -> dict:
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances.tolist()))
