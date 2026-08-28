#!/usr/bin/env python3
"""Train the fraud detection model"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.train import train_model


def main():
    print("=" * 50)
    print("AI Fraud Detection Model Training")
    print("=" * 50)

    metrics = train_model(output_path="models/fraud_model.pkl")

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"AUC-ROC Score: {metrics['auc_roc']:.4f}")
    print(f"Cross-Validation Score: {metrics['cv_scores_mean']:.4f}")


if __name__ == "__main__":
    main()
