"""
MLOps Assignment 8: Model Training and Serialization Script
Dataset: Scikit-learn built-in Iris dataset
Model: RandomForestClassifier
"""

import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train_model() -> None:
    # Resolve directory paths relative to this script
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "model"
    model_path = model_dir / "model.joblib"
    stats_path = model_dir / "training_stats.json"

    # Ensure model directory exists
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = list(iris.feature_names)
    target_names = list(iris.target_names)

    # 80/20 train-test split with fixed random_state and stratified target
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # 2. Train Model
    n_estimators = 100
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # 3. Evaluate Model
    y_pred = model.predict(X_test)
    test_accuracy = float(accuracy_score(y_test, y_pred))

    # Print training and evaluation summary
    print("=" * 60)
    print("TRAINING AND EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Dataset Size:               {len(X)} samples")
    print(f"Number of Features:         {X.shape[1]}")
    print(f"Number of Training Samples: {len(X_train)}")
    print(f"Number of Testing Samples:  {len(X_test)}")
    print(f"Model Algorithm:            {model.__class__.__name__}")
    print(f"Number of Trees:            {n_estimators}")
    print(f"Test Accuracy:              {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print("=" * 60)

    # 4. Save Trained Model
    joblib.dump(model, model_path)
    print(f"Model successfully saved to: {model_path}")

    # 5. Compute Training Statistics (population std: ddof=0)
    means_dict = {}
    stds_dict = {}
    feature_stats = {}

    for idx, name in enumerate(feature_names):
        col_mean = float(np.mean(X_train[:, idx]))
        col_std = float(np.std(X_train[:, idx], ddof=0))
        means_dict[name] = col_mean
        stds_dict[name] = col_std
        feature_stats[name] = {
            "mean": col_mean,
            "std": col_std,
        }

    training_stats = {
        "model_version": "1.0.0",
        "num_training_samples": int(len(X_train)),
        "feature_names": feature_names,
        "target_class_names": target_names,
        "target_names": target_names,
        "mean": means_dict,
        "std": stds_dict,
        "feature_stats": feature_stats,
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(training_stats, f, indent=2)

    print(f"Training statistics successfully saved to: {stats_path}")


if __name__ == "__main__":
    train_model()
