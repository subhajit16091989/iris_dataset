from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.train import train_logistic_regression
from src.utils.common import load_yaml
from src.utils.common import set_seed
from src.evaluate.metrics import classification_metrics
import joblib
import logging
import os
import argparse
import yaml
import json


def _bootstrap_dataframe(path: str) -> pd.DataFrame:
    """Load a binary-classification DataFrame from a named sklearn dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if path.suffix != ".csv":
        raise ValueError(f"Unsupported file format: {path.suffix}. Only .csv files are supported.")
    # binary classification datasets are expected to be in CSV format
    df = pd.read_csv(path)
    if "target" in df.columns and "target_name" in df.columns:
        df = df.drop(columns=["target_name"])
    df = df[df["target"].isin([0, 1])].reset_index(drop=True)
    return df
    
def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a binary-classification dataset from a named sklearn dataset."""
    df = _bootstrap_dataframe(path)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Train a logistic regression model on a binary classification dataset.")
    parser.add_argument("config", default="configs/train/default.yaml", help="Path to the training configuration YAML file.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    # Load configuration from YAML file
    train_cfg = load_yaml(args.config)
    data_cfg = load_yaml(train_cfg["paths"]["data_config"])
    feature_cfg = load_yaml(train_cfg["paths"]["feature_config"])
    set_seed(train_cfg["seed"])
    # Split and Train the model
    X, y = load_data(data_cfg["dataset_path"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_cfg["test_size"], random_state=data_cfg["random_state"], stratify=y
    )
    model = train_logistic_regression(X_train, y_train, model_params={"C": 1.0}, scale_nummeric=True)
    # Save the model
    output_dir = Path(train_cfg["paths"]["runs_dir"])/train_cfg["run_name"]
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / train_cfg["artifacts"]["model_file"]
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, y_pred, y_prob)
    metrics_path = output_dir / train_cfg["artifacts"]["metrics_file"]
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    
if __name__ == "__main__":
    main()