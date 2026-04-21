from __future__ import annotations
import json
from pathlib import Path
import joblib
import logging
import argparse
import pandas as pd
import yaml
from src.utils.common import load_yaml
from src.evaluate.metrics import classification_metrics

def load_model(model_path: str):
    """Load a trained model from the specified path."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if model_path.suffix != ".joblib":
        raise ValueError(f"Unsupported file format: {model_path.suffix}. Only .joblib files are supported.")
    model = joblib.load(model_path)
    logging.info(f"Model loaded from {model_path}")
    return model

def load_test_data(test_data_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load test data from the specified path."""
    test_data_path = Path(test_data_path)
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")
    if test_data_path.suffix != ".csv":
        raise ValueError(f"Unsupported file format: {test_data_path.suffix}. Only .csv files are supported.")
    df = pd.read_csv(test_data_path)
    if "target" not in df.columns:
        raise ValueError("Test data must contain a 'target' column.")
    X_test = df.drop(columns=["target"])
    y_test = df["target"]
    logging.info(f"Test data loaded from {test_data_path}")
    return X_test, y_test

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a test dataset.")
    parser.add_argument("--config", default="configs/evaluate/default.yaml", help="Path to the evaluation configuration YAML file.")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    # Load configuration from YAML file
    eval_cfg = load_yaml(args.config)
    data_cfg = load_yaml(eval_cfg["paths"]["data_config"])
    model_cfg = load_yaml(eval_cfg["paths"]["model_config"])
    train_cfg = load_yaml(eval_cfg["paths"]["train_config"])
    # Load the model and test data
    model = load_model(model_cfg["model_path"])
    X_test, y_test = load_test_data(data_cfg["test_data_path"])
    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    metrics = classification_metrics(y_test, y_pred, y_prob)
    logging.info(f"Evaluation metrics: {metrics}")
    #Output metrics to a JSON file
    #output_dir = Path(data_cfg["output_dir"])/ train_cfg["run_name"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Evaluation metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()