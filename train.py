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
    #This is useful for binary classification but we need multiclasses for this iris dataset. 
    #df = df[df["target"].isin([0, 1])].reset_index(drop=True)
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
    model_cfg = load_yaml(train_cfg["paths"]["model_config"])
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
    #y_prob = model.predict_proba(X_test)[:, 1]
    y_prob = model.predict_proba(X_test)
    metrics = classification_metrics(y_test, y_pred, y_prob)
    metrics_path = output_dir / train_cfg["artifacts"]["metrics_file"]
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Predict on the test set and save predictions
    pred_df = X_test.copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_pred"] = y_pred
    if y_prob is not None:
        if y_prob.ndim == 1:
            pred_df["y_score"] = y_prob
        else:
            for i in range(y_prob.shape[1]):
                pred_df[f"y_score_class_{i}"] = y_prob[:, i]
    predictions_path = output_dir / train_cfg["artifacts"]["predictions_file"]
    pred_df.to_csv(predictions_path, index=False)
    logging.info(f"Predictions saved to {predictions_path}")

    holdout = X_test.copy()
    holdout["target"] = y_test.values
    holdout_path = output_dir / train_cfg["artifacts"]["holdout_file"]
    holdout.to_csv(holdout_path, index=False)
    logging.info(f"Holdout set saved to {holdout_path}")

    # Summarize results
    summary_path = Path(train_cfg["paths"]["runs_dir"]) / train_cfg["run_name"] / "summary.csv"
    summary_row = {
        "run_name": train_cfg["run_name"],
        "model": model_cfg.get("model_type", "unknown"),
        "accuracy": metrics.get("accuracy", ""),
        "f1": metrics.get("f1", ""),
        "roc_auc": metrics.get("roc_auc", ""),
        "notes": "baseline run",
    }
    # Append instead of overwrite to keep a lightweight experiment ledger.
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([summary_row])
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved run artifacts to: {output_dir}")
    print(f"Metrics: {metrics}")
    
if __name__ == "__main__":
    main()