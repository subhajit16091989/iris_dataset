from __future__ import annotations

from html import parser
import subprocess
import sys
from pathlib import Path
import pandas as pd
import argparse
from src.utils.common import load_yaml
import uuid
import joblib

def run_step(cmd: list[str], step_name: str):
    print(f"Running {step_name}...")
    subprocess.run(cmd, check=True)
    print(f"{step_name} completed.")

def assert_file_exists(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"{name} not found at {path}")

def main():
    #parser = argparse.ArgumentParser(description="Train a logistic regression model on a multiclass classification dataset.")
    #parser.add_argument("--config", required=True, help="Path to config file")
    #args = parser.parse_args()

    # Load configuration from YAML file
    eval_cfg = load_yaml("configs/train/default.yaml")
    data_cfg = load_yaml(eval_cfg["paths"]["data_config"])
    model_cfg = load_yaml(eval_cfg["paths"]["model_config"])

    
    # Create isolated run directory
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    base_dir = Path(data_cfg["output_dir"])/ "smoketest" / run_id
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using run directory: {base_dir}")
    
   # Auto-train on first run so smoke tests can run in fresh clones.
    print("Running training...")
    run_step(
        [sys.executable, "train.py", "--config", "configs/train/default.yaml", "--output_dir", str(base_dir)],
        "Training",
    )
    ### Validate Model Artifacts
    model_path = base_dir / "model.joblib"
    assert_file_exists(model_path, "Trained model")
    # Load the model 
    print("Loading Model")
    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        raise TypeError("Loaded model is not a valid ML model (missing predict method)")
    ## lodad the model and holdout set, and run evaluation
    print("Running evaluation on holdout set...")
    run_step(
        [sys.executable, "evaluate.py", "--config", "configs/evaluate/default.yaml", "--output_dir", str(base_dir)],
        "Evaluation",
    )
    assert_file_exists(base_dir / "evaluation_metrics.json", "Evaluation metrics")
    #### Run inference on the holdout set and save predictions
    print("Running inference on holdout set...")
    run_step(
        [sys.executable, "predict.py", "--config", "configs/inference/default.yaml", "--output_dir", str(base_dir)],
        "Inference",
    )
    assert_file_exists(base_dir / "predictions_inference.csv", "Inference predictions")
    # if preditictions are empty, raise an error
    pred_df = pd.read_csv(base_dir / "predictions_inference.csv")
    if pred_df.empty:
        raise ValueError("Predictions file is empty.")
    if pred_df.isnull().sum().sum() > 0:
        raise ValueError("Predictions contain null values.")
    print("Smoke test passed! Model training, evaluation, and inference ran successfully.")

if __name__ == "__main__":    
    main()