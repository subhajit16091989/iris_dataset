import logging
import argparse
import joblib
import pandas as pd
from pathlib import Path
import yaml
from src.utils.common import load_yaml

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

def load_inference_data(inference_data_path: str) -> pd.DataFrame:
    """Load inference data from the specified path."""
    inference_data_path = Path(inference_data_path)
    if not inference_data_path.exists():
        raise FileNotFoundError(f"Inference data file not found: {inference_data_path}")
    if inference_data_path.suffix != ".csv":
        raise ValueError(f"Unsupported file format: {inference_data_path.suffix}. Only .csv files are supported.")
    df = pd.read_csv(inference_data_path)
    # remove target column if it exists, since we are doing inference
    if "target" in df.columns:
        df = df.drop(columns=["target","target_name"], errors="ignore")
    logging.info(f"Inference data loaded from {inference_data_path}")
    return df

def predict(config):
    logging.basicConfig(level=logging.INFO)
    # Load configuration from YAML file

    inference_cfg = load_yaml(config)
    data_cfg = load_yaml(inference_cfg["paths"]["data_config"])
    model_cfg = load_yaml(inference_cfg["paths"]["model_config"])

    # Load the model and inference data
    model = load_model(model_cfg["model_path"])
    inference_data = load_inference_data(inference_cfg["inference_data_path"])
    # Make predictions
    predictions = model.predict(inference_data)
    logging.info(f"Predictions made successfully.")
    return predictions