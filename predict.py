from src.inference.predictor import predict
from src.inference.predictor import load_model
from src.inference.predictor import load_inference_data
import argparse
from src.utils.common import load_yaml
import logging
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run inference using a trained model on new data.")
    parser.add_argument("--config", default="configs/inference/default.yaml", help="Path to the inference configuration YAML file.")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()
    inference_cfg = load_yaml(args.config)
    data_cfg = load_yaml(inference_cfg["paths"]["data_config"])
    model_cfg = load_yaml(inference_cfg["paths"]["model_config"])
    # Load infrence data 
    inference_data = load_inference_data(inference_cfg["inference_data_path"])
    # Load the model and inference data, and make predictions
    model_path = model_cfg["model_path"]
    load_inference_model = load_model(model_path)
    ## Make predictions
    predictions = predict(args.config)
    print(predictions)
    # Output predictions to a CSV file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    predictions_path = output_dir / "predictions_inference.csv"
    inference_data["predictions"] = predictions
    inference_data.to_csv(predictions_path, index=False)
    logging.info(f"Predictions saved to {predictions_path}")

if __name__ == "__main__":
    main()