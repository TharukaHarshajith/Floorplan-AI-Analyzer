#!/usr/bin/env python3
"""
Main pipeline script for floorplan AI analyzer.
This script automates the entire workflow: data preparation, training, validation, prediction, and export.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_preparation import prepare_data
from train import train_yolo_model
from validate import validate_model
from predict import predict_image, show_results
from export import export_model

def run_pipeline():
    """Run the complete ML pipeline"""
    try:
        print("=== Floorplan AI Analyzer Pipeline ===")

        print("\n1. Preparing data...")
        data_yaml = prepare_data()

        print("\n2. Training model...")
        train_results = train_yolo_model()

        print("\n3. Validating model...")
        val_metrics = validate_model()

        print("\n4. Running prediction...")
        pred_results = predict_image()
        show_results(pred_results)

        print("\n5. Exporting model...")
        export_model()

        print("\n=== Pipeline completed successfully! ===")

    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()