"""
Continuous Integration script for floorplan AI analyzer.
This can be run in CI/CD pipelines to automate testing and validation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_preparation import prepare_data
from train import train_yolo_model
from validate import validate_model
from utils import check_requirements

def run_ci_pipeline():
    """Run CI pipeline: check requirements, prepare data, train, validate"""
    try:
        print("=== CI Pipeline: Floorplan AI Analyzer ===")

        print("\nChecking requirements...")
        if not check_requirements():
            print("Requirements not met. Exiting.")
            sys.exit(1)

        print("\nPreparing data...")
        prepare_data()

        print("\nTraining model (CI mode - reduced epochs)...")
        # Use fewer epochs for CI to speed up
        from config import EPOCHS
        original_epochs = EPOCHS
        # Temporarily reduce epochs for CI
        import src.config as config
        config.EPOCHS = 2  # Quick training for CI

        train_results = train_yolo_model()

        # Restore original epochs
        config.EPOCHS = original_epochs

        print("\nValidating model...")
        val_metrics = validate_model()

        # Check if metrics meet minimum thresholds
        min_map50 = 0.5  # Example threshold
        if val_metrics.box.map50 < min_map50:
            print(f"Validation failed: mAP50 {val_metrics.box.map50} < {min_map50}")
            sys.exit(1)

        print("\nCI Pipeline passed!")
        return True

    except Exception as e:
        print(f"CI Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_ci_pipeline()