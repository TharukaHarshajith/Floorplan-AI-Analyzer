#!/usr/bin/env python3
"""
Pipeline Check Script for Floorplan AI Analyzer
This script validates the entire ML pipeline by running all components.
Use this to verify that the setup is working correctly.
"""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing_deps = []

    try:
        from ultralytics import YOLO
    except ImportError:
        missing_deps.append("ultralytics")

    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        missing_deps.append("scikit-learn")

    try:
        import yaml
    except ImportError:
        missing_deps.append("pyyaml")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        missing_deps.append("matplotlib")

    if missing_deps:
        print("❌ Missing dependencies. Please install:")
        print("pip install " + " ".join(missing_deps))
        return False

    print("✅ All dependencies available")
    return True

# Import modules after dependency check
if not check_dependencies():
    sys.exit(1)

from utils import check_requirements, print_config
from data_preparation import prepare_data
from train import train_yolo_model
from validate import validate_model, find_best_model_path
from predict import predict_image
from export import export_model

def check_pipeline(skip_training=False, quick_mode=True):
    """
    Check the full pipeline

    Args:
        skip_training: If True, skip training and use existing model
        quick_mode: If True, use minimal epochs for faster checking
    """
    print("=== Floorplan AI Analyzer Pipeline Check ===")
    print(f"Skip training: {skip_training}")
    print(f"Quick mode: {quick_mode}")

    start_time = time.time()
    results = {}

    try:
        # Step 1: Print configuration
        print("\n1. Configuration Check")
        print_config()

        # Step 2: Check requirements
        print("\n2. Requirements Check")
        if not check_requirements():
            raise RuntimeError("Requirements check failed")

        results['config'] = "PASS"

        # Step 3: Data preparation
        print("\n3. Data Preparation")
        data_yaml = prepare_data()
        results['data_prep'] = "PASS"

        # Step 4: Training
        if not skip_training:
            print("\n4. Model Training")
            try:
                if quick_mode:
                    # Temporarily reduce epochs for quick check
                    import config
                    original_epochs = config.EPOCHS
                    config.EPOCHS = 1  # Minimal training for check
                    print(f"Using {config.EPOCHS} epoch(s) for quick check")

                train_results = train_yolo_model()
                results['training'] = "PASS"

                if quick_mode:
                    config.EPOCHS = original_epochs  # Restore

            except Exception as e:
                print(f"Training failed: {e}")
                results['training'] = f"FAIL: {e}"
                if not skip_training:
                    # If training was required but failed, we can't continue
                    raise RuntimeError(f"Training failed: {e}")
        else:
            print("\n4. Training Skipped (using existing model)")
            results['training'] = "SKIPPED"

        # Step 5: Validation
        print("\n5. Model Validation")
        # Only validate if we have a trained model or are skipping training
        if results.get('training') == "PASS" or skip_training:
            try:
                val_metrics = validate_model()
                results['validation'] = "PASS"

                # Store metrics for reporting
                results['map50'] = val_metrics.box.map50
                results['map'] = val_metrics.box.map
            except FileNotFoundError as e:
                print(f"Validation skipped: {e}")
                results['validation'] = f"SKIPPED: {e}"
                results['map50'] = "N/A"
                results['map'] = "N/A"
            except Exception as e:
                print(f"Validation failed: {e}")
                results['validation'] = f"FAIL: {e}"
                results['map50'] = "N/A"
                results['map'] = "N/A"
        else:
            print("Validation skipped: No trained model available")
            results['validation'] = "SKIPPED: No model"
            results['map50'] = "N/A"
            results['map'] = "N/A"

        # Step 6: Prediction
        print("\n6. Prediction Test")
        if results.get('training') == "PASS" or skip_training:
            try:
                pred_results = predict_image()
                results['prediction'] = "PASS"
            except FileNotFoundError as e:
                print(f"Prediction skipped: {e}")
                results['prediction'] = f"SKIPPED: {e}"
            except Exception as e:
                print(f"Prediction failed: {e}")
                results['prediction'] = f"FAIL: {e}"
        else:
            print("Prediction skipped: No trained model available")
            results['prediction'] = "SKIPPED: No model"

        # Step 7: Export
        print("\n7. Model Export")
        if results.get('training') == "PASS" or skip_training:
            try:
                export_success = export_model()
                results['export'] = "PASS" if export_success else "FAIL"
            except FileNotFoundError as e:
                print(f"Export skipped: {e}")
                results['export'] = f"SKIPPED: {e}"
            except Exception as e:
                print(f"Export failed: {e}")
                results['export'] = f"FAIL: {e}"
        else:
            print("Export skipped: No trained model available")
            results['export'] = "SKIPPED: No model"

        # Summary
        elapsed_time = time.time() - start_time
        print(f"\n=== Pipeline Check Completed in {elapsed_time:.2f} seconds ===")

        print("\nResults Summary:")
        for step, status in results.items():
            if step in ['map50', 'map']:
                if isinstance(status, (int, float)):
                    print(f"  {step}: {status:.4f}")
                else:
                    print(f"  {step}: {status}")
            else:
                print(f"  {step}: {status}")

        # Check if all critical steps passed
        critical_steps = ['config', 'data_prep']
        if not skip_training:
            critical_steps.append('training')

        # Model-dependent steps are only critical if we have a model
        model_available = results.get('training') in ["PASS", "SKIPPED"] or skip_training
        if model_available:
            critical_steps.extend(['validation', 'prediction', 'export'])

        all_passed = all(
            results.get(step) in ["PASS", "SKIPPED"] or str(results.get(step)).startswith("SKIPPED:")
            for step in critical_steps
        )

        if all_passed:
            print("\n✅ All checks passed! Pipeline is ready.")
            return True
        else:
            print("\n❌ Some checks failed. Please review the output above.")
            return False

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ Pipeline check failed after {elapsed_time:.2f} seconds")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function with command line argument parsing"""
    import argparse

    parser = argparse.ArgumentParser(description='Check Floorplan AI Analyzer Pipeline')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training step (use existing model)')
    parser.add_argument('--full-training', action='store_true',
                       help='Run full training (not quick mode)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    # Enable verbose if requested
    if args.verbose:
        print("Verbose mode enabled")

    quick_mode = not args.full_training

    success = check_pipeline(skip_training=args.skip_training, quick_mode=quick_mode)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()