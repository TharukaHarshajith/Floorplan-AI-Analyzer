from ultralytics import YOLO
from pathlib import Path
import config
from config import DATA_YAML
import train
from train import train_yolo_model

def find_best_model_path():
    """Find the best model weights path from training runs"""
    import config
    from config import PROJECT, NAME

    # First, check the expected path from the current training run
    expected_path = Path(PROJECT) / NAME / 'weights' / 'best.pt'
    if expected_path.exists():
        return expected_path

    # Fallback: look for any recent training runs
    runs_dir = Path(PROJECT)
    if not runs_dir.exists():
        raise FileNotFoundError(f"No training runs found in {runs_dir}")

    # Find all subdirectories
    subdirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No training subdirectories found in {runs_dir}")

    # Sort by modification time (newest first)
    subdirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Check each run for best.pt
    for run_dir in subdirs:
        weights_dir = run_dir / 'weights'
        best_pt = weights_dir / 'best.pt'
        if best_pt.exists():
            return best_pt

    # If no best.pt found, check for last.pt as fallback
    for run_dir in subdirs:
        weights_dir = run_dir / 'weights'
        last_pt = weights_dir / 'last.pt'
        if last_pt.exists():
            print(f"Warning: Using last.pt instead of best.pt from {run_dir}")
            return last_pt

    raise FileNotFoundError(f"No model weights found in {runs_dir}")

def validate_model(model_path=None, split='test'):
    """Validate the trained model"""
    if model_path is None:
        model_path = find_best_model_path()

    print(f"Loading model from: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    model = YOLO(str(model_path))

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found at {DATA_YAML}")

    print(f"Validating on {split} set...")
    metrics = model.val(data=str(DATA_YAML), split=split)

    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")

    return metrics