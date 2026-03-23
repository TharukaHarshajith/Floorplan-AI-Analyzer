from ultralytics import YOLO
from pathlib import Path
import validate
from validate import find_best_model_path

def export_model(format='onnx', dynamic=True):
    """Export the trained model to production format"""
    model_path = find_best_model_path()
    print(f"Loading model from: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    model = YOLO(str(model_path))

    print(f"Exporting model to {format} format...")
    model.export(format=format, dynamic=dynamic)

    print("Export completed")
    return True