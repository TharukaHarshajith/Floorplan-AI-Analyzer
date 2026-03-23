from ultralytics import YOLO
from pathlib import Path
import config
from config import (
    MODEL_PATH, DATA_YAML, EPOCHS, IMGSZ, BATCH, NAME, PROJECT, DEVICE
)

def train_yolo_model():
    """Train a YOLO model using data.yaml"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Data YAML not found: {DATA_YAML}")

    model = YOLO(str(MODEL_PATH))

    print(f"Starting training with data file {DATA_YAML}")
    print(f"Data file exists: {DATA_YAML.exists()}")

    # Verify data.yaml content
    with open(DATA_YAML, 'r') as f:
        content = f.read()
        print(f"data.yaml content preview:\n{content[:200]}...")

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        name=NAME,
        project=PROJECT,
        device=DEVICE,
        save=True,  # Ensure weights are saved
        save_period=1  # Save every epoch
    )

    print("Training completed")

    # Check if weights were saved
    weights_dir = Path(PROJECT) / NAME / 'weights'
    if weights_dir.exists():
        best_pt = weights_dir / 'best.pt'
        last_pt = weights_dir / 'last.pt'
        print(f"Weights directory: {weights_dir}")
        print(f"best.pt exists: {best_pt.exists()}")
        print(f"last.pt exists: {last_pt.exists()}")
    else:
        print(f"Warning: Weights directory not found: {weights_dir}")

    return results