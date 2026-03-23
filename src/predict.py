from ultralytics import YOLO
from pathlib import Path
import config
from config import TEST_IMAGE_PATH
import validate
from validate import find_best_model_path

def predict_image(image_path=None, save=True):
    """Run prediction on an image"""
    if image_path is None:
        image_path = TEST_IMAGE_PATH

    model_path = find_best_model_path()
    print(f"Loading model from: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    model = YOLO(str(model_path))

    print(f"Running prediction on: {image_path}")
    results = model.predict(source=str(image_path), save=save)

    print("Prediction completed")
    return results

def show_results(results):
    """Display prediction results"""
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Boxes: {len(result.boxes)}")
        print(f"  Classes: {result.names}")
        # Note: result.show() would display the image, but in headless CI this might not work
        # result.show()