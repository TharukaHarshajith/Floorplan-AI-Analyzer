"""
Utility functions for the floorplan AI analyzer project.
"""

def print_config():
    """Print current configuration"""
    import config
    from config import (
        DATA_DIR, IMAGES_DIR, LABELS_DIR, CLASSES_FILE, DATA_YAML,
        MODEL_PATH, EPOCHS, IMGSZ, BATCH, DEVICE
    )

    print("=== Configuration ===")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Images Directory: {IMAGES_DIR}")
    print(f"Labels Directory: {LABELS_DIR}")
    print(f"Classes File: {CLASSES_FILE}")
    print(f"Data YAML: {DATA_YAML}")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Training Epochs: {EPOCHS}")
    print(f"Image Size: {IMGSZ}")
    print(f"Batch Size: {BATCH}")
    print(f"Device: {DEVICE}")
    print("====================")

def check_requirements():
    """Check if required files and directories exist"""
    import config
    from config import DATA_DIR, IMAGES_DIR, LABELS_DIR, CLASSES_FILE, MODEL_PATH

    missing = []

    if not DATA_DIR.exists():
        missing.append(f"Data directory: {DATA_DIR}")
    if not IMAGES_DIR.exists():
        missing.append(f"Images directory: {IMAGES_DIR}")
    if not LABELS_DIR.exists():
        missing.append(f"Labels directory: {LABELS_DIR}")
    if not CLASSES_FILE.exists():
        missing.append(f"Classes file: {CLASSES_FILE}")
    if not MODEL_PATH.exists():
        missing.append(f"Model file: {MODEL_PATH}")

    if missing:
        print("Missing requirements:")
        for item in missing:
            print(f"  - {item}")
        return False

    print("All requirements satisfied.")
    return True