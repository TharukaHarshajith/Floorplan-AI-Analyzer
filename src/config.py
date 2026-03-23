from pathlib import Path

# Project root (assuming src/ is at root level)
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
IMAGES_DIR = DATA_DIR / 'images'
LABELS_DIR = DATA_DIR / 'labels'
TRAIN_DIR = DATA_DIR / 'train'
VAL_DIR = DATA_DIR / 'val'
TEST_DIR = DATA_DIR / 'test'

# Files
CLASSES_FILE = DATA_DIR / 'classes.txt'
DATA_YAML = DATA_DIR / 'data.yaml'

# Model settings
MODEL_PATH = PROJECT_ROOT / 'yolo11s.pt'
PROJECT = 'runs/train'
NAME = 'floorplan_yolo'
EPOCHS = 2
IMGSZ = 640
BATCH = 8
DEVICE = 'cpu'

# Random state for reproducibility
RANDOM_STATE = 42

# Test image path (example)
TEST_IMAGE_PATH = PROJECT_ROOT / 'images' / '14027' / 'F1_scaled.png'