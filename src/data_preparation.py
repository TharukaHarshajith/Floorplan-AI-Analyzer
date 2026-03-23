import os
import shutil
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
import config
from config import (
    IMAGES_DIR, LABELS_DIR, CLASSES_FILE, DATA_DIR,
    TRAIN_DIR, VAL_DIR, TEST_DIR, DATA_YAML, RANDOM_STATE
)

def load_classes():
    """Load class names from classes.txt"""
    with open(CLASSES_FILE, 'r') as f:
        return [line.strip() for line in f.readlines()]

def get_image_files():
    """Get list of image files"""
    return list(IMAGES_DIR.glob('*.png'))

def split_data():
    """Split data into train/val/test sets"""
    image_files = get_image_files()
    image_names = [f.stem for f in image_files]

    # Split: 70% train, 20% val, 10% test
    train_names, temp_names = train_test_split(
        image_names, test_size=0.3, random_state=RANDOM_STATE
    )
    val_names, test_names = train_test_split(
        temp_names, test_size=0.33, random_state=RANDOM_STATE
    )

    return train_names, val_names, test_names

def create_directories():
    """Create train/val/test directories"""
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        (split_dir / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'labels').mkdir(parents=True, exist_ok=True)

def move_files(names, split):
    """Move image and label files to split directory"""
    for name in names:
        # Move image
        src_img = IMAGES_DIR / f"{name}.png"
        dst_img = DATA_DIR / split / 'images' / f"{name}.png"
        if src_img.exists():
            shutil.copy2(src_img, dst_img)

        # Move label
        src_label = LABELS_DIR / f"{name}.txt"
        dst_label = DATA_DIR / split / 'labels' / f"{name}.txt"
        if src_label.exists():
            shutil.copy2(src_label, dst_label)

def create_data_yaml():
    """Create data.yaml file for YOLO training"""
    class_names = load_classes()

    # Use absolute paths
    train_path = str((TRAIN_DIR / 'images').resolve())
    val_path = str((VAL_DIR / 'images').resolve())
    test_path = str((TEST_DIR / 'images').resolve())

    data_yaml = {
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'nc': len(class_names),
        'names': class_names
    }

    with open(DATA_YAML, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    return data_yaml

def prepare_data():
    """Main function to prepare data"""
    print("Loading classes...")
    class_names = load_classes()
    print(f"Classes: {class_names}")

    print("Counting files...")
    image_files = get_image_files()
    label_files = list(LABELS_DIR.glob('*.txt'))
    print(f"Number of images: {len(image_files)}")
    print(f"Number of labels: {len(label_files)}")

    print("Splitting data...")
    train_names, val_names, test_names = split_data()
    print(f"Train: {len(train_names)}, Val: {len(val_names)}, Test: {len(test_names)}")

    print("Creating directories...")
    create_directories()

    print("Moving files...")
    move_files(train_names, 'train')
    move_files(val_names, 'val')
    move_files(test_names, 'test')

    print("Creating data.yaml...")
    data_yaml = create_data_yaml()
    print("Data preparation completed!")

    return data_yaml