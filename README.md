# Floorplan AI Analyzer

This project automates floorplan analysis using YOLO object detection.

## Project Structure

```
src/
├── config.py          # Configuration settings and paths
├── data_preparation.py # Data splitting and YAML creation
├── train.py           # Model training functions
├── validate.py        # Model validation functions
├── predict.py         # Prediction functions
├── export.py          # Model export functions
├── utils.py           # Utility functions
├── main.py            # Main pipeline script
├── ci_pipeline.py     # CI/CD pipeline script
└── check_pipeline.py  # Pipeline validation script
test_pipeline.py       # Basic setup test (no dependencies needed)
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure data is in the correct structure:
   ```
   data/
   ├── images/     # PNG images
   ├── labels/     # TXT label files
   └── classes.txt # Class names
   ```

## Usage

### Full Pipeline
Run the complete workflow:
```bash
python src/main.py
```

### Individual Steps
- Data preparation: `python -c "from src.data_preparation import prepare_data; prepare_data()"`
- Training: `python -c "from src.train import train_yolo_model; train_yolo_model()"`
- Validation: `python -c "from src.validate import validate_model; validate_model()"`
- Prediction: `python -c "from src.predict import predict_image; predict_image()"`
- Export: `python -c "from src.export import export_model; export_model()"`

### CI Pipeline
For continuous integration:
```bash
python src/ci_pipeline.py
```

### Pipeline Check
Validate that the entire pipeline works correctly:
```bash
# Quick check (1 epoch training)
python src/check_pipeline.py

# Skip training (use existing model)
python src/check_pipeline.py --skip-training

# Full training check
python src/check_pipeline.py --full-training

# Verbose output
python src/check_pipeline.py --verbose
```

### Basic Test
Run a lightweight test that doesn't require all dependencies:
```bash
python test_pipeline.py
```

## Configuration

Edit `src/config.py` to modify:
- Data paths
- Training parameters (epochs, batch size, etc.)
- Model settings