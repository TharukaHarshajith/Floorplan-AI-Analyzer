#!/usr/bin/env python3
"""
Simple Pipeline Test Script for Floorplan AI Analyzer
This is a lightweight version that checks basic functionality without heavy dependencies.
"""

import sys
import os
from pathlib import Path

def test_basic_setup():
    """Test basic project setup"""
    print("=== Basic Setup Test ===")

    # Check if src directory exists
    src_dir = Path('src')
    if src_dir.exists():
        print("✅ src/ directory exists")
    else:
        print("❌ src/ directory missing")
        return False

    # Check if main modules exist
    required_files = [
        'src/config.py',
        'src/data_preparation.py',
        'src/train.py',
        'src/validate.py',
        'src/predict.py',
        'src/export.py',
        'src/main.py',
        'src/utils.py',
        'src/check_pipeline.py'
    ]

    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("✅ All required files present")

    # Check data directory
    data_dir = Path('data')
    if data_dir.exists():
        print("✅ data/ directory exists")
        # Check for key files
        key_files = ['data/classes.txt', 'data/data.yaml']
        for file in key_files:
            if Path(file).exists():
                print(f"✅ {file} exists")
            else:
                print(f"⚠️  {file} missing (will be created during data prep)")
    else:
        print("⚠️  data/ directory missing (will be created during data prep)")

    # Check model file
    model_file = Path('yolo11s.pt')
    if model_file.exists():
        print("✅ Model file (yolo11s.pt) exists")
    else:
        print("❌ Model file (yolo11s.pt) missing - download from ultralytics")

    return True

def test_python_syntax():
    """Test Python syntax of all modules"""
    print("\n=== Python Syntax Test ===")

    import py_compile
    import sys

    modules = [
        'src/config.py',
        'src/data_preparation.py',
        'src/train.py',
        'src/validate.py',
        'src/predict.py',
        'src/export.py',
        'src/main.py',
        'src/utils.py',
        'src/check_pipeline.py'
    ]

    failed = []
    for module in modules:
        try:
            py_compile.compile(module, doraise=True)
            print(f"✅ {module} syntax OK")
        except py_compile.PyCompileError as e:
            print(f"❌ {module} syntax error: {e}")
            failed.append(module)
        except FileNotFoundError:
            print(f"❌ {module} not found")
            failed.append(module)

    return len(failed) == 0

def test_imports():
    """Test basic imports without running functions"""
    print("\n=== Import Test ===")

    # Test basic Python imports
    try:
        import sys
        import os
        import pathlib
        print("✅ Basic Python imports OK")
    except ImportError as e:
        print(f"❌ Basic Python import failed: {e}")
        return False

    # Test config import (should work without dependencies)
    try:
        sys.path.insert(0, 'src')
        import config
        print("✅ Config import OK")
        print(f"   Data dir: {config.DATA_DIR}")
        print(f"   Model: {config.MODEL_PATH}")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False

    # Test utils import
    try:
        import utils
        print("✅ Utils import OK")
    except ImportError as e:
        print(f"❌ Utils import failed: {e}")
        return False

    return True

def main():
    """Run all tests"""
    print("Floorplan AI Analyzer - Basic Pipeline Test")
    print("=" * 50)

    tests = [
        ("Basic Setup", test_basic_setup),
        ("Python Syntax", test_python_syntax),
        ("Basic Imports", test_imports)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 50)
    print("SUMMARY:")

    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All basic tests passed!")
        print("\nNext steps:")
        print("1. Install missing dependencies: pip install ultralytics scikit-learn")
        print("2. Run full pipeline check: python src/check_pipeline.py")
        print("3. Run main pipeline: python src/main.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())