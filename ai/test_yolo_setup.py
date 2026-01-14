#!/usr/bin/env python3
"""Test script to check YOLO setup"""

import os
import sys

print("=" * 60)
print("YOLO Setup Check")
print("=" * 60)

# Check ultralytics
print("\n1. Checking ultralytics package...")
try:
    from ultralytics import YOLO
    print("   ✓ ultralytics is installed")
    print(f"   Location: {YOLO.__module__}")
except ImportError as e:
    print(f"   ✗ ultralytics is NOT installed")
    print(f"   Error: {e}")
    print(f"   Install with: pip install ultralytics")
    sys.exit(1)

# Check for model files
print("\n2. Checking for YOLO model files...")
current_dir = os.getcwd()
pt_files = [f for f in os.listdir(current_dir) if f.endswith('.pt')]

if pt_files:
    print(f"   ✓ Found {len(pt_files)} .pt file(s):")
    for pt_file in pt_files:
        full_path = os.path.abspath(pt_file)
        size = os.path.getsize(full_path) / (1024 * 1024)  # Size in MB
        print(f"      - {pt_file} ({size:.2f} MB)")
    
    # Try to load each model
    print("\n3. Testing model loading...")
    for pt_file in pt_files:
        try:
            print(f"   Trying to load: {pt_file}")
            model = YOLO(pt_file)
            print(f"   ✓ Successfully loaded {pt_file}")
            print(f"   Model info: {type(model).__name__}")
            break
        except Exception as e:
            print(f"   ✗ Failed to load {pt_file}: {e}")
            continue
else:
    print(f"   ✗ No .pt files found in {current_dir}")
    print(f"   Download yolov8n-face.pt from:")
    print(f"   https://github.com/derronqi/yolov8-face/releases")

# Check deepface
print("\n4. Checking DeepFace package...")
try:
    from deepface import DeepFace
    print("   ✓ DeepFace is installed")
except ImportError as e:
    print(f"   ✗ DeepFace is NOT installed")
    print(f"   Error: {e}")
    print(f"   Install with: pip install deepface")

print("\n" + "=" * 60)
