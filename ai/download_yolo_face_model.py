#!/usr/bin/env python3
"""
Script to download YOLOv8-face model automatically
"""

import urllib.request
import os
import sys

MODEL_URLS = {
    "yolov8n-face.pt": "https://github.com/derronqi/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt",
    # Alternative: Use standard YOLOv8 if face model not available
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
}

def download_model(model_name="yolov8n-face.pt"):
    """Download YOLO model from GitHub releases"""
    if model_name not in MODEL_URLS:
        print(f"Error: Unknown model {model_name}")
        return False
    
    url = MODEL_URLS[model_name]
    output_path = os.path.join(os.path.dirname(__file__), model_name)
    
    if os.path.exists(output_path):
        print(f"Model {model_name} already exists at {output_path}")
        return True
    
    print(f"Downloading {model_name} from {url}...")
    print("This may take a few minutes depending on your internet connection...")
    
    try:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\rProgress: {percent}%")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
        print(f"\n✓ Successfully downloaded {model_name}")
        print(f"  Saved to: {os.path.abspath(output_path)}")
        return True
    except Exception as e:
        print(f"\n✗ Failed to download {model_name}: {e}")
        print(f"\nManual download:")
        print(f"  1. Visit: https://github.com/derronqi/yolov8-face/releases")
        print(f"  2. Download: yolov8n-face.pt")
        print(f"  3. Place it in: {os.path.dirname(os.path.abspath(__file__))}")
        return False

if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "yolov8n-face.pt"
    download_model(model_name)
