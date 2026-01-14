# YOLOv8-Face Recognition Module

A robust face detection and recognition system using YOLOv8-face for detection and ArcFace (via DeepFace) for recognition.

## Features

- ✅ **YOLOv8-face Detection**: More accurate than Haar cascades
- ✅ **ArcFace Embeddings**: State-of-the-art face recognition
- ✅ **Video Processing**: Extracts and processes 65 frames from 10-second videos
- ✅ **Embedding Aggregation**: Median or average aggregation for robust matching
- ✅ **Visualization**: Draws bounding boxes on detected faces
- ✅ **Database Management**: Save/load student embeddings from JSON
- ✅ **Cosine Similarity**: Efficient matching algorithm

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements_yolo.txt
```

### 2. Download YOLOv8-face Model

Download the pre-trained YOLOv8-face model from:
- **GitHub Releases**: https://github.com/derronqi/yolov8-face/releases
- Place `yolov8n-face.pt` in the `ai/` directory

**Recommended Models:**
- `yolov8n-face.pt` - Nano (fastest, smallest)
- `yolov8s-face.pt` - Small (balanced)
- `yolov8m-face.pt` - Medium (more accurate)

## Quick Start

### Basic Usage

```python
from yolo_face_recognition import YOLOFaceRecognizer

# Initialize recognizer
recognizer = YOLOFaceRecognizer(
    model_path="yolov8n-face.pt",
    detection_confidence=0.95,
    recognition_threshold=0.75,
    embedding_model="ArcFace"
)

# Process a video
student_name, similarity, embedding = recognizer.process_video(
    video_path="query_video.mp4",
    max_frames=65,
    aggregate_method="median",
    visualize=True,
    save_output="output.mp4"
)

# Print results
if student_name:
    print(f"Identified: {student_name} (similarity: {similarity:.4f})")
else:
    print(f"Not recognized (best similarity: {similarity:.4f})")
```

### Building a Student Database

```python
from yolo_face_recognition import create_sample_database

# Dictionary mapping student names to video paths
student_videos = {
    "Alice": "videos/alice.mp4",
    "Bob": "videos/bob.mp4",
    # ... add more students
}

# Create database from videos
recognizer = create_sample_database(recognizer, student_videos)
```

### Loading/Saving Database

```python
import json
import numpy as np

# Save database
database = {
    name: embedding.tolist() 
    for name, embedding in recognizer.known_embeddings.items()
}
with open("student_database.json", "w") as f:
    json.dump(database, f)

# Load database
with open("student_database.json", "r") as f:
    database = json.load(f)
for name, embedding_list in database.items():
    embedding = np.array(embedding_list, dtype=np.float32)
    recognizer.add_student_embedding(name, embedding)
```

## Complete Example

See `example_yolo_usage.py` for a complete working example.

```bash
python example_yolo_usage.py
```

## API Reference

### `YOLOFaceRecognizer`

#### Initialization

```python
recognizer = YOLOFaceRecognizer(
    model_path="yolov8n-face.pt",      # Path to YOLOv8-face model
    detection_confidence=0.95,          # Min confidence for detection (0-1)
    recognition_threshold=0.75,         # Min similarity for recognition (0-1)
    embedding_model="ArcFace"          # DeepFace model name
)
```

#### Key Methods

##### `extract_frames(video_path, max_frames=None)`
Extract frames from a video file.

##### `detect_faces_yolo(frame, visualize=False)`
Detect faces in a frame using YOLOv8-face.
Returns: `List[Tuple[x1, y1, x2, y2, confidence]]`

##### `generate_embedding(face_img)`
Generate ArcFace embedding for a face image.
Returns: `np.ndarray` (embedding vector)

##### `aggregate_embeddings(embeddings, method="median")`
Aggregate multiple embeddings into one.
- `method="median"`: Robust to outliers (recommended)
- `method="average"`: Simple average

##### `recognize_student(query_embedding)`
Identify student from embedding.
Returns: `Tuple[student_name, similarity_score]`

##### `process_video(video_path, max_frames=65, aggregate_method="median", visualize=True, save_output=None)`
Complete pipeline: extract → detect → embed → aggregate → recognize.
Returns: `Tuple[student_name, similarity, aggregated_embedding]`

## Configuration

### Detection Confidence

- **0.95** (default): High precision, may miss some faces
- **0.80**: Balanced
- **0.60**: More sensitive, may include false positives

### Recognition Threshold

- **0.75** (default): Strict matching
- **0.65**: Moderate matching
- **0.50**: Lenient matching (may have false positives)

### Embedding Models

DeepFace supports multiple models:
- `"ArcFace"` - Recommended, best accuracy
- `"Facenet"`
- `"VGG-Face"`
- `"OpenFace"`

### Aggregation Methods

- **"median"** (recommended): Robust to outliers, better for varying lighting/angles
- **"average"**: Simple mean, faster but sensitive to outliers

## Integration with Existing System

To integrate with your existing `ai_module.py`:

1. **Replace face detection** in `_detect_largest_face()`:

```python
from yolo_face_recognition import YOLOFaceRecognizer

# Initialize once (global or in class)
yolo_recognizer = YOLOFaceRecognizer("yolov8n-face.pt")

def _detect_largest_face(frame):
    detections = yolo_recognizer.detect_faces_yolo(frame, visualize=False)
    if not detections:
        return None
    
    # Return largest detection
    largest = max(detections, key=lambda d: (d[2]-d[0]) * (d[3]-d[1]))
    x1, y1, x2, y2 = largest[:4]
    return (x1, y1, x2-x1, y2-y1)  # Return as (x, y, w, h)
```

2. **Keep existing recognition pipeline** or replace with embeddings.

## Performance Tips

1. **Model Size**: Use `yolov8n-face.pt` for speed, `yolov8m-face.pt` for accuracy
2. **Frame Selection**: Process every Nth frame instead of all frames
3. **Embedding Cache**: Cache embeddings for known faces
4. **Batch Processing**: Process multiple frames in parallel

## Troubleshooting

### "Model file not found"
- Download `yolov8n-face.pt` from GitHub releases
- Place in the same directory as the script

### "No faces detected"
- Lower `detection_confidence` (try 0.80)
- Check video quality and lighting
- Ensure faces are clearly visible

### "Low similarity scores"
- Increase video quality
- Ensure good lighting in videos
- Use more frames for aggregation
- Check that database videos are representative

### "DeepFace installation errors"
```bash
pip install deepface --upgrade
pip install tensorflow
```

## File Structure

```
ai/
├── yolo_face_recognition.py    # Main module
├── example_yolo_usage.py       # Example usage script
├── requirements_yolo.txt       # Dependencies
├── README_YOLO.md             # This file
├── yolov8n-face.pt            # YOLOv8-face model (download separately)
└── student_database.json      # Saved embeddings (auto-generated)
```

## License

This module is designed to integrate with your existing project. Please refer to:
- YOLOv8-face: Check original repository license
- DeepFace: MIT License
- Ultralytics YOLOv8: AGPL-3.0

