from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import sys
import threading

# Global lock for thread safety with ML models
processing_lock = threading.Lock()

# Ensure we can import the package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ai_module_yolo import FaceRecognizer, RECOGNITION_THRESHOLD
    recognizer = FaceRecognizer()
    print(f"[AI Server] Initialized Face Detection System (YOLOv8-face: {recognizer.yolo_model.model.pt_path if recognizer and recognizer.yolo_model else 'Unknown'})")
except Exception as e:
    print(f"[AI Server] Error initializing face detection: {e}")
    recognizer = None

app = Flask(__name__)

@app.route("/train", methods=["POST"])
def train():
    if not recognizer:
        return jsonify({"error": "AI module not initialized"}), 500
    
    data = request.json or {}
    student_id = data.get("studentId")
    frames_dir = data.get("framesDir")

    if not student_id or not frames_dir:
        return jsonify({"error": "Invalid payload: studentId and framesDir required"}), 400

    # Handle paths
    if not os.path.isabs(frames_dir):
        # Assuming relative to backend
        backend_frames_dir = os.path.join("..", "backend", frames_dir)
        if os.path.exists(backend_frames_dir):
            frames_dir = os.path.abspath(backend_frames_dir)
        else:
            frames_dir = os.path.abspath(frames_dir)

    print(f"[AI Server] Training student {student_id} from {frames_dir}")
    
    success = recognizer.train_from_frames(frames_dir, student_id)
    
    if not success:
        return jsonify({"error": "Training failed (no faces found or other error)"}), 400

    return jsonify({"status": "trained", "message": f"Successfully trained {student_id}"}), 200

@app.route("/recognize", methods=["POST"])
def recognize():
    file = request.files.get("frame")
    if not file:
        return jsonify({"error": "No frame received"}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if recognizer:
        with processing_lock:
            student_id = recognizer.recognize_face(frame)
    else:
        student_id = None

    if student_id is None:
        return jsonify({"recognized": False})

    return jsonify({"recognized": True, "studentId": student_id})

@app.route("/recognize-live", methods=["POST"])
def recognize_live():
    if not recognizer:
        return jsonify({"error": "AI module not initialized"}), 500
    
    try:
        file = request.files.get("frame")
        if not file:
            return jsonify({"error": "No frame received"}), 400

        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Failed to decode image", "recognized": False}), 400

        # Try recognition (Batch mode)
        with processing_lock:
            # Returns list of {"student_id", "confidence", "bbox", "recognized"}
            results = recognizer.recognize_all_faces(frame)
        
        # Backward compatibility / Summary flag
        any_recognized = any(r["recognized"] for r in results)

        return jsonify({
            "results": results,
            "recognized": any_recognized,
            "count": len(results)
        })

    except Exception as e:
        print(f"[AI Server] Error: {e}")
        return jsonify({"error": str(e), "recognized": False}), 500

if __name__ == "__main__":
    app.run(port=8000, debug=False)
