from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

# Try to use new YOLO-based module, fallback to old if not available
try:
    from ai_module_yolo import train_from_frames, recognize_face, recognize_face_with_coords, detect_all_faces
    USE_YOLO = True
    print("[AI Server] Using YOLO + ArcFace based recognition")
except ImportError:
    try:
        from ai_module import train_from_frames, recognize_face, recognize_face_with_coords, detect_all_faces
        USE_YOLO = False
        print("[AI Server] Using Haar + LBPH based recognition (fallback)")
    except ImportError:
        print("[AI Server] ERROR: No AI module found!")
        train_from_frames = None
        recognize_face = None
        recognize_face_with_coords = None
        detect_all_faces = None

app = Flask(__name__)

@app.route("/train", methods=["POST"])
def train():
    if train_from_frames is None:
        return jsonify({"error": "AI module not available"}), 500
    
    data = request.json or {}
    student_id = data.get("studentId")
    frames_dir = data.get("framesDir")

    if not student_id or not frames_dir:
        return jsonify({"error": "Invalid payload: studentId and framesDir required"}), 400

    # Convert relative path to absolute if needed
    if not os.path.isabs(frames_dir):
        # If frames_dir is relative, assume it's relative to backend directory
        # But we're in ai/ directory, so we need to go up one level
        backend_frames_dir = os.path.join("..", "backend", frames_dir)
        if os.path.exists(backend_frames_dir):
            frames_dir = os.path.abspath(backend_frames_dir)
        else:
            frames_dir = os.path.abspath(frames_dir)

    print(f"[AI Server] Training from frames: {frames_dir} for student: {student_id}")
    
    success = train_from_frames(frames_dir, student_id)
    if not success:
        return jsonify({"error": "No face data found or training failed"}), 400

    return jsonify({"status": "trained", "message": f"Successfully trained model for student {student_id}"}), 200


@app.route("/recognize", methods=["POST"])
def recognize():
    file = request.files.get("frame")
    if not file:
        return jsonify({"error": "No frame received"}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    student_id = recognize_face(frame)

    if student_id is None:
        return jsonify({"recognized": False})

    return jsonify({"recognized": True, "studentId": student_id})

@app.route("/recognize-live", methods=["POST"])
def recognize_live():
    if recognize_face_with_coords is None or detect_all_faces is None:
        return jsonify({
            "recognized": False,
            "faces": [],
            "error": "AI recognition module not available"
        }), 500
    
    try:
        file = request.files.get("frame")
        if not file:
            return jsonify({"error": "No frame received"}), 400

        file_data = file.read()
        if not file_data or len(file_data) == 0:
            return jsonify({
                "recognized": False,
                "faces": [],
                "error": "Empty file received"
            })

        npimg = np.frombuffer(file_data, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({
                "recognized": False,
                "faces": [],
                "error": "Failed to decode image"
            })

        # Detect all faces first
        all_faces = detect_all_faces(frame) or []
        print(f"[AI Server] Detected {len(all_faces)} face(s) in frame")
        
        if not all_faces:
            print(f"[AI Server] No faces detected in frame")
            return jsonify({
                "recognized": False,
                "faces": [],
                "error": "No faces detected"
            })
        
        # Try to recognize the largest face
        # New YOLO module doesn't use threshold parameter in recognize_face_with_coords
        if USE_YOLO:
            student_id, face_box = recognize_face_with_coords(frame)
        else:
            # Old LBPH module uses threshold parameter
            student_id, face_box = recognize_face_with_coords(frame, threshold=70.0)
        
        if student_id is None:
            result = {
                "recognized": False,
                "faces": [{"x": int(f[0]), "y": int(f[1]), "w": int(f[2]), "h": int(f[3])} for f in all_faces] if all_faces else []
            }
            print(f"[AI Server] Face detected but not recognized - check training data or similarity threshold")
            return jsonify(result)

        print(f"[AI Server] ✓ Recognized student: {student_id}")
        return jsonify({
            "recognized": True,
            "studentId": student_id,
            "faceBox": {
                "x": int(face_box[0]),
                "y": int(face_box[1]),
                "w": int(face_box[2]),
                "h": int(face_box[3])
            },
            "faces": [{"x": int(f[0]), "y": int(f[1]), "w": int(f[2]), "h": int(f[3])} for f in all_faces]
        })
    except Exception as e:
        print(f"[AI Server] Error in recognize_live: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "recognized": False,
            "faces": [],
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(port=8000, debug=True)
