import cv2
import os
import numpy as np

DATASET_DIR = "dataset"
MODEL_FILE = "lbph_model.yml"
LABELS_FILE = "labels.npy"

os.makedirs(DATASET_DIR, exist_ok=True)

# -----------------------------
# Utility: convert to grayscale
# -----------------------------
def _to_gray(frame):
    if frame is None:
        return None
    if len(frame.shape) == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

# -----------------------------
# 1) Extract frames (same)
# -----------------------------
def extract_frames(video_path, student_id, target_frames=65):
    video_path = os.path.abspath(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[AI ERROR] Cannot open video: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("[AI ERROR] Video has no frames")
        cap.release()
        return 0

    frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)

    out_dir = os.path.join(DATASET_DIR, student_id)
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.resize(frame, (640, 480))
        cv2.imwrite(os.path.join(out_dir, f"{saved}.jpg"), frame)
        saved += 1

    cap.release()
    print(f"[AI] Saved {saved} frames for {student_id}")
    return saved

# -----------------------------
# Face detector (Haar cascade)
# -----------------------------
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def _detect_largest_face(gray):
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,  # Lowered from 5 to 3 for better detection
        minSize=(50, 50)  # Lowered from 60x60 to 50x50 for better detection
    )
    if len(faces) == 0:
        return None
    # pick largest face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return (x, y, w, h)

def _preprocess_face(gray, bbox):
    x, y, w, h = bbox
    face = gray[y:y+h, x:x+w]
    # LBPH works well with consistent size
    face = cv2.resize(face, (200, 200))
    # normalize contrast a bit
    face = cv2.equalizeHist(face)
    return face

# -----------------------------
# Build / load label mapping
# -----------------------------
def _load_labels():
    if os.path.exists(LABELS_FILE):
        return np.load(LABELS_FILE, allow_pickle=True).item()
    return {"id_to_label": {}, "label_to_id": {}}

def _save_labels(labels):
    np.save(LABELS_FILE, labels, allow_pickle=True)

def _ensure_label(labels, student_id):
    if student_id in labels["id_to_label"]:
        return labels["id_to_label"][student_id]
    new_label = len(labels["id_to_label"]) + 1
    labels["id_to_label"][student_id] = new_label
    labels["label_to_id"][new_label] = student_id
    return new_label

# -----------------------------
# 2) Train model from dataset
# -----------------------------
def train_model():
    images = []
    labels_list = []

    labels = _load_labels()

    for student_id in os.listdir(DATASET_DIR):
        student_path = os.path.join(DATASET_DIR, student_id)
        if not os.path.isdir(student_path):
            continue

        label = _ensure_label(labels, student_id)

        for img_name in sorted(os.listdir(student_path)):
            img_path = os.path.join(student_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = _to_gray(img)
            bbox = _detect_largest_face(gray)
            if bbox is None:
                continue

            face = _preprocess_face(gray, bbox)
            images.append(face)
            labels_list.append(label)

    if not images:
        print("[AI WARNING] No face data found — training skipped")
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels_list))
    recognizer.save(MODEL_FILE)
    _save_labels(labels)

    print(f"[AI] Training complete ({len(images)} samples)")
    return True

# -----------------------------
# 3) Train from frames directory
# -----------------------------
def train_from_frames(frames_dir, student_id):
    frames_dir = os.path.abspath(frames_dir)
    if not os.path.exists(frames_dir):
        print(f"[AI ERROR] Frames dir not found: {frames_dir}")
        return False

    labels = _load_labels()
    label = _ensure_label(labels, student_id)

    images = []
    labels_list = []

    for img_name in os.listdir(frames_dir):
        img_path = os.path.join(frames_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = _to_gray(img)
        bbox = _detect_largest_face(gray)
        if bbox is None:
            continue

        face = _preprocess_face(gray, bbox)
        images.append(face)
        labels_list.append(label)

    if not images:
        print("[AI WARNING] No face data found — training skipped")
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # If model exists, update; else train fresh
    if os.path.exists(MODEL_FILE):
        recognizer.read(MODEL_FILE)
        recognizer.update(images, np.array(labels_list))
    else:
        recognizer.train(images, np.array(labels_list))

    recognizer.save(MODEL_FILE)
    _save_labels(labels)

    print(f"[AI] Trained {len(images)} faces for {student_id}")
    return True

# -----------------------------
# 4) Recognize face from frame
# -----------------------------
def recognize_face(frame, threshold=80.0):
    """
    LBPH returns a 'confidence' (lower is better).
    threshold ~ 50-80 typical. Tune based on your camera.
    Higher threshold = more lenient recognition
    """
    if frame is None:
        return None
        
    if not os.path.exists(MODEL_FILE) or not os.path.exists(LABELS_FILE):
        return None

    labels = _load_labels()
    if not labels.get("label_to_id"):
        return None

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_FILE)
    except:
        return None

    gray = _to_gray(frame)
    if gray is None:
        return None
        
    bbox = _detect_largest_face(gray)
    if bbox is None:
        return None

    try:
        face = _preprocess_face(gray, bbox)
        pred_label, conf = recognizer.predict(face)
        # lower conf = better match
        if conf <= threshold and pred_label in labels["label_to_id"]:
            student_id = labels["label_to_id"][pred_label]
            print(f"[AI] Recognized student: {student_id} with confidence {conf:.2f}")
            return student_id
        else:
            if pred_label in labels["label_to_id"]:
                print(f"[AI] Confidence {conf:.2f} too high (threshold {threshold})")
    except Exception as e:
        print(f"[AI] Error during prediction: {e}")

    return None

def recognize_face_with_coords(frame, threshold=70.0):
    """
    Recognize face and return student ID with bounding box coordinates.
    Returns (student_id, bbox) or (None, None)
    Note: LBPH returns lower confidence = better match
    Threshold 70 is balanced - not too strict, not too lenient
    """
    if frame is None:
        return None, None
        
    if not os.path.exists(MODEL_FILE):
        return None, None
        
    if not os.path.exists(LABELS_FILE):
        return None, None

    labels = _load_labels()
    if not labels.get("label_to_id") or len(labels["label_to_id"]) == 0:
        return None, None

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_FILE)
    except Exception as e:
        return None, None

    gray = _to_gray(frame)
    if gray is None:
        return None, None
        
    bbox = _detect_largest_face(gray)
    if bbox is None:
        return None, None

    try:
        face = _preprocess_face(gray, bbox)
        pred_label, conf = recognizer.predict(face)
        
        # Lower conf = better match in LBPH
        # Use threshold 70 for balanced accuracy
        # Only accept if confidence is good enough and label exists
        if conf <= threshold and pred_label in labels["label_to_id"]:
            student_id = labels["label_to_id"][pred_label]
            # Additional validation: confidence should be reasonable
            # For LBPH: conf < 50 = excellent, 50-70 = good, 70-100 = fair, >100 = poor
            # Only accept if confidence is below threshold (better match)
            if conf < threshold:
                return student_id, bbox
                
    except Exception as e:
        pass

    return None, None

def detect_all_faces(frame):
    """
    Detect all faces in frame and return their bounding boxes.
    Returns list of (x, y, w, h) tuples.
    """
    if frame is None:
        return []
    
    gray = _to_gray(frame)
    if gray is None:
        return []
        
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,  # Lowered from 5 to 3 for better detection
        minSize=(50, 50)  # Lowered from 60x60 to 50x50 for better detection
    )
    return [tuple(face) for face in faces]