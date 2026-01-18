"""
Enhanced Face Recognition Module using YOLOv8-face and ArcFace
Replaces the old Haar + LBPH implementation with more robust detection and recognition
"""

import cv2
import os
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings
import urllib.request
import sys
warnings.filterwarnings('ignore')

# YOLOv8 imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not installed. Install with: pip install ultralytics")

# DeepFace imports
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("[WARNING] deepface not installed. Install with: pip install deepface")

DATASET_DIR = "dataset"
FRAMES_DIR = "frames"  # Backend frames directory
EMBEDDINGS_FILE = "encodings.npy"  # Store embeddings for known students (numpy format)
# YOLOv8-face model for face detection
YOLO_MODEL_PATH = "yolov8n-face.pt"  # YOLOv8-face model specifically for faces
YOLO_MODEL_URL = "https://github.com/derronqi/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt"
FACE_SIZE = (112, 112)  # Standard face size for ArcFace
# YOLO detection confidence thresholds
# Note: 0.95 is too strict - YOLO typically returns 0.3-0.9 for faces
# Using more reasonable thresholds that balance accuracy and detection rate
DETECTION_CONFIDENCE = 0.35  # Increased from 0.15 to reduce false positives
MIN_FACE_CONFIDENCE = 0.45   # Lowered from 0.60 to be more inclusive but still quality
RECOGNITION_THRESHOLD = 0.60  # Sweet spot (higher than 0.75, lower than 0.85)
FRAME_MATCH_PERCENTAGE = 0.25  # If 25% of frames match, mark attendance

os.makedirs(DATASET_DIR, exist_ok=True)


class FaceRecognizer:
    """YOLO + ArcFace based face recognizer"""
    
    def __init__(self):
        self.yolo_model = None
        self._model_logged = False  # Track if we've logged the active model
        self.student_embeddings = {}  # {student_id: aggregated_embedding}
        
        # Load YOLO model if available
        if not YOLO_AVAILABLE:
            print(f"[AI] ⚠ ultralytics not available - install with: pip install ultralytics")
            print(f"[AI] Falling back to RetinaFace (if available)")
        else:
            try:
                model_loaded = False
                abs_model_path = os.path.abspath(YOLO_MODEL_PATH)
                
                # First priority: Check if yolov8n-face.pt exists
                if os.path.exists(abs_model_path):
                    try:
                        print(f"[AI] Loading YOLOv8-face model: {YOLO_MODEL_PATH}")
                        self.yolo_model = YOLO(abs_model_path)
                        print(f"[AI] ✓ Successfully loaded YOLOv8-face model: {YOLO_MODEL_PATH}")
                        model_loaded = True
                    except Exception as e:
                        error_msg = str(e)
                        print(f"[AI] ✗ Failed to load {YOLO_MODEL_PATH}: {error_msg[:200]}")
                        if "StemBlock" in error_msg or "attribute" in error_msg.lower():
                            print(f"[AI]   Model is incompatible with ultralytics version")
                            print(f"[AI]   Attempting to download a fresh model...")
                            try:
                                os.remove(abs_model_path)
                                print(f"[AI]   Removed incompatible model")
                            except:
                                pass
                
                # Second: Auto-download if model doesn't exist
                if not model_loaded:
                    print(f"[AI] ⚠ YOLOv8-face model not found: {abs_model_path}")
                    print(f"[AI] Attempting to auto-download yolov8n-face.pt...")
                    print(f"[AI] Source: {YOLO_MODEL_URL}")
                    print(f"[AI] This may take a few minutes depending on your connection...")
                    
                    try:
                        def progress_hook(count, block_size, total_size):
                            if total_size > 0:
                                percent = min(100, int(count * block_size * 100 / total_size))
                                sys.stdout.write(f"\r[AI] Download progress: {percent}% ({count * block_size / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)")
                                sys.stdout.flush()
                        
                        urllib.request.urlretrieve(YOLO_MODEL_URL, abs_model_path, reporthook=progress_hook)
                        print(f"\n[AI] ✓ Successfully downloaded yolov8n-face.pt")
                        
                        # Try to load the downloaded model
                        try:
                            self.yolo_model = YOLO(abs_model_path)
                            print(f"[AI] ✓ Successfully loaded downloaded YOLOv8-face model")
                            model_loaded = True
                        except Exception as load_err:
                            print(f"[AI] ✗ Failed to load downloaded model: {load_err}")
                            # Remove corrupted download
                            try:
                                os.remove(abs_model_path)
                                print(f"[AI] Removed corrupted download")
                            except:
                                pass
                    except Exception as download_err:
                        print(f"\n[AI] ✗ Auto-download failed: {download_err}")
                        print(f"[AI] Please manually download yolov8n-face.pt:")
                        print(f"[AI]   1. Visit: https://github.com/derronqi/yolov8-face/releases")
                        print(f"[AI]   2. Download: yolov8n-face.pt")
                        print(f"[AI]   3. Place it in: {os.path.dirname(abs_model_path)}")
                
                # Third: Check for any other .pt files as fallback (but prioritize face model)
                if not model_loaded:
                    try:
                        current_dir = os.getcwd()
                        pt_files = [f for f in os.listdir(current_dir) if f.endswith('.pt') and 'face' in f.lower()]
                        if not pt_files:
                            pt_files = [f for f in os.listdir(current_dir) if f.endswith('.pt')]
                        
                        if pt_files:
                            print(f"[AI] Found other .pt files: {pt_files}")
                            for pt_file in pt_files:
                                pt_path = os.path.abspath(pt_file)
                                try:
                                    print(f"[AI] Attempting to load: {pt_path}")
                                    self.yolo_model = YOLO(pt_path)
                                    print(f"[AI] ✓ Successfully loaded YOLO model: {pt_file}")
                                    print(f"[AI]   Note: This may not be optimized for face detection")
                                    model_loaded = True
                                    break
                                except Exception as e:
                                    print(f"[AI] Failed to load {pt_file}: {str(e)[:100]}")
                                    continue
                    except Exception as dir_err:
                        pass
                
                if not model_loaded:
                    print(f"[AI] ⚠ YOLOv8-face model not available")
                    print(f"[AI] System will use DeepFace RetinaFace detection")
                    print(f"[AI] These work well, but YOLOv8-face provides better accuracy")
            except Exception as e:
                print(f"[AI] ⚠ Failed to initialize YOLO: {e}")
                import traceback
                traceback.print_exc()
        
        # Load saved embeddings
        self.load_embeddings()
    
    def detect_faces_yolo(self, frame: np.ndarray, min_conf: float = None) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces using YOLOv8-face.
        Falls back to DeepFace RetinaFace (best), then Haar cascade if YOLO is not available.
        Filters out low-confidence detections based on MIN_FACE_CONFIDENCE.
        Returns list of (x1, y1, x2, y2, confidence) tuples.
        """
        if self.yolo_model is None:
            # Try DeepFace RetinaFace first (best face detector)
            if DEEPFACE_AVAILABLE:
                print("[AI] YOLO model not loaded, using DeepFace RetinaFace detector...")
                return self._detect_faces_retinaface(frame, min_conf)
            # Fallback to nothing if RetinaFace also fails/not available
            print("[AI] YOLO model not loaded, standard detection unavailable.")
            return []
        
        if min_conf is None:
            min_conf = MIN_FACE_CONFIDENCE
        
        try:
            # Use YOLOv8-face for face detection
            # YOLOv8-face is specifically trained for faces and provides better accuracy
            if not self._model_logged and self.yolo_model:
                try:
                    # Log the actual model file being used
                    print(f"[AI] DEBUG: Performing detection using model file: {self.yolo_model.model.pt_path}")
                    self._model_logged = True
                except:
                    print(f"[AI] DEBUG: Performing detection using YOLO model (path check failed)")
            
            results = self.yolo_model(frame, conf=DETECTION_CONFIDENCE, verbose=False)
            detections = []
            all_detections_raw = []  # Store all detections for logging
            
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    all_detections_raw.append(confidence)
                    
                    # Filter based on minimum confidence threshold
                    if confidence >= min_conf:
                        detections.append((int(x1), int(y1), int(x2), int(y2), confidence))
            
            # Log detection statistics for debugging
            if len(all_detections_raw) > 0:
                max_conf = max(all_detections_raw)
                # avg_conf = sum(all_detections_raw) / len(all_detections_raw)
                # print(f"[AI] YOLOv8-face detected {len(all_detections_raw)} faces (max conf: {max_conf:.3f})")
                
                if len(detections) == 0:
                    # Try with even lower threshold if no faces passed
                    # print(f"[AI] No faces passed {min_conf} threshold, trying with lower threshold...")
                    pass
            
            if len(detections) == 0:
                # Optional: Failover to RetinaFace if YOLO finds nothing? 
                # For now, just return empty to be fast
                pass
            
            return detections
        except Exception as e:
            print(f"[AI] Error in YOLO detection: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to RetinaFace on error
            return self._detect_faces_retinaface(frame, min_conf)
    
    def _detect_faces_retinaface(self, frame: np.ndarray, min_conf: float = None) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces using DeepFace's RetinaFace detector (most accurate face detector).
        This uses DeepFace's built-in face detection which is more robust than YOLO for faces.
        Returns list of (x1, y1, x2, y2, confidence) tuples.
        """
        if not DEEPFACE_AVAILABLE:
            return []
        
        if min_conf is None:
            min_conf = 0.5  # Default confidence for RetinaFace
        
        try:
            # Try to use DeepFace's RetinaFace detector via functions module
            try:
                from deepface.commons import functions
                
                # Create a temporary file for DeepFace (works better with file paths)
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    cv2.imwrite(tmp_file.name, frame)
                    tmp_path = tmp_file.name
                
                try:
                            # Use DeepFace's functions to extract faces with RetinaFace
                    detector_backend = "retinaface"
                    
                    # Build detector model
                    # detector_model = functions.build_model(detector_backend)
                    
                    # Detect faces - this returns list of dicts with facial_area
                    img_objs = functions.extract_faces(
                        img_path=tmp_path,
                        detector_backend=detector_backend,
                        grayscale=False,
                        enforce_detection=False,
                        align=False
                    )
                    
                    detections = []
                    # img_objs is a list - each element can be a dict with facial_area or just the face image
                    # If it's a dict, extract coordinates; if it's just an image, we can't get coordinates easily
                    for idx, img_obj in enumerate(img_objs):
                        if isinstance(img_obj, dict):
                            if 'facial_area' in img_obj:
                                region = img_obj['facial_area']
                                x = int(region.get('x', 0))
                                y = int(region.get('y', 0))
                                w = int(region.get('w', 0))
                                h = int(region.get('h', 0))
                                confidence = float(region.get('confidence', 0.95))
                                
                                if confidence >= min_conf and w > 20 and h > 20:
                                    detections.append((x, y, x + w, y + h, confidence))
                    
                    if detections:
                        # conf_str = ", ".join([f"{d[4]:.3f}" for d in detections])
                        # print(f"[AI] RetinaFace detected {len(detections)} face(s)")
                        os.unlink(tmp_path)
                        return detections
                    
                    os.unlink(tmp_path)
                except Exception as e2:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    # print(f"[AI] RetinaFace error: {e2}")
                    
            except ImportError as import_err:
                print(f"[AI] DeepFace functions module not available: {import_err}")
            except Exception as e:
                print(f"[AI] RetinaFace detection error: {e}")
            
            # Fallback to nothing
            return []
            
        except Exception as e:
            print(f"[AI] Error in RetinaFace detection: {e}")
            return []
    

    
    def preprocess_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Preprocess face: Crop with margin, but KEEP ORIGINAL RESOLUTION.
        Do NOT normalize to 0-1 here. Return BGR uint8 image.
        DeepFace handles resizing and normalization better internally.
        
        Args:
            frame: Input frame (BGR)
            bbox: (x1, y1, x2, y2) bounding box
        Returns:
            Cropped face (BGR uint8) or None
        """
        x1, y1, x2, y2 = bbox[:4]
        
        # Crop face with proper checks
        h, w = frame.shape[:2]
        
        # Add a small margin (10%) to include full face features
        margin_w = int((x2 - x1) * 0.1)
        margin_h = int((y2 - y1) * 0.1)
        
        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(w, x2 + margin_w)
        y2 = min(h, y2 + margin_h)
        
        if x2 <= x1 or y2 <= y1:
            return None

        # Crop the face
        face_crop = frame[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return None
            
        # Optimization: Don't resize or normalize here!
        # Pass the high-quality BGR crop directly to DeepFace
        return face_crop
    
    def generate_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate ArcFace embedding for a face image.
        Args:
            face_img: BGR face crop (uint8)
        Returns:
            Embedding vector or None
        """
        if not DEEPFACE_AVAILABLE:
            return None
        
        try:
            # Generate embedding using ArcFace
            # Pass BGR image directly (DeepFace handles BGR/RGB conversion if needed, usually expects RGB or BGR path)
            # We will rely on DeepFace's internal processing which is optimized
            
            embedding_obj = DeepFace.represent(
                img_path=face_img,
                model_name="ArcFace",
                enforce_detection=False,  # Face already detected by YOLO
                align=True,  # Enable alignment (ArcFace requires aligned faces)
                detector_backend="skip"  # Skip detection, providing cropped face
            )
            
            if not embedding_obj:
                return None
                
            # Extract embedding vector
            embedding = np.array(embedding_obj[0]['embedding'], dtype=np.float32)
            
            # Normalize to unit vector for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
        except Exception as e:
            # print(f"[AI] Error generating embedding: {e}")
            return None
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        # Both should already be normalized
        similarity = np.dot(emb1, emb2)
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
    
    def aggregate_embeddings(self, embeddings: List[np.ndarray], method: str = "median") -> Optional[np.ndarray]:
        """
        Aggregate multiple embeddings into one.
        Args:
            embeddings: List of embedding vectors
            method: 'median' or 'average'
        """
        if not embeddings:
            return None
        
        embeddings_array = np.array(embeddings)
        
        if method == "median":
            aggregated = np.median(embeddings_array, axis=0)
        elif method == "average" or method == "mean":
            aggregated = np.mean(embeddings_array, axis=0)
        else:
            aggregated = np.mean(embeddings_array, axis=0)
        
        # Normalize to unit vector
        norm = np.linalg.norm(aggregated)
        if norm > 0:
            aggregated = aggregated / norm
        
        return aggregated
    
    def train_from_frames(self, frames_dir: str, student_id: str) -> bool:
        """
        Train model from frames directory.
        Args:
            frames_dir: Path to frames directory (e.g., "frames/696230c059be41d32ea65c4a")
            student_id: Student ID
        """
        frames_dir = os.path.abspath(frames_dir)
        if not os.path.exists(frames_dir):
            print(f"[AI] ERROR: Frames directory not found: {frames_dir}")
            return False
        
        print(f"[AI] Training from frames: {frames_dir} for student {student_id}")
        
        # Get all frame images
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if not frame_files:
            print(f"[AI] WARNING: No frame images found in {frames_dir}")
            return False
        
        print(f"[AI] Found {len(frame_files)} frame files")
        
        all_embeddings = []
        frames_with_faces = 0
        
        # Process each frame
        processed_count = 0
        for idx, frame_file in enumerate(frame_files):
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            
            if frame is None:
                print(f"[AI] Failed to read frame: {frame_file}")
                continue
            
            processed_count += 1
            
            # Detect faces using YOLO (use higher threshold for training to ensure high-quality reference)
            detections = self.detect_faces_yolo(frame, min_conf=0.5)  # Increased from 0.3 to ensure quality
            
            if not detections:
                if idx < 5 or idx % 10 == 0:  # Log first few and every 10th
                    print(f"[AI] Frame {idx+1}/{len(frame_files)} ({frame_file}): No faces detected")
                continue
            
            # Use largest face (first detection after sorting by area)
            detections_sorted = sorted(
                detections,
                key=lambda d: (d[2] - d[0]) * (d[3] - d[1]),
                reverse=True
            )
            largest_detection = detections_sorted[0]
            
            # Preprocess face
            face_preprocessed = self.preprocess_face(frame, largest_detection[:4])
            
            if face_preprocessed is None:
                print(f"[AI] Frame {idx+1}: Failed to preprocess face")
                continue
            
            # Generate embedding
            embedding = self.generate_embedding(face_preprocessed)
            
            if embedding is not None:
                all_embeddings.append(embedding)
                frames_with_faces += 1
                if frames_with_faces <= 5:  # Log first few successes
                    print(f"[AI] Frame {idx+1}/{len(frame_files)}: Successfully processed face")
            else:
                print(f"[AI] Frame {idx+1}: Failed to generate embedding")
        
        print(f"[AI] Processed {processed_count}/{len(frame_files)} frames")
        
        if not all_embeddings:
            print(f"[AI] ERROR: No faces detected in any of {processed_count} processed frames for {student_id}")
            print(f"[AI] Troubleshooting:")
            print(f"   - Check if frames contain clear, frontal faces")
            print(f"   - Verify YOLO model is loaded (yolov8n-face.pt)")
            print(f"   - Try lowering MIN_FACE_CONFIDENCE in ai_module_yolo.py")
            return False
        
        print(f"[AI] Detected faces in {frames_with_faces}/{len(frame_files)} frames")
        
        # Aggregate embeddings (use median for robustness)
        aggregated_embedding = self.aggregate_embeddings(all_embeddings, method="median")
        
        if aggregated_embedding is None:
            return False
        
        # Store aggregated embedding for this student
        self.student_embeddings[student_id] = aggregated_embedding
        
        # Save embeddings to file
        self.save_embeddings()
        
        print(f"[AI] ✓ Trained student {student_id} with {len(all_embeddings)} face embeddings")
        return True
    
    def recognize_face(self, frame: np.ndarray) -> Optional[str]:
        """
        Recognize face from a single frame.
        Returns student_id if recognized, None otherwise.
        """
        if not self.student_embeddings:
            return None
        
        # Detect faces
        detections = self.detect_faces_yolo(frame)
        
        if not detections:
            return None
        
        # Use largest face
        detections_sorted = sorted(
            detections,
            key=lambda d: (d[2] - d[0]) * (d[3] - d[1]),
            reverse=True
        )
        largest_detection = detections_sorted[0]
        
        # Preprocess face
        face_preprocessed = self.preprocess_face(frame, largest_detection[:4])
        
        if face_preprocessed is None:
            return None
        
        # Generate embedding
        query_embedding = self.generate_embedding(face_preprocessed)
        
        if query_embedding is None:
            return None
        
        # Compare with known embeddings
        best_match = None
        best_similarity = 0.0
        
        for student_id, known_embedding in self.student_embeddings.items():
            similarity = self.cosine_similarity(query_embedding, known_embedding)
            # Log all comparisons for debugging accuracy
            if similarity > 0.5:
                print(f"[AI] Similarity with {student_id}: {similarity:.4f}")
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = student_id
        
        # Check if similarity meets threshold
        # Check if similarity meets threshold
        if best_match and best_similarity >= RECOGNITION_THRESHOLD:
            # print(f"[AI] Recognized {best_match} with similarity {best_similarity:.4f}")
            return best_match
        else:
            # if best_match:
            #     print(f"[AI] Best match {best_match} but similarity {best_similarity:.4f} < threshold {RECOGNITION_THRESHOLD}")
            return None
    
    def recognize_from_multiple_frames(self, frames: List[np.ndarray], min_match_percentage: float = FRAME_MATCH_PERCENTAGE) -> Optional[str]:
        """
        Recognize face from multiple frames using aggregation.
        If X% of frames match a student, return that student_id.
        Args:
            frames: List of frame images
            min_match_percentage: Minimum percentage of frames that must match (default 0.6 = 60%)
        Returns:
            student_id if enough frames match, None otherwise
        """
        if not frames or not self.student_embeddings:
            return None
        
        recognition_results = {}  # {student_id: count of matches}
        
        # Process each frame
        valid_frames = 0
        for frame in frames:
            if frame is None:
                continue
            
            student_id = self.recognize_face(frame)
            valid_frames += 1
            
            if student_id:
                recognition_results[student_id] = recognition_results.get(student_id, 0) + 1
        
        if valid_frames == 0:
            return None
        
        # Find student with highest match percentage
        best_student = None
        best_percentage = 0.0
        
        for student_id, match_count in recognition_results.items():
            percentage = match_count / valid_frames
            if percentage > best_percentage:
                best_percentage = percentage
                best_student = student_id
        
        # Check if best match meets minimum percentage threshold
        if best_student and best_percentage >= min_match_percentage:
            print(f"[AI] ✓ Recognized {best_student} in {recognition_results[best_student]}/{valid_frames} frames ({best_percentage*100:.1f}%)")
            return best_student
        else:
            if best_student:
                print(f"[AI] Best match {best_student} but only {best_percentage*100:.1f}% frames matched (required {min_match_percentage*100:.1f}%)")
            return None
    
    def recognize_face_with_coords(self, frame: np.ndarray) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]], float]:
        """
        Recognize face and return student_id, bounding box, and confidence.
        Returns (student_id, (x, y, w, h), confidence)
        """
        if not self.student_embeddings:
            return None, None, 0.0
        
        # Detect faces
        detections = self.detect_faces_yolo(frame)
        
        if not detections:
            return None, None, 0.0
        
        # Use largest face
        detections_sorted = sorted(
            detections,
            key=lambda d: (d[2] - d[0]) * (d[3] - d[1]),
            reverse=True
        )
        largest_detection = detections_sorted[0]
        x1, y1, x2, y2 = largest_detection[:4]
        bbox = (x1, y1, x2 - x1, y2 - y1)  # Convert to (x, y, w, h)
        
        # Preprocess and recognize
        face_preprocessed = self.preprocess_face(frame, largest_detection[:4])
        
        if face_preprocessed is None:
            return None, None, 0.0
        
        query_embedding = self.generate_embedding(face_preprocessed)
        
        if query_embedding is None:
            return None, None, 0.0
        
        # Compare with known embeddings
        best_match = None
        best_similarity = 0.0
        
        for student_id, known_embedding in self.student_embeddings.items():
            similarity = self.cosine_similarity(query_embedding, known_embedding)
            # Log all comparisons
            if similarity > 0.5:
                print(f"[AI] Similarity with {student_id}: {similarity:.4f}")
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = student_id
        
        # Return best match regardless of threshold, so backend can decide
        return best_match, bbox, float(best_similarity)

    def recognize_all_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Optimized single-pass recognition for multiple faces.
        Returns a list of dictionaries with student_id, confidence, and bounding box.
        """
        results = []
        if not self.student_embeddings:
            # Still detect faces even if none are registered
            detections = self.detect_faces_yolo(frame)
            for d in detections:
                x1, y1, x2, y2 = d[:4]
                results.append({
                    "student_id": None,
                    "confidence": 0.0,
                    "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                    "recognized": False
                })
            return results

        # Detect faces once
        detections = self.detect_faces_yolo(frame)
        print(f"[AI] Batch processing: {len(detections)} faces detected")
        
        for d in detections:
            x1, y1, x2, y2 = d[:4]
            bbox_h = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            
            # Recognition for this face
            face_img = self.preprocess_face(frame, d[:4])
            if face_img is None:
                results.append({
                    "student_id": None,
                    "confidence": 0.0,
                    "bbox": bbox_h,
                    "recognized": False
                })
                continue
                
            emb = self.generate_embedding(face_img)
            if emb is None:
                results.append({
                    "student_id": None,
                    "confidence": 0.0,
                    "bbox": bbox_h,
                    "recognized": False
                })
                continue
            
            best_match = None
            best_similarity = 0.0
            
            for student_id, known_emb in self.student_embeddings.items():
                sim = self.cosine_similarity(emb, known_emb)
                if sim > best_similarity:
                    best_similarity = sim
                    best_match = student_id
            
            is_rec = bool(best_match and best_similarity >= RECOGNITION_THRESHOLD)
            
            results.append({
                "student_id": best_match if is_rec else None,
                "confidence": float(best_similarity),
                "bbox": bbox_h,
                "recognized": is_rec
            })
            
        return results
    
    def detect_all_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect all faces in frame and return bounding boxes as (x, y, w, h).
        """
        detections = self.detect_faces_yolo(frame)
        
        # Convert from (x1, y1, x2, y2) to (x, y, w, h)
        faces = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            faces.append((x1, y1, x2 - x1, y2 - y1))
        
        return faces
    
    def save_embeddings(self):
        """Save student embeddings to file (numpy format)"""
        try:
            np.save(EMBEDDINGS_FILE, self.student_embeddings)
            print(f"[AI] ✓ Saved embeddings for {len(self.student_embeddings)} students to {EMBEDDINGS_FILE}")
        except Exception as e:
            print(f"[AI] Error saving embeddings: {e}")
    
    def load_embeddings(self):
        """Load student embeddings from file (numpy format)"""
        if os.path.exists(EMBEDDINGS_FILE):
            try:
                # Load numpy file, allowing pickle for dictionary structure
                self.student_embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
                print(f"[AI] ✓ Loaded embeddings for {len(self.student_embeddings)} students from {EMBEDDINGS_FILE}")
            except Exception as e:
                print(f"[AI] Error loading embeddings: {e}")
                # Try loading old pickle format as fallback if migration
                try:
                    with open("student_embeddings.pkl", 'rb') as f:
                        self.student_embeddings = pickle.load(f)
                    print(f"[AI] ✓ Loaded legacy pickle embeddings. Will save as npy on next update.")
                    # Immediately save as npy
                    self.save_embeddings()
                except:
                    self.student_embeddings = {}


# Global recognizer instance
_recognizer = None

def get_recognizer():
    """Get or create global recognizer instance"""
    global _recognizer
    if _recognizer is None:
        _recognizer = FaceRecognizer()
    return _recognizer


# Backward compatibility functions
def train_from_frames(frames_dir: str, student_id: str) -> bool:
    """Train model from frames directory"""
    recognizer = get_recognizer()
    return recognizer.train_from_frames(frames_dir, student_id)


def recognize_face(frame: np.ndarray) -> Optional[str]:
    """Recognize face from a single frame"""
    recognizer = get_recognizer()
    return recognizer.recognize_face(frame)


def recognize_face_with_coords(frame: np.ndarray) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]], float]:
    """Recognize face and return student_id, bounding box, and confidence"""
    recognizer = get_recognizer()
    return recognizer.recognize_face_with_coords(frame)

def recognize_all_faces(frame: np.ndarray) -> List[Dict]:
    """Multi-face recognition optimized"""
    recognizer = get_recognizer()
    return recognizer.recognize_all_faces(frame)


def detect_all_faces(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect all faces in frame"""
    recognizer = get_recognizer()
    return recognizer.detect_all_faces(frame)
