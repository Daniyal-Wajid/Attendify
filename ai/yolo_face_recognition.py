"""
YOLOv8-Face Detection and ArcFace Recognition Module

This module provides robust face detection using YOLOv8-face and recognition
using ArcFace embeddings via DeepFace. It's designed to replace the existing
Haar cascade-based detection with more accurate YOLOv8 detection.

Author: AI Assistant
Date: 2024
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# YOLOv8 imports
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "ultralytics not installed. Install with: pip install ultralytics"
    )

# DeepFace imports
try:
    from deepface import DeepFace
except ImportError:
    raise ImportError(
        "deepface not installed. Install with: pip install deepface"
    )

# For face alignment (optional but recommended)
try:
    from align_faces import align_face  # Optional custom alignment
except ImportError:
    align_face = None


class YOLOFaceRecognizer:
    """
    Face detection and recognition system using YOLOv8-face and ArcFace embeddings.
    
    Attributes:
        yolo_model: YOLOv8-face model for face detection
        detection_confidence: Minimum confidence threshold for face detection
        recognition_threshold: Minimum cosine similarity for recognition (default: 0.75)
        known_embeddings: Dictionary mapping student names to their aggregated embeddings
        embedding_model: DeepFace model name (default: 'ArcFace')
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n-face.pt",
        detection_confidence: float = 0.95,
        recognition_threshold: float = 0.75,
        embedding_model: str = "ArcFace"
    ):
        """
        Initialize the YOLOv8-face recognizer.
        
        Args:
            model_path: Path to YOLOv8-face model file
            detection_confidence: Minimum confidence for face detection (0.0-1.0)
            recognition_threshold: Minimum cosine similarity for recognition (0.0-1.0)
            embedding_model: DeepFace model name ('ArcFace', 'Facenet', etc.)
        """
        self.detection_confidence = detection_confidence
        self.recognition_threshold = recognition_threshold
        self.embedding_model = embedding_model
        self.known_embeddings: Dict[str, np.ndarray] = {}
        
        # Load YOLOv8-face model
        print(f"[YOLO] Loading YOLOv8-face model from {model_path}...")
        try:
            self.yolo_model = YOLO(model_path)
            print("[YOLO] Model loaded successfully!")
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to load YOLOv8 model from {model_path}. "
                f"Download it from: https://github.com/derronqi/yolov8-face/releases"
            ) from e
    
    def extract_frames(self, video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the input video file
            max_frames: Maximum number of frames to extract (None = extract all)
            
        Returns:
            List of frame images (numpy arrays)
        """
        print(f"[Extract] Loading video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames.append(frame)
            frame_count += 1
            
            if max_frames and frame_count >= max_frames:
                break
        
        cap.release()
        print(f"[Extract] Extracted {len(frames)} frames from video")
        return frames
    
    def detect_faces_yolo(
        self,
        frame: np.ndarray,
        visualize: bool = False
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in a frame using YOLOv8-face.
        
        Args:
            frame: Input frame (BGR format)
            visualize: Whether to draw bounding boxes on frame (modifies frame in-place)
            
        Returns:
            List of detections as (x1, y1, x2, y2, confidence) tuples
            Coordinates are in (x_min, y_min, x_max, y_max) format
        """
        # Run YOLOv8 inference
        results = self.yolo_model(frame, conf=self.detection_confidence, verbose=False)
        
        detections = []
        
        # Parse results (YOLOv8 returns results in a list)
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                # Get bounding box coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                # Convert to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                detections.append((x1, y1, x2, y2, float(confidence)))
                
                # Visualize if requested
                if visualize:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Face {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
        
        return detections
    
    def crop_face(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        margin: float = 0.1
    ) -> np.ndarray:
        """
        Crop face from frame with optional margin.
        
        Args:
            frame: Input frame
            bbox: Bounding box as (x1, y1, x2, y2)
            margin: Percentage margin to add around face (0.1 = 10%)
            
        Returns:
            Cropped face image
        """
        x1, y1, x2, y2 = bbox
        
        # Calculate margin
        width = x2 - x1
        height = y2 - y1
        margin_x = int(width * margin)
        margin_y = int(height * margin)
        
        # Expand bounding box with margin
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(frame.shape[1], x2 + margin_x)
        y2 = min(frame.shape[0], y2 + margin_y)
        
        # Crop and return
        face_crop = frame[y1:y2, x1:x2]
        return face_crop
    
    def align_face_optional(self, face_img: np.ndarray) -> np.ndarray:
        """
        Optionally align face using custom alignment function or DeepFace.
        
        Args:
            face_img: Cropped face image
            
        Returns:
            Aligned face image (or original if alignment not available)
        """
        if align_face is not None:
            return align_face(face_img)
        
        # Try DeepFace's built-in alignment
        try:
            # DeepFace can handle alignment internally, but we can also do it here
            # For now, return original - DeepFace will handle alignment during embedding
            return face_img
        except:
            return face_img
    
    def generate_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """
        Generate face embedding using ArcFace via DeepFace.
        
        Args:
            face_img: Face image (BGR format)
            
        Returns:
            512-dimensional embedding vector (or model's default dimension)
        """
        try:
            # DeepFace expects RGB format
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Generate embedding using ArcFace
            embedding_obj = DeepFace.represent(
                img_path=face_rgb,
                model_name=self.embedding_model,
                enforce_detection=False,  # Face is already detected
                align=True  # Enable alignment for better accuracy
            )
            
            # Extract embedding vector
            embedding = embedding_obj[0]['embedding']
            return np.array(embedding, dtype=np.float32)
            
        except Exception as e:
            print(f"[Embedding] Error generating embedding: {e}")
            return None
    
    def aggregate_embeddings(
        self,
        embeddings: List[np.ndarray],
        method: str = "median"
    ) -> np.ndarray:
        """
        Aggregate multiple embeddings into a single representative embedding.
        
        Args:
            embeddings: List of embedding vectors
            method: Aggregation method ('median' or 'average')
            
        Returns:
            Aggregated embedding vector
        """
        if not embeddings:
            raise ValueError("Cannot aggregate empty list of embeddings")
        
        embeddings_array = np.array(embeddings)
        
        if method == "median":
            aggregated = np.median(embeddings_array, axis=0)
        elif method == "average" or method == "mean":
            aggregated = np.mean(embeddings_array, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        # Normalize to unit vector for cosine similarity
        norm = np.linalg.norm(aggregated)
        if norm > 0:
            aggregated = aggregated / norm
        
        return aggregated
    
    def cosine_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        embedding1_norm = embedding1 / norm1
        embedding2_norm = embedding2 / norm2
        
        # Cosine similarity
        similarity = np.dot(embedding1_norm, embedding2_norm)
        
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, similarity))
    
    def add_student_embedding(
        self,
        name: str,
        embedding: np.ndarray
    ):
        """
        Add a known student's embedding to the database.
        
        Args:
            name: Student name/ID
            embedding: Aggregated embedding vector for the student
        """
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        self.known_embeddings[name] = embedding
        print(f"[Database] Added student '{name}' to database")
    
    def recognize_student(
        self,
        query_embedding: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Recognize a student from their embedding by comparing with known embeddings.
        
        Args:
            query_embedding: Embedding vector to identify
            
        Returns:
            Tuple of (student_name, best_similarity_score) or (None, score) if not recognized
        """
        if not self.known_embeddings:
            return None, 0.0
        
        best_match = None
        best_similarity = 0.0
        
        # Compare with all known embeddings
        for student_name, known_embedding in self.known_embeddings.items():
            similarity = self.cosine_similarity(query_embedding, known_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = student_name
        
        # Check if similarity meets threshold
        if best_similarity >= self.recognition_threshold:
            return best_match, best_similarity
        else:
            return None, best_similarity
    
    def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = 65,
        aggregate_method: str = "median",
        visualize: bool = True,
        save_output: Optional[str] = None
    ) -> Tuple[Optional[str], float, np.ndarray]:
        """
        Complete pipeline: Extract frames, detect faces, generate embeddings, and recognize.
        
        Args:
            video_path: Path to input video file
            max_frames: Maximum number of frames to process (default: 65)
            aggregate_method: Method to aggregate embeddings ('median' or 'average')
            visualize: Whether to draw bounding boxes on frames
            save_output: Optional path to save visualized video
            
        Returns:
            Tuple of (student_name, best_similarity, aggregated_embedding)
        """
        print(f"\n{'='*60}")
        print(f"[Pipeline] Starting video processing: {video_path}")
        print(f"{'='*60}\n")
        
        # Step 1: Extract frames from video
        frames = self.extract_frames(video_path, max_frames=max_frames)
        
        if not frames:
            raise ValueError("No frames extracted from video")
        
        # Step 2: Detect faces and collect embeddings
        all_embeddings = []
        frames_with_detections = []
        
        print(f"[Pipeline] Processing {len(frames)} frames for face detection...")
        
        for frame_idx, frame in enumerate(frames):
            # Detect faces in frame
            detections = self.detect_faces_yolo(frame, visualize=visualize)
            
            if not detections:
                continue
            
            # Process largest face (assume main person)
            # Sort by area (largest first)
            detections_sorted = sorted(
                detections,
                key=lambda d: (d[2] - d[0]) * (d[3] - d[1]),
                reverse=True
            )
            
            largest_detection = detections_sorted[0]
            
            # Crop face
            face_crop = self.crop_face(frame, largest_detection[:4], margin=0.1)
            
            if face_crop.size == 0:
                continue
            
            # Optional: Align face
            face_aligned = self.align_face_optional(face_crop)
            
            # Generate embedding
            embedding = self.generate_embedding(face_aligned)
            
            if embedding is not None:
                all_embeddings.append(embedding)
                frames_with_detections.append((frame_idx, frame))
            
            print(f"[Pipeline] Frame {frame_idx + 1}/{len(frames)}: "
                  f"Detected {len(detections)} face(s), collected {len(all_embeddings)} embeddings")
        
        if not all_embeddings:
            print("[Pipeline] No faces detected in video!")
            return None, 0.0, None
        
        print(f"\n[Pipeline] Collected {len(all_embeddings)} embeddings from {len(frames_with_detections)} frames")
        
        # Step 3: Aggregate embeddings
        print(f"[Pipeline] Aggregating embeddings using {aggregate_method} method...")
        aggregated_embedding = self.aggregate_embeddings(all_embeddings, method=aggregate_method)
        print(f"[Pipeline] Aggregated embedding shape: {aggregated_embedding.shape}")
        
        # Step 4: Recognize student
        print(f"[Pipeline] Recognizing student from database ({len(self.known_embeddings)} students)...")
        student_name, best_similarity = self.recognize_student(aggregated_embedding)
        
        # Step 5: Create visualization if requested
        if visualize and save_output:
            self._create_output_video(frames_with_detections, save_output)
        
        return student_name, best_similarity, aggregated_embedding
    
    def _create_output_video(
        self,
        frames_with_detections: List[Tuple[int, np.ndarray]],
        output_path: str,
        fps: int = 10
    ):
        """
        Create output video with visualizations.
        
        Args:
            frames_with_detections: List of (frame_idx, frame) tuples
            output_path: Path to save output video
            fps: Frames per second for output video
        """
        if not frames_with_detections:
            return
        
        h, w = frames_with_detections[0][1].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for _, frame in frames_with_detections:
            out.write(frame)
        
        out.release()
        print(f"[Visualization] Saved output video to: {output_path}")


def create_sample_database(
    recognizer: YOLOFaceRecognizer,
    student_videos: Dict[str, str]
) -> YOLOFaceRecognizer:
    """
    Helper function to create a database of known students from their videos.
    
    Args:
        recognizer: YOLOFaceRecognizer instance
        student_videos: Dictionary mapping student names to their video paths
        
    Returns:
        Recognizer with populated database
    """
    print("\n[Database] Creating student database from videos...")
    
    for student_name, video_path in student_videos.items():
        print(f"\n[Database] Processing video for student: {student_name}")
        _, _, embedding = recognizer.process_video(
            video_path,
            visualize=False,
            max_frames=65
        )
        
        if embedding is not None:
            recognizer.add_student_embedding(student_name, embedding)
    
    print(f"\n[Database] Database created with {len(recognizer.known_embeddings)} students")
    return recognizer


# Example usage and main function
if __name__ == "__main__":
    """
    Example usage of the YOLOv8-face recognition system.
    """
    
    # Initialize recognizer
    recognizer = YOLOFaceRecognizer(
        model_path="yolov8n-face.pt",
        detection_confidence=0.95,
        recognition_threshold=0.75,
        embedding_model="ArcFace"
    )
    
    # Option 1: Load embeddings from saved files (if you have them)
    # For now, we'll create from videos
    
    # Option 2: Create database from student videos
    student_videos = {
        "Student1": "path/to/student1_video.mp4",
        "Student2": "path/to/student2_video.mp4",
        "Student3": "path/to/student3_video.mp4",
        "Student4": "path/to/student4_video.mp4",
        "Student5": "path/to/student5_video.mp4",
        "Student6": "path/to/student6_video.mp4",
        "Student7": "path/to/student7_video.mp4",
    }
    
    # Uncomment to create database:
    # recognizer = create_sample_database(recognizer, student_videos)
    
    # Process query video
    query_video = "path/to/query_video.mp4"
    
    print("\n" + "="*60)
    print("PROCESSING QUERY VIDEO")
    print("="*60 + "\n")
    
    student_name, similarity, embedding = recognizer.process_video(
        query_video,
        max_frames=65,
        aggregate_method="median",
        visualize=True,
        save_output="output_visualization.mp4"
    )
    
    # Print results
    print("\n" + "="*60)
    print("RECOGNITION RESULTS")
    print("="*60)
    
    if student_name:
        print(f"✓ Student Identified: {student_name}")
        print(f"  Similarity Score: {similarity:.4f}")
    else:
        print("✗ Student not recognized")
        print(f"  Best Similarity: {similarity:.4f} (threshold: {recognizer.recognition_threshold})")
    
    print("="*60 + "\n")

