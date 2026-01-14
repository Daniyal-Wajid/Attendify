"""
Example Usage Script for YOLOv8-face Recognition System

This script demonstrates how to use the YOLOFaceRecognizer module
to detect faces using YOLOv8 and recognize students using ArcFace embeddings.

Usage:
    python example_yolo_usage.py

Make sure to:
1. Download yolov8n-face.pt from: https://github.com/derronqi/yolov8-face/releases
2. Install requirements: pip install -r requirements_yolo.txt
3. Prepare your student video database
"""

from yolo_face_recognition import YOLOFaceRecognizer, create_sample_database
import os
import json
from pathlib import Path


def load_student_database(recognizer: YOLOFaceRecognizer, database_path: str = "student_database.json"):
    """
    Load student embeddings from a JSON file.
    
    Args:
        recognizer: YOLOFaceRecognizer instance
        database_path: Path to JSON file containing student embeddings
    """
    if not os.path.exists(database_path):
        print(f"[Database] Database file not found: {database_path}")
        return False
    
    try:
        with open(database_path, 'r') as f:
            database = json.load(f)
        
        import numpy as np
        for student_name, embedding_list in database.items():
            embedding = np.array(embedding_list, dtype=np.float32)
            recognizer.add_student_embedding(student_name, embedding)
        
        print(f"[Database] Loaded {len(database)} students from {database_path}")
        return True
    except Exception as e:
        print(f"[Database] Error loading database: {e}")
        return False


def save_student_database(recognizer: YOLOFaceRecognizer, database_path: str = "student_database.json"):
    """
    Save student embeddings to a JSON file.
    
    Args:
        recognizer: YOLOFaceRecognizer instance
        database_path: Path to save JSON file
    """
    try:
        database = {}
        for student_name, embedding in recognizer.known_embeddings.items():
            database[student_name] = embedding.tolist()
        
        with open(database_path, 'w') as f:
            json.dump(database, f, indent=2)
        
        print(f"[Database] Saved {len(database)} students to {database_path}")
        return True
    except Exception as e:
        print(f"[Database] Error saving database: {e}")
        return False


def main():
    """
    Main example function demonstrating the complete pipeline.
    """
    
    print("="*70)
    print("YOLOv8-Face Recognition System - Example Usage")
    print("="*70)
    print()
    
    # ====================================================================
    # Step 1: Initialize the recognizer
    # ====================================================================
    print("[Step 1] Initializing YOLOv8-face recognizer...")
    
    model_path = "yolov8n-face.pt"  # Download from: https://github.com/derronqi/yolov8-face/releases
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n⚠️  WARNING: Model file '{model_path}' not found!")
        print("Please download it from: https://github.com/derronqi/yolov8-face/releases")
        print("Place it in the 'ai' directory.")
        return
    
    recognizer = YOLOFaceRecognizer(
        model_path=model_path,
        detection_confidence=0.95,  # High confidence threshold for accuracy
        recognition_threshold=0.75,  # Cosine similarity threshold
        embedding_model="ArcFace"    # Use ArcFace for embeddings
    )
    
    print("✓ Recognizer initialized successfully!\n")
    
    # ====================================================================
    # Step 2: Build or load student database
    # ====================================================================
    print("[Step 2] Building student database...")
    
    database_file = "student_database.json"
    
    # Try to load existing database
    if load_student_database(recognizer, database_file):
        print("✓ Database loaded from file!\n")
    else:
        # Create database from videos
        print("\n[Step 2a] Database not found. Creating from student videos...")
        print("Please update the 'student_videos' dictionary with your actual video paths.\n")
        
        # Dictionary mapping student names to their video paths
        # UPDATE THESE PATHS WITH YOUR ACTUAL VIDEO FILES
        student_videos = {
            "Alice": "path/to/alice_video.mp4",
            "Bob": "path/to/bob_video.mp4",
            "Charlie": "path/to/charlie_video.mp4",
            "Diana": "path/to/diana_video.mp4",
            "Eve": "path/to/eve_video.mp4",
            "Frank": "path/to/frank_video.mp4",
            "Grace": "path/to/grace_video.mp4",
        }
        
        # Filter out non-existent videos
        existing_videos = {
            name: path for name, path in student_videos.items()
            if os.path.exists(path)
        }
        
        if not existing_videos:
            print("⚠️  No valid video files found in student_videos dictionary.")
            print("Please update the paths and try again.\n")
            print("Skipping database creation for this example...")
        else:
            print(f"Found {len(existing_videos)} valid video files.")
            print("Processing videos to create database...\n")
            
            # Create database from videos
            recognizer = create_sample_database(recognizer, existing_videos)
            
            # Save database for future use
            save_student_database(recognizer, database_file)
            print("✓ Database created and saved!\n")
    
    # Check if database is populated
    if len(recognizer.known_embeddings) == 0:
        print("⚠️  WARNING: No students in database!")
        print("Cannot proceed with recognition without a populated database.")
        return
    
    print(f"✓ Database ready with {len(recognizer.known_embeddings)} students:\n")
    for student_name in recognizer.known_embeddings.keys():
        print(f"  - {student_name}")
    print()
    
    # ====================================================================
    # Step 3: Process query video
    # ====================================================================
    print("[Step 3] Processing query video...")
    
    # UPDATE THIS PATH WITH YOUR QUERY VIDEO
    query_video = "path/to/query_video.mp4"
    
    if not os.path.exists(query_video):
        print(f"\n⚠️  Query video not found: {query_video}")
        print("Please update the 'query_video' variable with an actual video path.\n")
        print("For demonstration, here's what the process would do:")
        print("\n  recognizer.process_video(")
        print("      query_video,")
        print("      max_frames=65,")
        print("      aggregate_method='median',  # or 'average'")
        print("      visualize=True,")
        print("      save_output='output_visualization.mp4'")
        print("  )")
        return
    
    print(f"Processing video: {query_video}\n")
    
    # Process the video
    student_name, similarity, embedding = recognizer.process_video(
        query_video,
        max_frames=65,           # Process up to 65 frames
        aggregate_method="median",  # Use median aggregation (robust to outliers)
        visualize=True,          # Draw bounding boxes on frames
        save_output="output_visualization.mp4"  # Save visualized video
    )
    
    # ====================================================================
    # Step 4: Display results
    # ====================================================================
    print("\n" + "="*70)
    print("RECOGNITION RESULTS")
    print("="*70)
    
    if student_name:
        print(f"\n✓ Student Identified: {student_name}")
        print(f"  Similarity Score: {similarity:.4f}")
        print(f"  Threshold: {recognizer.recognition_threshold:.2f}")
        print(f"  Status: MATCH FOUND")
    else:
        print("\n✗ Student not recognized")
        print(f"  Best Similarity: {similarity:.4f}")
        print(f"  Threshold: {recognizer.recognition_threshold:.2f}")
        print(f"  Status: NO MATCH (below threshold)")
    
    print("\n" + "="*70)
    
    # ====================================================================
    # Optional: Compare with all students for debugging
    # ====================================================================
    if embedding is not None:
        print("\n[Debug] Similarity scores with all known students:")
        print("-" * 70)
        
        import numpy as np
        similarities = []
        for known_name, known_embedding in recognizer.known_embeddings.items():
            sim = recognizer.cosine_similarity(embedding, known_embedding)
            similarities.append((known_name, sim))
            print(f"  {known_name:20s}: {sim:.4f}")
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        print("\nRanked matches:")
        for i, (name, sim) in enumerate(similarities[:3], 1):
            marker = "✓" if sim >= recognizer.recognition_threshold else " "
            print(f"  {i}. {marker} {name:20s}: {sim:.4f}")


def quick_test():
    """
    Quick test function with minimal setup.
    """
    print("\n" + "="*70)
    print("QUICK TEST MODE")
    print("="*70 + "\n")
    
    # Initialize recognizer
    model_path = "yolov8n-face.pt"
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Please download it first.")
        return
    
    recognizer = YOLOFaceRecognizer(
        model_path=model_path,
        detection_confidence=0.95,
        recognition_threshold=0.75
    )
    
    # Test with a single video (assumes you have at least one video)
    test_video = "test_video.mp4"
    
    if os.path.exists(test_video):
        print(f"Testing with video: {test_video}\n")
        
        # Extract frames and detect faces
        frames = recognizer.extract_frames(test_video, max_frames=10)
        
        print(f"Extracted {len(frames)} frames")
        
        # Detect faces in first frame
        if frames:
            print("\nDetecting faces in first frame...")
            detections = recognizer.detect_faces_yolo(frames[0], visualize=True)
            print(f"Found {len(detections)} face(s)")
            
            if detections:
                # Crop and get embedding
                face_crop = recognizer.crop_face(frames[0], detections[0][:4])
                embedding = recognizer.generate_embedding(face_crop)
                
                if embedding is not None:
                    print(f"Generated embedding with shape: {embedding.shape}")
                else:
                    print("Failed to generate embedding")
            else:
                print("No faces detected")
    else:
        print(f"Test video '{test_video}' not found.")
        print("Please provide a video file for testing.")


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_test()
    else:
        main()

