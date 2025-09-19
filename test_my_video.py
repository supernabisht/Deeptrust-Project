"""
Enhanced video analysis script for deepfake detection
"""
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
import os

def print_debug(message):
    """Print debug messages with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

class VideoAnalyzer:
    """Enhanced video analyzer with detailed face detection."""
    
    def __init__(self):
        """Initialize the video analyzer with face detection models."""
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # 1 for longer range faces
            min_detection_confidence=0.1  # Lowered to detect more faces
        )
        
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1,
            static_image_mode=False
        )
        
        self.lip_landmarks = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 408,  # Outer lips
            146, 91, 181, 84, 17, 314, 405, 321, 375, 291  # Inner lips
        ]
    
    def analyze_video(self, video_path, output_path=None):
        """Analyze video and return detailed face detection results."""
        print(f"Analyzing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds")
        
        # Initialize variables
        frame_count = 0
        face_detected_frames = 0
        face_confidence_scores = []
        face_sizes = []
        
        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 5th frame for efficiency
            if frame_count % 5 != 0:
                frame_count += 1
                continue
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = self.face_detection.process(frame_rgb)
            
            if results.detections:
                for detection in results.detections:
                    confidence = detection.score[0]
                    face_confidence_scores.append(confidence)
                    
                    # Get face bounding box
                    bbox = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), \
                                int(bbox.width * iw), int(bbox.height * ih)
                    
                    # Calculate face size as percentage of frame
                    face_area = (w * h) / (iw * ih)
                    face_sizes.append(face_area)
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f'Conf: {confidence:.2f}', (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # If we have a reasonably sized face, try to get landmarks
                    if face_area > 0.002 and confidence > 0.2:
                        face_detected_frames += 1
            
            # Show frame with detections
            if output_path:
                cv2.imwrite(f"{output_path}/frame_{frame_count:04d}.jpg", frame)
            
            frame_count += 1
            
            # Show progress
            if frame_count % 10 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")
        
        cap.release()
        
        # Calculate statistics
        avg_confidence = np.mean(face_confidence_scores) if face_confidence_scores else 0
        avg_face_size = np.mean(face_sizes) * 100 if face_sizes else 0
        face_detection_rate = face_detected_frames / (frame_count / 5) if frame_count > 0 else 0
        
        print("\n=== Analysis Results ===")
        print(f"Total frames processed: {frame_count}")
        print(f"Frames with faces: {face_detected_frames} ({face_detection_rate*100:.1f}%)")
        print(f"Average face confidence: {avg_confidence:.2f}")
        print(f"Average face size: {avg_face_size:.2f}% of frame")
        
        return {
            'total_frames': total_frames,
            'frames_analyzed': frame_count,
            'face_detected_frames': face_detected_frames,
            'face_detection_rate': face_detection_rate,
            'avg_face_confidence': float(avg_confidence),
            'avg_face_size_percent': float(avg_face_size)
        }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_my_video.py <video_path> [output_directory]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    analyzer = VideoAnalyzer()
    results = analyzer.analyze_video(video_path, output_dir)
    
    if results:
        print("\n=== Final Assessment ===")
        if results['face_detection_rate'] > 0.3:
            print("✅ This video appears to be AUTHENTIC (high face detection rate)")
        elif results['face_detection_rate'] > 0.1:
            print("ℹ️ This video might be authentic (moderate face detection rate)")
        else:
            print("❓ Unable to determine authenticity (low face detection rate)")
        
        print(f"Confidence: {results['face_detection_rate']*100:.1f}% based on face detection")
    else:
        print("Analysis failed. Please check the video file and try again.")
