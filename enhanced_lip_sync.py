"""
Enhanced Lip Sync Detection Module
---------------------------------
This module provides advanced lip-sync analysis using both MediaPipe and dlib
for robust facial landmark detection and analysis.

Features:
- Dual detection with MediaPipe and dlib for improved accuracy
- Robust error handling and image processing
- Support for both video files and live camera input
- Detailed lip movement analysis and synchronization metrics
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import warnings
import logging
from pathlib import Path
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Check for required dependencies
try:
    import dlib
    from imutils import face_utils
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    warnings.warn("dlib not available. Some features may be limited.")

class DetectionMethod(Enum):
    """Available face detection methods"""
    MEDIAPIPE = "mediapipe"
    DLIB = "dlib"
    AUTO = "auto"  # Try MediaPipe first, fall back to dlib

class LipSyncAnalyzer:
    """Enhanced lip-sync analysis with robust error handling and multiple detection methods"""
    
    def __init__(
        self,
        method: DetectionMethod = DetectionMethod.AUTO,
        face_detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        shape_predictor_path: str = "shape_predictor_68_face_landmarks.dat"
    ):
        """Initialize the lip-sync analyzer
        
        Args:
            method: Detection method to use (MediaPipe, dlib, or auto)
            face_detection_confidence: Minimum confidence for face detection (0-1)
            tracking_confidence: Minimum confidence for landmark tracking (0-1)
            shape_predictor_path: Path to dlib's shape predictor file
        """
        self.method = method if isinstance(method, DetectionMethod) else DetectionMethod(method.lower())
        self.face_detection_confidence = max(0.1, min(1.0, face_detection_confidence))
        self.tracking_confidence = max(0.1, min(1.0, tracking_confidence))
        self.shape_predictor_path = shape_predictor_path
        
        # Initialize models
        self.face_mesh = None
        self.face_detector = None
        self.predictor = None
        self.active_method = None
        
        self._initialize_models()
    
    def _initialize_models(self) -> bool:
        """Initialize face detection models with error handling
        
        Returns:
            bool: True if at least one model was initialized successfully
        """
        success = False
        
        # Try to initialize MediaPipe if needed
        if self.method in [DetectionMethod.MEDIAPIPE, DetectionMethod.AUTO]:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=self.face_detection_confidence,
                    min_tracking_confidence=self.tracking_confidence
                )
                self.active_method = DetectionMethod.MEDIAPIPE
                logger.info("Initialized MediaPipe face mesh model")
                return True
            except Exception as e:
                logger.warning(f"Failed to initialize MediaPipe: {str(e)}")
                if self.method == DetectionMethod.MEDIAPIPE:
                    # Only fail if MediaPipe was explicitly requested
                    raise
        
        # Try to initialize dlib if needed and available
        if (self.method in [DetectionMethod.DLIB, DetectionMethod.AUTO] and 
            DLIB_AVAILABLE):
            try:
                self.face_detector = dlib.get_frontal_face_detector()
                if os.path.exists(self.shape_predictor_path):
                    self.predictor = dlib.shape_predictor(self.shape_predictor_path)
                    self.active_method = DetectionMethod.DLIB
                    logger.info("Initialized dlib face detector and shape predictor")
                    return True
                else:
                    logger.warning(f"Shape predictor file not found: {self.shape_predictor_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize dlib: {str(e)}")
                if self.method == DetectionMethod.DLIB:
                    # Only fail if dlib was explicitly requested
                    raise
        
        # If we get here, no models were initialized successfully
        if not success:
            error_msg = "Failed to initialize any face detection models"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        return success
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for face detection
        
        Args:
            frame: Input BGR image
            
        Returns:
            Processed frame in the correct format
        """
        if frame is None or frame.size == 0:
            raise ValueError("Empty frame provided")
            
        # Convert to RGB if needed (MediaPipe expects RGB)
        if len(frame.shape) == 2:  # Grayscale
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:  # RGBA
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif frame.shape[2] == 3:  # BGR
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image format with {frame.shape[2]} channels")
            
        return frame_rgb
    
    def detect_lips_mediapipe(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect lips using MediaPipe
        
        Args:
            frame: Input BGR image
            
        Returns:
            Numpy array of lip landmarks or None if not detected
        """
        if self.face_mesh is None:
            logger.warning("MediaPipe model not initialized")
            return None
            
        try:
            # Process the frame
            frame_rgb = self._preprocess_frame(frame)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                # Extract lip landmarks (indices for lips in MediaPipe face mesh)
                # These are the landmark indices for the lips in MediaPipe's 478-point model
                LIP_LANDMARKS = [
                    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,  # Upper lip
                    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,  # Lower lip
                    78, 95, 88, 178, 87, 14, 317, 402, 318, 324  # Lip corners and outer edges
                ]
                
                landmarks = results.multi_face_landmarks[0]
                h, w = frame.shape[:2]
                
                # Extract lip landmarks and convert to pixel coordinates
                lip_points = np.array([
                    (int(landmarks.landmark[i].x * w), 
                     int(landmarks.landmark[i].y * h))
                    for i in LIP_LANDMARKS
                ])
                
                return lip_points
                
        except Exception as e:
            logger.error(f"Error in MediaPipe detection: {str(e)}")
            
        return None
    
    def detect_lips_dlib(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect lips using dlib
        
        Args:
            frame: Input BGR image
            
        Returns:
            Numpy array of lip landmarks or None if not detected
        """
        if self.face_detector is None or self.predictor is None:
            logger.warning("dlib models not initialized")
            return None
            
        try:
            # Convert to grayscale for dlib
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
                
            # Ensure 8-bit format
            if gray.dtype != np.uint8:
                gray = (gray * 255).astype(np.uint8)
                
            # Detect faces
            faces = self.face_detector(gray, 1)
            
            if len(faces) > 0:
                # Get facial landmarks
                shape = self.predictor(gray, faces[0])
                shape = face_utils.shape_to_np(shape)
                
                # Lip landmarks for 68-point model (indices 48-67)
                lip_points = shape[48:68]
                
                return lip_points
                
        except Exception as e:
            logger.error(f"Error in dlib detection: {str(e)}")
            
        return None
    
    def detect_lips(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect lips using the active method
        
        Args:
            frame: Input BGR image
            
        Returns:
            Numpy array of lip landmarks or None if not detected
        """
        if self.active_method == DetectionMethod.MEDIAPIPE:
            return self.detect_lips_mediapipe(frame)
        elif self.active_method == DetectionMethod.DLIB:
            return self.detect_lips_dlib(frame)
        else:
            # Auto-detect: Try MediaPipe first, fall back to dlib
            result = self.detect_lips_mediapipe(frame)
            if result is not None:
                return result
            return self.detect_lips_dlib(frame)
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """Analyze a single frame for lip movement
        
        Args:
            frame: Input BGR image
            
        Returns:
            Dictionary containing analysis results
        """
        result = {
            'lip_landmarks': None,
            'lip_width': 0,
            'lip_height': 0,
            'mouth_open': False,
            'detection_method': str(self.active_method) if self.active_method else None,
            'error': None
        }
        
        try:
            # Detect lips
            lip_points = self.detect_lips(frame)
            
            if lip_points is not None and len(lip_points) > 0:
                result['lip_landmarks'] = lip_points
                
                # Calculate basic lip metrics
                x_coords = lip_points[:, 0]
                y_coords = lip_points[:, 1]
                
                # Lip width (horizontal distance between corners)
                lip_width = np.max(x_coords) - np.min(x_coords)
                
                # Lip height (vertical distance between upper and lower lips)
                lip_height = np.max(y_coords) - np.min(y_coords)
                
                # Simple mouth open/closed detection
                mouth_open = lip_height > (lip_width * 0.15)  # Adjust threshold as needed
                
                result.update({
                    'lip_width': float(lip_width),
                    'lip_height': float(lip_height),
                    'mouth_open': bool(mouth_open)
                })
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error analyzing frame: {str(e)}")
            
        return result
    
    def process_video(self, video_path: Union[str, int], output_path: Optional[str] = None) -> Dict:
        """Process a video file or camera stream
        
        Args:
            video_path: Path to video file or camera index
            output_path: Optional path to save processed video
            
        Returns:
            Dictionary with analysis results
        """
        # Initialize video capture
        if isinstance(video_path, (int, str)):
            cap = cv2.VideoCapture(video_path)
        else:
            raise ValueError("video_path must be a file path or camera index")
            
        if not cap.isOpened():
            raise IOError(f"Could not open video source: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        results = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Analyze frame
            result = self.analyze_frame(frame)
            result['frame_idx'] = frame_idx
            result['timestamp'] = frame_idx / fps if fps > 0 else 0
            results.append(result)
            
            # Visualize results (optional)
            if writer is not None or output_path is None:
                vis_frame = self.visualize_results(frame.copy(), result)
                
                if writer is not None:
                    writer.write(vis_frame)
                
                # Display frame (for testing)
                if output_path is None:
                    cv2.imshow('Lip Sync Analysis', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            frame_idx += 1
            
            # Progress update
            if frame_count > 0 and frame_idx % 10 == 0:
                progress = (frame_idx / frame_count) * 100
                logger.info(f"Processed {frame_idx}/{frame_count} frames ({progress:.1f}%)")
        
        # Clean up
        cap.release()
        if writer is not None:
            writer.release()
            
        if output_path is None:
            cv2.destroyAllWindows()
        
        return {
            'fps': fps,
            'frame_count': frame_idx,
            'width': width,
            'height': height,
            'results': results,
            'detection_method': str(self.active_method)
        }
    
    def visualize_results(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Visualize lip detection results on a frame
        
        Args:
            frame: Input BGR image
            result: Analysis result from analyze_frame()
            
        Returns:
            Frame with visualization
        """
        # Draw lip landmarks
        if result['lip_landmarks'] is not None:
            for (x, y) in result['lip_landmarks']:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
            
            # Draw mouth open/closed status
            status = "OPEN" if result.get('mouth_open', False) else "CLOSED"
            color = (0, 255, 0) if status == "OPEN" else (0, 0, 255)
            cv2.putText(frame, f"Mouth: {status}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw detection method
            method = result.get('detection_method', 'unknown')
            cv2.putText(frame, f"Method: {method}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Lip Sync Analysis')
    parser.add_argument('input', help='Input video file or camera index')
    parser.add_argument('-o', '--output', help='Output video file (optional)')
    parser.add_argument('--method', default='auto', 
                       choices=['auto', 'mediapipe', 'dlib'],
                       help='Detection method to use')
    parser.add_argument('--shape-predictor', default='shape_predictor_68_face_landmarks.dat',
                       help='Path to dlib shape predictor file')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    try:
        analyzer = LipSyncAnalyzer(
            method=DetectionMethod(args.method.upper()),
            shape_predictor_path=args.shape_predictor
        )
        
        # Check if input is a file or camera index
        try:
            video_input = int(args.input)
        except ValueError:
            video_input = args.input
            if not os.path.isfile(video_input):
                raise FileNotFoundError(f"Input file not found: {video_input}")
        
        # Process video
        results = analyzer.process_video(video_input, args.output)
        
        # Print summary
        print("\n=== Analysis Complete ===")
        print(f"Frames processed: {results['frame_count']}")
        print(f"Detection method: {results['detection_method']}")
        print(f"Average FPS: {results['fps']:.2f}")
        
        if args.output:
            print(f"\nOutput saved to: {os.path.abspath(args.output)}")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())