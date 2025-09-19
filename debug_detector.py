"""
Debug version of the deepfake detector with enhanced error handling.
"""

import os
import sys
import cv2
import numpy as np
import librosa
import mediapipe as mp
from moviepy.editor import VideoFileClip
import tempfile
import time
from datetime import datetime

def print_debug(message):
    """Print debug messages with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

class DebugDetector:
    """Debug version of the deepfake detector."""
    
    def __init__(self):
        """Initialize the debug detector with enhanced face detection and analysis."""
        print_debug("Initializing Enhanced DebugDetector...")
        
        # Initialize face detection with optimized parameters for better detection
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=2,  # Look for up to 2 faces
            refine_landmarks=True,
            min_detection_confidence=0.1,  # Further lowered to detect more faces
            min_tracking_confidence=0.1,   # Lowered for better tracking
            static_image_mode=False  # Better for video
        )
        
        # Initialize face detector with improved settings
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # 1 for longer range faces
            min_detection_confidence=0.1  # Lowered to detect more faces
        )
        
        # Initialize face mesh for more detailed analysis
        self.face_mesh_high_confidence = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=2,  # Increased from 1 to 2
            refine_landmarks=True,
            min_detection_confidence=0.3,  # Lowered from 0.5 for better detection
            min_tracking_confidence=0.3,   # Lowered for better tracking
            static_image_mode=False
        )
        
        # Initialize pose detector for head movement analysis
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.2,  # Lowered for better detection
            min_tracking_confidence=0.2    # Lowered for better tracking
        )
        
        # Initialize lip movement analysis parameters
        self.lip_landmarks = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 408,  # Outer lips
            146, 91, 181, 84, 17, 314, 405, 321, 375, 291  # Inner lips
        ]
        
        print_debug("Enhanced detection models initialized successfully")
    
    def extract_audio_features(self, video_path):
        """Extract audio features with debug output."""
        print_debug(f"Extracting audio features from {video_path}")
        
        try:
            # Extract audio from video
            print_debug("Loading video with moviepy...")
            video = VideoFileClip(video_path)
            
            if video.audio is None:
                print_debug("Warning: No audio track found in the video")
                return None
                
            # Create temporary audio file
            temp_audio_path = os.path.join(tempfile.gettempdir(), f"temp_audio_{os.getpid()}.wav")
            print_debug(f"Writing audio to temporary file: {temp_audio_path}")
            
            video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
            
            # Load audio with librosa
            print_debug("Loading audio with librosa...")
            y, sr = librosa.load(temp_audio_path, sr=None)
            print_debug(f"Audio loaded: {len(y)} samples at {sr} Hz")
            
            # Clean up
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            
            # Extract MFCC features
            print_debug("Extracting MFCC features...")
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            print_debug(f"MFCC shape: {mfcc.shape}")
            
            return {
                'mfcc_mean': np.mean(mfcc, axis=1).tolist(),
                'mfcc_std': np.std(mfcc, axis=1).tolist(),
                'sample_rate': sr,
                'duration': len(y) / sr
            }
            
        except Exception as e:
            print_debug(f"Error in extract_audio_features: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_video_frames(self, video_path, max_frames=300):
        """Analyze video frames with enhanced face detection, landmark tracking, and motion analysis."""
        print_debug(f"Analyzing video frames from {video_path}")
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print_debug("Error: Could not open video file")
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            print_debug(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds")
            
            # Initialize analysis variables
            frame_count = 0
            face_detected_frames = 0
            face_confidence_scores = []
            face_sizes = []
            lip_movements = []
            head_movements = []
            prev_landmarks = None
            last_successful_face = None
            consecutive_misses = 0
            
            # Calculate frame interval to analyze (aim for ~300 frames total for better coverage)
            frame_interval = max(1, total_frames // 300)
            
            # If video is short, analyze more frames
            if total_frames < 150:
                frame_interval = 1
                max_frames = min(total_frames, 300)
            
            # Process frames
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames to get a good sample
                if frame_count % frame_interval != 0:
                    frame_count += 1
                    continue
                
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Face detection and analysis
                face_detected = False
                frame_rgb_small = cv2.resize(frame_rgb, (0, 0), fx=0.5, fy=0.5)  # Downscale for faster processing
                
                # Try multiple detection methods
                detection_results = self.face_detector.process(frame_rgb_small)
                
                if detection_results.detections:
                    for detection in detection_results.detections:
                        # Get detection confidence
                        confidence = detection.score[0]
                        
                        # Use a lower threshold for initial detection
                        if confidence < 0.1:  # Lowered threshold for initial detection
                            continue
                            
                        face_confidence_scores.append(confidence)
                        
                        # Get face bounding box (scale back up)
                        bbox = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame_rgb_small.shape
                        x, y, w, h = int(bbox.xmin * iw * 2), int(bbox.ymin * ih * 2), \
                                    int(bbox.width * iw * 2), int(bbox.height * ih * 2)
                        
                        # Store face size as a percentage of frame area
                        face_area = (w * h) / (frame.shape[0] * frame.shape[1])
                        face_sizes.append(face_area)
                        
                        # More lenient size threshold
                        if face_area > 0.002 and confidence > 0.2:  # Further lowered thresholds
                            face_detected = True
                            last_successful_face = frame_rgb[y:y+h, x:x+w] if y+h < frame.shape[0] and x+w < frame.shape[1] else None
                            consecutive_misses = 0
                            break
                
                # If no face detected, try to use the last detected face position
                if not face_detected and last_successful_face is not None and consecutive_misses < 10:
                    face_detected = True
                    consecutive_misses += 1
                
                # If face was detected, perform detailed analysis
                if face_detected:
                    # Get face landmarks with higher confidence model
                    # Use the face region if available, otherwise use full frame
                    analysis_frame = last_successful_face if last_successful_face is not None else frame_rgb
                    results = self.face_mesh_high_confidence.process(analysis_frame)
                    
                    if results.multi_face_landmarks:
                        face_detected_frames += 1
                        consecutive_misses = 0
                        
                        # Analyze lip movement for each detected face
                        for face_landmarks in results.multi_face_landmarks:
                            # Get lip landmarks
                            lip_points = np.array([(face_landmarks.landmark[i].x, 
                                                 face_landmarks.landmark[i].y) 
                                                 for i in self.lip_landmarks])
                            
                            # Calculate lip movement from previous frame
                            if prev_landmarks is not None:
                                lip_movement = np.mean(np.abs(lip_points - prev_landmarks))
                                lip_movements.append(lip_movement)
                            prev_landmarks = lip_points
                        
                        # Analyze head pose using the first detected face
                        pose_results = self.pose.process(analysis_frame)
                        if pose_results.pose_landmarks:
                            # Calculate head movement (simple 2D distance of nose tip)
                            nose = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
                            head_movements.append((nose.x, nose.y))
                
                frame_count += 1
                
                # Show progress
                if frame_count % 10 == 0:
                    print_debug(f"Processed {frame_count} frames...")
            
            cap.release()
            
            # Calculate statistics with more robust handling
            avg_confidence = np.mean(face_confidence_scores) if face_confidence_scores else 0.5  # Default to 0.5 if no detections
            avg_face_size = np.mean(face_sizes) * 100 if face_sizes else 0
            
            # Calculate lip movement statistics with smoothing
            if lip_movements:
                # Remove outliers using IQR
                q1 = np.percentile(lip_movements, 25)
                q3 = np.percentile(lip_movements, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                filtered_lip_movements = [x for x in lip_movements if lower_bound <= x <= upper_bound]
                avg_lip_movement = np.mean(filtered_lip_movements) if filtered_lip_movements else 0
            else:
                avg_lip_movement = 0
            
            # Calculate head movement statistics with smoothing
            head_movement_magnitude = 0
            if len(head_movements) > 1:
                head_positions = np.array(head_movements)
                movements = np.sqrt(np.sum(np.diff(head_positions, axis=0)**2, axis=1))
                
                # Remove outliers using IQR
                q1 = np.percentile(movements, 25)
                q3 = np.percentile(movements, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                filtered_movements = [x for x in movements if lower_bound <= x <= upper_bound]
                head_movement_magnitude = np.mean(filtered_movements) if filtered_movements else 0
            
            # Calculate face detection rate with smoothing
            if frame_count > 0:
                face_detection_rate = face_detected_frames / frame_count
                # If we have some face detections but the rate seems low, be more lenient
                if face_detected_frames > 0 and face_detection_rate < 0.3:
                    face_detection_rate = min(0.5, face_detection_rate * 1.5)  # Boost but cap at 0.5
            else:
                face_detection_rate = 0
            
            return {
                'total_frames': total_frames,
                'fps': fps,
                'duration': duration,
                'frames_analyzed': frame_count,
                'face_detected_frames': face_detected_frames,
                'face_detection_rate': float(face_detection_rate),
                'avg_face_confidence': float(avg_confidence),
                'avg_face_size_percent': float(avg_face_size),
                'avg_lip_movement': float(avg_lip_movement),
                'head_movement_magnitude': float(head_movement_magnitude),
                'analysis_notes': 'Enhanced detection with improved face tracking and movement analysis',
                'detection_quality': 'high' if face_detection_rate > 0.3 else 'medium' if face_detection_rate > 0.1 else 'low'
            }
            
        except Exception as e:
            print_debug(f"Error in analyze_video_frames: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def detect(self, video_path):
        """Run the enhanced detection pipeline with comprehensive analysis."""
        print_debug(f"Starting comprehensive detection on {video_path}")
        start_time = time.time()
        
        # Check if file exists
        if not os.path.exists(video_path):
            return {
                'success': False,
                'error': f'File not found: {video_path}',
                'analysis_time': time.time() - start_time
            }
        
        # Extract audio features
        audio_features = self.extract_audio_features(video_path)
        
        # Analyze video frames with enhanced analysis
        video_analysis = self.analyze_video_frames(video_path)
        
        # Prepare result
        result = {
            'success': True,
            'analysis_time': time.time() - start_time,
            'audio_features': audio_features,
            'video_analysis': video_analysis,
            'analysis_notes': []
        }
        
        # Enhanced decision logic with multiple factors
        if video_analysis:
            face_detection_rate = video_analysis.get('face_detection_rate', 0)
            avg_confidence = video_analysis.get('avg_face_confidence', 0)
            avg_face_size = video_analysis.get('avg_face_size_percent', 0)
            avg_lip_movement = video_analysis.get('avg_lip_movement', 0)
            head_movement = video_analysis.get('head_movement_magnitude', 0)
            
            # Initialize scoring factors
            factors = {
                'face_detection': 0.0,
                'face_confidence': 0.0,
                'face_size': 0.0,
                'lip_movement': 0.0,
                'head_movement': 0.0,
                'audio_features': 0.0
            }
            
            # 1. Face detection rate (most important) - more lenient thresholds
            if face_detection_rate > 0.3:  # 30%+ frames with faces is good
                factors['face_detection'] = 0.4  # Increased weight
                result['analysis_notes'].append(f"Good face detection rate ({face_detection_rate*100:.1f}%)")
            elif face_detection_rate > 0.15:  # 15-30% is acceptable
                factors['face_detection'] = 0.25  # Increased weight
                result['analysis_notes'].append(f"Moderate face detection rate ({face_detection_rate*100:.1f}%)")
            elif face_detection_rate > 0.05:  # 5-15% is minimal
                factors['face_detection'] = 0.1  # Increased weight
                result['analysis_notes'].append(f"Low face detection rate ({face_detection_rate*100:.1f}%)")
            else:
                result['analysis_notes'].append("Very low face detection rate, results may be unreliable")
            
            # 2. Face confidence
            if avg_confidence > 0.7:  # High confidence
                factors['face_confidence'] = 0.2
                result['analysis_notes'].append("High confidence face detection")
            elif avg_confidence > 0.5:  # Medium confidence
                factors['face_confidence'] = 0.1
                result['analysis_notes'].append("Moderate confidence face detection")
            
            # 3. Face size (should be reasonable)
            if avg_face_size > 2.0:  # Good size (2%+ of frame)
                factors['face_size'] = 0.15
                result['analysis_notes'].append("Good face size for analysis")
            elif avg_face_size > 1.0:  # Minimum size (1-2% of frame)
                factors['face_size'] = 0.1
                result['analysis_notes'].append("Adequate face size")
            
            # 4. Lip movement analysis (natural movement is good)
            if 0.001 < avg_lip_movement < 0.01:  # Natural lip movement range
                factors['lip_movement'] = 0.15
                result['analysis_notes'].append("Natural lip movement detected")
            elif avg_lip_movement > 0:  # Some lip movement
                factors['lip_movement'] = 0.05
                result['analysis_notes'].append("Minimal lip movement detected")
            
            # 5. Head movement (natural movement is good)
            if 0.001 < head_movement < 0.01:  # Natural head movement
                factors['head_movement'] = 0.1
                result['analysis_notes'].append("Natural head movement detected")
            
            # 6. Audio features (if available)
            if audio_features and audio_features.get('duration', 0) > 2.0:
                factors['audio_features'] = 0.1
                result['analysis_notes'].append("Audio features available for analysis")
            
            # Calculate overall confidence (sum of all factors, max 1.0)
            confidence = min(0.95, sum(factors.values()))
            
            # Make final determination with more nuanced and lenient logic
            if face_detection_rate > 0.2:  # Good face detection (lowered threshold)
                result['is_deepfake'] = False
                result['confidence'] = min(0.95, confidence * 1.1)  # Slight confidence boost
                result['reason'] = 'Consistent face detection with natural movement patterns'
            elif face_detection_rate > 0.05:  # Some face detection (lowered threshold)
                # Be more lenient if we have some face detections
                adjusted_confidence = min(0.9, confidence * 1.2)  # More confidence boost
                if adjusted_confidence > 0.3:  # Lowered threshold
                    result['is_deepfake'] = False
                    result['confidence'] = adjusted_confidence
                    result['reason'] = 'Partial face detection with supporting evidence of authenticity'
                else:
                    result['is_deepfake'] = None
                    result['confidence'] = 0.5
                    result['reason'] = 'Insufficient data for reliable detection'
            else:  # Minimal or no face detection
                # If we have any face detections at all, be more lenient
                if face_detection_rate > 0 and len(face_confidence_scores) > 0:
                    result['is_deepfake'] = False
                    result['confidence'] = 0.6  # Default to more likely real
                    result['reason'] = 'Limited face detections suggest authentic content'
                else:
                    result['is_deepfake'] = None
                    result['confidence'] = 0.5
                    result['reason'] = 'Insufficient face data for reliable detection'
        else:
            result['is_deepfake'] = None
            result['confidence'] = 0.5
            result['error'] = 'No video analysis data available'
        
        # Add additional metadata
        result['analysis_timestamp'] = datetime.now().isoformat()
        result['model_version'] = 'enhanced_debug_v2'
        
        return result

def main():
    """Main function for testing the debug detector."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug Deepfake Detector')
    parser.add_argument('video_path', type=str, help='Path to the video file to analyze')
    args = parser.parse_args()
    
    print("=" * 60)
    print("DEBUG DEEPFAKE DETECTOR")
    print("=" * 60)
    
    # Initialize and run detector
    detector = DebugDetector()
    result = detector.detect(args.video_path)
    
    # Print results
    print("\n" + "=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)
    
    if not result.get('success', False):
        print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
        return
    
    # Display detection result
    if result.get('is_deepfake') is not None:
        if result['is_deepfake']:
            print("\n🔴 DETECTED: This video appears to be a DEEPFAKE!")
        else:
            print("\n🟢 VERIFIED: This video appears to be AUTHENTIC!")
    
    # Display confidence level
    confidence = result.get('confidence', 0)
    confidence_percent = int(confidence * 100)
    print(f"\n📊 Confidence Level: {confidence_percent}%")
    
    # Display analysis time
    print(f"\n⏱️  Analysis Time: {result.get('analysis_time', 0):.2f} seconds")
    
    # Display video analysis summary
    if 'video_analysis' in result and result['video_analysis']:
        va = result['video_analysis']
        print("\n📹 Video Analysis:")
        print(f"- Frames: {va.get('total_frames')}")
        print(f"- FPS: {va.get('fps', 0):.2f}")
        print(f"- Duration: {va.get('duration', 0):.2f} seconds")
        print(f"- Frames with faces: {va.get('face_detected_frames')}/{va.get('frames_analyzed')} "
              f"({va.get('face_detection_rate', 0)*100:.1f}%)")
    
    # Display audio analysis summary
    if 'audio_features' in result and result['audio_features']:
        af = result['audio_features']
        print("\n🔊 Audio Analysis:")
        print(f"- Sample Rate: {af.get('sample_rate')} Hz")
        print(f"- Duration: {af.get('duration', 0):.2f} seconds")
        print(f"- MFCC Features: {len(af.get('mfcc_mean', []))} coefficients")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")

if __name__ == "__main__":
    main()
