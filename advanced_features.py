import cv2
import numpy as np
import librosa
import dlib
from scipy.spatial import distance
from scipy import signal
import mediapipe as mp

class AdvancedFeatureExtractor:
    def __init__(self):
        # Initialize face detector and landmark predictor
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye aspect ratio constants
        self.EYE_AR_THRESH = 0.2
        self.EYE_AR_CONSEC_FRAMES = 3
        
    def extract_eye_blink_features(self, frame):
        """Extract eye blink rate and duration"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        
        if len(faces) == 0:
            return {"blink_rate": 0, "avg_blink_duration": 0, "eye_aspect_ratio": 0}
        
        landmarks = self.landmark_predictor(gray, faces[0])
        
        # Extract eye landmarks
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        
        # Calculate eye aspect ratio
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        # Simple blink detection (you can enhance this with temporal analysis)
        blink = 1 if ear < self.EYE_AR_THRESH else 0
        
        return {
            "blink_rate": 1 if blink else 0,  # per frame
            "avg_blink_duration": 1 if blink else 0,  # in frames
            "eye_aspect_ratio": ear
        }
    
    @staticmethod
    def eye_aspect_ratio(eye):
        """Calculate eye aspect ratio"""
        # Compute the euclidean distances between the vertical eye landmarks
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        
        # Compute the euclidean distance between the horizontal eye landmarks
        C = distance.euclidean(eye[0], eye[3])
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def extract_facial_movement_features(self, frame):
        """Extract facial movement and expression features"""
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return {
                "facial_movement": 0,
                "expression_intensity": 0,
                "facial_symmetry": 1.0
            }
        
        # Get face landmarks
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Calculate facial symmetry
        left_eye = np.array([landmarks[33].x, landmarks[33].y])
        right_eye = np.array([landmarks[263].x, landmarks[263].y])
        nose_tip = np.array([landmarks[1].x, landmarks[1].y])
        
        # Calculate symmetry score (0-1, 1 is perfectly symmetrical)
        left_dist = np.linalg.norm(left_eye - nose_tip)
        right_dist = np.linalg.norm(right_eye - nose_tip)
        symmetry = 1 - (abs(left_dist - right_dist) / max(left_dist, right_dist))
        
        return {
            "facial_movement": 0,  # Would need temporal analysis
            "expression_intensity": 0,  # Would need baseline comparison
            "facial_symmetry": symmetry
        }
    
    def extract_audio_video_sync(self, audio_features, video_features, fps):
        """Calculate audio-visual synchronization metrics"""
        if not audio_features or not video_features:
            return {"av_sync_correlation": 0, "av_offset_ms": 0}
        
        # Simple cross-correlation between audio and video features
        audio_signal = np.array(audio_features)
        video_signal = np.array(video_features)
        
        # Ensure same length
        min_len = min(len(audio_signal), len(video_signal))
        audio_signal = audio_signal[:min_len]
        video_signal = video_signal[:min_len]
        
        # Calculate cross-correlation
        correlation = np.correlate(audio_signal, video_signal, mode='valid')[0]
        
        # Normalize correlation
        norm = np.sqrt(np.sum(audio_signal**2) * np.sum(video_signal**2))
        if norm > 0:
            correlation /= norm
        
        return {
            "av_sync_correlation": float(correlation),
            "av_offset_ms": 0  # Would need more sophisticated calculation
        }
    
    def extract_heart_rate_variability(self, frame_sequence):
        """Extract heart rate variability from subtle facial color changes"""
        # This is a simplified version - real implementation would need more frames
        if len(frame_sequence) < 10:  # Need sufficient frames
            return {"hrv": 0, "heart_rate": 0}
        
        # Simple RGB analysis (real implementation would use more sophisticated methods)
        green_channel = [np.mean(frame[..., 1]) for frame in frame_sequence]
        
        # Simple FFT to find dominant frequency
        fft = np.fft.fft(green_channel)
        freqs = np.fft.fftfreq(len(green_channel))
        
        # Find peak frequency in plausible heart rate range (0.8-4 Hz = 48-240 BPM)
        mask = (freqs > 0.8) & (freqs < 4)
        if not np.any(mask):
            return {"hrv": 0, "heart_rate": 0}
        
        peak_freq = freqs[mask][np.argmax(np.abs(fft)[mask])]
        heart_rate = peak_freq * 60  # Convert to BPM
        
        return {
            "hrv": 0,  # Would need more sophisticated calculation
            "heart_rate": heart_rate
        }
