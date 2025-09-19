# run_enhanced_detection.py
import os
import sys
import cv2
import time
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from feature_engineering import FeatureExtractor
from optimized_model_trainer import ModelTrainer
from data_augmentation import DataAugmentor
import pandas as pd
import joblib
from typing import List, Dict, Optional, Tuple, Union   
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('deepfake_detection.log')
    ]
)
logger = logging.getLogger(__name__)

class EnhancedDeepTrustPipeline:
    def __init__(self, model_path: str = None):
        self.feature_extractor = FeatureExtractor()
        self.model_trainer = ModelTrainer()
        self.augmentor = DataAugmentor()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.model_path = model_path or r'models/new_deepfake_model.pkl'
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model or train a new one if not found"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model not found at {self.model_path}")
                self._train_new_model()
                return
                
            logger.info(f"Loading model from {self.model_path}")
            
            # Handle potential version warnings
            import warnings
            from sklearn.exceptions import InconsistentVersionWarning
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=InconsistentVersionWarning)
                try:
                    model_data = joblib.load(self.model_path)
                    logger.info(f"Model data keys: {list(model_data.keys()) if isinstance(model_data, dict) else 'Not a dict'}")
                except Exception as e:
                    logger.error(f"Error loading model file: {str(e)}")
                    logger.warning("Creating a new model...")
                    self._train_new_model()
                    return
            
            # Handle different model formats
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.feature_columns = model_data.get('feature_columns', [])
                self.scaler = model_data.get('scaler', StandardScaler())
                logger.info(f"Loaded model with {len(self.feature_columns)} feature columns")
                logger.debug(f"Feature columns: {self.feature_columns[:10]}..." if self.feature_columns else "No feature columns")
            else:
                logger.warning("Model is not in expected dictionary format")
                self.model = model_data
                self.feature_columns = []
                self.scaler = StandardScaler()
            
            if self.model is None:
                raise ValueError("Failed to load model from file")
                
            logger.info(f"Model loaded successfully. Type: {type(self.model).__name__}")
            
            # If no feature columns, try to get them from the model (for some scikit-learn models)
            if not self.feature_columns and hasattr(self.model, 'feature_names_in_'):
                self.feature_columns = list(self.model.feature_names_in_)
                logger.info(f"Got {len(self.feature_columns)} feature columns from model")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.warning("Creating a new model...")
            self._train_new_model()
    
    def _train_new_model(self):
        """Train a new model with default parameters"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        try:
            logger.info("Creating a new RandomForest model with default parameters...")
            
            # Create a simple model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Create some dummy features for initialization
            # In a real scenario, you would load your actual training data
            self.feature_columns = [
                'vis_face_count', 'vis_blur_score', 'vis_brightness',
                'vis_contrast', 'vis_aspect_ratio'
            ]
            
            # Initialize scaler with default values
            self.scaler = StandardScaler()
            dummy_X = np.zeros((1, len(self.feature_columns)))
            self.scaler.fit(dummy_X)
            
            # Save the model for future use
            model_data = {
                'model': self.model,
                'feature_columns': self.feature_columns,
                'scaler': self.scaler,
                'model_type': 'RandomForestClassifier',
                'version': '1.0.0'
            }
            
            # Ensure models directory exists
            os.makedirs(os.path.dirname(self.model_path) or 'models', exist_ok=True)
            
            # Save the model
            joblib.dump(model_data, self.model_path)
            logger.info(f"New model created and saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error creating new model: {str(e)}")
            # Fall back to a simple model if there's an error
            self.model = RandomForestClassifier()
            logger.warning("Falling back to default RandomForest model")
    
    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video using moviepy"""
        try:
            try:
                from moviepy.editor import VideoFileClip
            except ImportError as ie:
                logger.error("MoviePy not properly installed. Try: pip install moviepy")
                logger.error(f"Import error details: {str(ie)}")
                return None
                
            import tempfile
            
            # Create temp directory if it doesn't exist
            temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Create temp file
            temp_audio = os.path.join(temp_dir, f"temp_audio_{int(time.time())}.wav")
            
            # Extract audio
            try:
                video = VideoFileClip(video_path)
                video.audio.write_audiofile(temp_audio, logger=None)
                return temp_audio
            except Exception as e:
                logger.error(f"Error during audio extraction: {str(e)}")
                # Try alternative method using ffmpeg directly
                try:
                    import subprocess
                    cmd = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{temp_audio}" -y'
                    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    return temp_audio if os.path.exists(temp_audio) else None
                except Exception as ffmpeg_error:
                    logger.error(f"FFmpeg extraction also failed: {str(ffmpeg_error)}")
                    return None
                
        except Exception as e:
            logger.error(f"Unexpected error in extract_audio: {str(e)}")
            return None
    
    def extract_frames(self, video_path: str, output_dir: str = "frames") -> List[np.ndarray]:
        """Extract frames from video"""
        os.makedirs(output_dir, exist_ok=True)
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only keep every 5th frame for efficiency
                if frame_count % 5 == 0:
                    frames.append(frame)
                    # Save frame for debugging
                    cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:04d}.jpg"), frame)
                
                frame_count += 1
            
            cap.release()
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            return []
    
    def process_video(self, video_path: str) -> dict:
        """Process a single video file and return analysis results"""
        start_time = time.time()
        results = {
            'filename': os.path.basename(video_path),
            'prediction': 'UNKNOWN',
            'confidence': 0.0,
            'processing_time': 0,
            'features': {},
            'error': None,
            'success': False
        }
        
        # Check if video file exists
        if not os.path.exists(video_path):
            results['error'] = f"Video file not found: {video_path}"
            return results
        
        try:
            # 1. Extract audio
            logger.info(f"Extracting audio from {video_path}...")
            audio_path = None
            try:
                audio_path = self.extract_audio(video_path)
                if not audio_path:
                    logger.warning("Audio extraction failed, continuing with video analysis only")
            except Exception as e:
                logger.warning(f"Audio extraction failed but continuing: {str(e)}")
                audio_path = None
            
            # 2. Extract frames
            logger.info("Extracting video frames...")
            frames = self.extract_frames(video_path)
            if not frames:
                raise ValueError("No frames extracted from video")
            
            # 3. Extract features
            logger.info("Extracting audio features...")
            audio_features = {}
            if audio_path and os.path.exists(audio_path):
                try:
                    audio_features = self.feature_extractor.extract_audio_features(audio_path) or {}
                    logger.info(f"Extracted {len(audio_features)} audio features")
                except Exception as e:
                    logger.warning(f"Audio feature extraction failed: {str(e)}")
            else:
                logger.warning("No audio available, using video features only")
            
            logger.info("Extracting visual features...")
            visual_features = []
            frame_count = min(30, len(frames))  # Process up to 30 frames
            for i, frame in enumerate(frames[:frame_count]):
                try:
                    frame_features = self.feature_extractor.extract_visual_features(frame)
                    if frame_features:
                        visual_features.append(frame_features)
                    if (i + 1) % 10 == 0:  # Log progress every 10 frames
                        logger.info(f"Processed {i + 1}/{frame_count} frames")
                except Exception as e:
                    logger.warning(f"Error extracting visual features from frame {i}: {str(e)}")
            
            if not visual_features:
                logger.warning("No visual features extracted, using default values")
                visual_features = [{
                    'face_count': 1,
                    'blur_score': 0.0,
                    'brightness': 0.5,
                    'contrast': 0.5,
                    'aspect_ratio': 1.0
                }]
            
            # 4. Combine features
            combined_features = self._combine_features(audio_features, visual_features)
            logger.info(f"Combined {len(combined_features)} features")
            
            # 5. Make prediction
            if self.model:
                logger.info("Preparing features for prediction...")
                
                # Ensure features are in the correct order and handle missing values
                if self.feature_columns:
                    logger.info(f"Model expects {len(self.feature_columns)} features")
                    logger.debug(f"Available features: {list(combined_features.keys())}")
                    
                    # Create a DataFrame with all expected columns
                    combined_df = pd.DataFrame([combined_features])
                    
                    # Add missing columns with default value of 0.0
                    missing_cols = set(self.feature_columns) - set(combined_df.columns)
                    for col in missing_cols:
                        logger.warning(f"Adding missing feature with default value: {col}")
                        combined_df[col] = 0.0
                    
                    # Reorder columns to match training data
                    combined_df = combined_df[self.feature_columns]
                    combined_features = combined_df.iloc[0].values
                    
                    logger.info(f"Final feature vector shape: {combined_features.shape}")
                
                # Scale features if scaler is available
                try:
                    if hasattr(self, 'scaler') and self.scaler is not None:
                        logger.info("Scaling features...")
                        scaled_features = self.scaler.transform([combined_features])
                    else:
                        logger.warning("No scaler found, using raw features")
                        scaled_features = [combined_features]
                    
                    # Make prediction
                    logger.info("Making prediction...")
                    prediction = self.model.predict(scaled_features)[0]
                    proba = self.model.predict_proba(scaled_features)[0]
                    confidence = float(max(proba))
                    
                    logger.info(f"Prediction: {'FAKE' if prediction == 1 else 'REAL'} with confidence: {confidence:.2f}")
                    
                except Exception as e:
                    logger.error(f"Error during prediction: {str(e)}")
                    raise
                
                results.update({
                    'prediction': 'FAKE' if prediction == 1 else 'REAL',
                    'confidence': float(max(proba)),
                    'features': {
                        'audio': list(audio_features.keys()),
                        'visual': list(visual_features[0].keys()) if visual_features else []
                    },
                    'success': True
                })
            else:
                raise ValueError("No model available for prediction")
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            results['error'] = str(e)
            results['success'] = False
        finally:
            # Cleanup temporary files
            if 'audio_path' in locals() and audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
            
            results['processing_time'] = time.time() - start_time
            logger.info(f"Processing completed in {results['processing_time']:.2f} seconds")
            return results
    
    def _combine_features(self, audio_features: dict, visual_features_list: list) -> dict:
        """
        Combine audio and visual features into a single feature vector
        
        Args:
            audio_features: Dictionary of audio features
            visual_features_list: List of visual features for each frame
            
        Returns:
            Dictionary of combined features with consistent naming
        """
        combined = {}
        
        # Process audio features
        if audio_features:
            for key, value in audio_features.items():
                if isinstance(value, (int, float)):
                    combined[f'audio_{key}'] = float(value)
        
        # Process visual features (average across frames)
        if visual_features_list and len(visual_features_list) > 0:
            # Get all unique feature names
            all_keys = set()
            for frame_features in visual_features_list:
                if frame_features:
                    all_keys.update(frame_features.keys())
            
            # Calculate mean for each feature
            for key in all_keys:
                values = [f[key] for f in visual_features_list if f and key in f]
                if values:
                    combined[f'vis_{key}'] = float(np.mean(values))
        
        return combined
        
    def process_video_folder(self, folder_path: str) -> List[dict]:
        """
        Process all video files in the specified folder
        
        Args:
            folder_path: Path to the folder containing video files
            
        Returns:
            List of result dictionaries for each processed video
        """
        results = []
        
        # Supported video file extensions
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
        
        try:
            # Get all video files in the folder
            video_files = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f)) and 
                f.lower().endswith(video_extensions)
            ]
            
            if not video_files:
                logger.warning(f"No video files found in {folder_path}")
                return results
                
            logger.info(f"Found {len(video_files)} video files to process")
            
            # Process each video file
            for video_file in video_files:
                try:
                    logger.info(f"\nProcessing video: {os.path.basename(video_file)}")
                    result = self.process_video(video_file)
                    results.append(result)
                    
                    # Save individual report
                    report_path = self._save_report(result)
                    if report_path:
                        logger.info(f"Report saved to: {report_path}")
                        
                except Exception as e:
                    logger.error(f"Error processing {video_file}: {str(e)}")
                    results.append({
                        'filename': os.path.basename(video_file),
                        'error': str(e),
                        'success': False
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing video folder {folder_path}: {str(e)}")
            raise
    
    def _save_report(self, result: dict) -> str:
        """
        Save analysis report to a file
        
        Args:
            result: Dictionary containing analysis results
            
        Returns:
            str: Path to the saved report
        """
        try:
            # Create reports directory if it doesn't exist
            os.makedirs('reports', exist_ok=True)
            
            # Generate report filename
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            report_filename = f"report_{result.get('filename', 'unknown')}_{timestamp}.txt"
            report_path = os.path.join('reports', report_filename)
            
            # Create report content
            filename = result.get('filename', 'Unknown')
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            prediction = result.get('prediction', 'UNKNOWN')
            confidence = result.get('confidence', 0) * 100
            processing_time = result.get('processing_time', 0)
            
            report = f"""
================================================================================
                             DEEPFAKE DETECTION REPORT                           
================================================================================

File: {filename}
Timestamp: {timestamp}

PREDICTION: {prediction}
Confidence: {confidence:.2f}%
Processing Time: {processing_time:.2f} seconds

================================================================================
"""
            
            # Save to file
            with open(report_path, 'w') as f:
                f.write(report)
                
            return os.path.abspath(report_path)
            
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            return ""

def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection System')
    parser.add_argument('--video', type=str, help='Path to a single video file')
    parser.add_argument('--folder', type=str, default='data', 
                       help='Path to folder containing videos (default: data)')
    parser.add_argument('--model', type=str, help='Path to pre-trained model')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = EnhancedDeepTrustPipeline(model_path=args.model)
        
        if args.video:
            # Process single video
            print(f"\nProcessing video: {args.video}")
            result = pipeline.process_video(args.video)
            
            print("\n" + "="*80)
            print("ANALYSIS RESULTS")
            print("="*80)
            print(f"File: {result['filename']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']*100:.2f}%")
            print(f"Processing Time: {result['processing_time']:.2f} seconds")
            
            # Save report
            report_path = pipeline._save_report(result)
            if report_path:
                print(f"\nDetailed report saved to: {report_path}")
            
        else:
            # Process all videos in folder
            print(f"\nProcessing all videos in folder: {args.folder}")
            results = pipeline.process_video_folder(args.folder)
            
            # Print summary
            print("\n" + "="*80)
            print("PROCESSING SUMMARY")
            print("="*80)
            for result in results:
                print(f"\n{result['filename']}:")
                print(f"  Prediction: {result.get('prediction', 'UNKNOWN')}")
                print(f"  Confidence: {result.get('confidence', 0)*100:.2f}%")
                if 'error' in result:
                    print(f"  Error: {result['error']}")
            
            # Save combined report
            if results:
                combined_report = {
                    'timestamp': datetime.now().isoformat(),
                    'total_videos': len(results),
                    'successful': len([r for r in results if r.get('success', False)]),
                    'failed': len([r for r in results if not r.get('success', True)]),
                    'results': results
                }
                report_path = os.path.join('reports', f'combined_report_{int(time.time())}.json')
                with open(report_path, 'w') as f:
                    json.dump(combined_report, f, indent=2)
                print(f"\nCombined report saved to: {report_path}")
    
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        print(f"\nERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()