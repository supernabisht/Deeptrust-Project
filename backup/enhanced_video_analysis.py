#!/usr/bin/env python3
"""
enhanced_video_analysis.py - Complete video analysis pipeline for deepfake detection
"""

import cv2
import numpy as np
import os
import json
import pickle
from extract_frames import extract_key_frames
from extract_audio import extract_audio_from_video
from extract_mfcc import extract_mfcc_features
from lip_sync import analyze_lip_sync
from advanced_feature_extractor import extract_video_features

def analyze_video(video_path, model_path='optimized_deepfake_model.pkl'):
    """Complete video analysis pipeline"""
    print(f"🎬 Analyzing video: {video_path}")
    
    # Create output directory
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join('results', video_name)
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'video_path': video_path,
        'video_name': video_name,
        'analyses': {}
    }
    
    try:
        # 1. Extract key frames
        print("📸 Extracting key frames...")
        frames_dir = os.path.join(output_dir, 'frames')
        frames = extract_key_frames(video_path, frames_dir)
        results['analyses']['frame_extraction'] = {
            'status': 'success',
            'frames_extracted': len(frames)
        }
        
        # 2. Extract audio
        print("🔊 Extracting audio...")
        audio_path = os.path.join(output_dir, 'audio.wav')
        audio_extracted = extract_audio_from_video(video_path, audio_path)
        results['analyses']['audio_extraction'] = {
            'status': 'success' if audio_extracted else 'failed'
        }
        
        # 3. Extract MFCC features
        print("🎵 Extracting MFCC features...")
        mfcc_features = extract_mfcc_features(audio_path)
        results['analyses']['mfcc_extraction'] = {
            'status': 'success',
            'mfcc_shape': mfcc_features.shape if mfcc_features is not None else None
        }
        
        # 4. Analyze lip sync
        print("👄 Analyzing lip sync...")
        lip_sync_score = analyze_lip_sync(video_path, audio_path)
        results['analyses']['lip_sync'] = {
            'status': 'success',
            'score': lip_sync_score
        }
        
        # 5. Extract video features
        print("🔍 Extracting video features...")
        video_features = extract_video_features(frames_dir)
        results['analyses']['feature_extraction'] = {
            'status': 'success',
            'features_extracted': len(video_features) if video_features else 0
        }
        
        # 6. Load model and predict
        print("🤖 Making prediction...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Combine all features for prediction
        combined_features = combine_features(video_features, mfcc_features, lip_sync_score)
        prediction = model.predict([combined_features])
        confidence = model.predict_proba([combined_features])
        
        results['prediction'] = {
            'label': 'Fake' if prediction[0] == 1 else 'Real',
            'confidence': float(np.max(confidence)),
            'probabilities': {
                'real': float(confidence[0][0]),
                'fake': float(confidence[0][1])
            }
        }
        
        # Save results
        results_path = os.path.join(output_dir, 'analysis_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Analysis complete! Results saved to: {results_path}")
        print(f"🎯 Prediction: {results['prediction']['label']} "
              f"(Confidence: {results['prediction']['confidence']:.2%})")
        
        return results
        
    except Exception as e:
        print(f"❌ Error analyzing video: {e}")
        results['analyses']['error'] = str(e)
        return results

def combine_features(video_features, mfcc_features, lip_sync_score):
    """Combine all extracted features into a single feature vector"""
    # This is a simplified version - adapt based on your actual feature structure
    combined = []
    
    if video_features:
        combined.extend(video_features)
    
    if mfcc_features is not None:
        mfcc_flat = mfcc_features.flatten()
        combined.extend(mfcc_flat[:10])  # Use first 10 MFCC coefficients
    
    combined.append(lip_sync_score)
    
    return np.array(combined)

if __name__ == "__main__":
    video_path = "my_video.mp4"  # Change to your video path
    analyze_video(video_path)