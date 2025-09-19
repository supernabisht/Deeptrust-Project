import os
import sys
import json
import numpy as np
from run_enhanced_detection import EnhancedDeepTrustPipeline

def run_pipeline_test(video_path):
    """Run the deepfake detection pipeline with detailed logging"""
    print(f"\n{'='*80}")
    print(f"RUNNING DEEPFAKE DETECTION PIPELINE TEST")
    print(f"{'='*80}\n")
    
    # Initialize the pipeline
    print("[1/6] Initializing pipeline...")
    pipeline = EnhancedDeepTrustPipeline()
    pipeline.video_path = video_path
    
    # Test environment setup
    print("\n[2/6] Testing environment setup...")
    try:
        pipeline._setup_environment()
        print("✅ Environment setup completed successfully")
    except Exception as e:
        print(f"❌ Environment setup failed: {str(e)}")
        return False
    
    # Test audio extraction
    print("\n[3/6] Testing audio extraction...")
    try:
        audio_path = pipeline._extract_audio(video_path)
        print(f"✅ Audio extracted to: {audio_path}")
        print(f"   File exists: {os.path.exists(audio_path)}")
        print(f"   File size: {os.path.getsize(audio_path) / 1024:.2f} KB")
    except Exception as e:
        print(f"❌ Audio extraction failed: {str(e)}")
        return False
    
    # Test frame extraction
    print("\n[4/6] Testing frame extraction...")
    try:
        frames_path = pipeline._extract_frames(video_path)
        print(f"✅ Frames extracted to: {frames_path}")
        print(f"   Directory exists: {os.path.exists(frames_path)}")
        frame_files = [f for f in os.listdir(frames_path) if f.endswith('.jpg')]
        print(f"   Number of frames extracted: {len(frame_files)}")
        if frame_files:
            print(f"   First frame: {frame_files[0]}")
    except Exception as e:
        print(f"❌ Frame extraction failed: {str(e)}")
        return False
    
    # Test feature extraction
    print("\n[5/6] Testing feature extraction...")
    try:
        features = pipeline._extract_features(audio_path, frames_path)
        print("✅ Feature extraction completed")
        print(f"   Number of features: {len(features) if features else 0}")
        if features:
            print("   First 5 features:")
            for i, (k, v) in enumerate(features.items()):
                if i >= 5:
                    break
                print(f"     {k}: {v}")
    except Exception as e:
        print(f"❌ Feature extraction failed: {str(e)}")
        return False
    
    # Test prediction
    print("\n[6/6] Testing prediction...")
    try:
        if not features:
            print("⚠️  No features available for prediction")
            return False
            
        prediction = pipeline._make_prediction(features)
        print("✅ Prediction completed")
        print("\nPREDICTION RESULTS:")
        print(f"- Is Fake: {prediction.get('is_fake', 'N/A')}")
        print(f"- Confidence: {prediction.get('confidence', 'N/A')}")
        print(f"- Real Probability: {prediction.get('real_prob', 'N/A')}")
        print(f"- Fake Probability: {prediction.get('fake_prob', 'N/A')}")
        
        # Generate report
        print("\nGenerating report...")
        report_path = pipeline._generate_report(prediction)
        print(f"✅ Report generated at: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Check if video path is provided as argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "data/real/video1.mp4"
    
    # Convert to absolute path
    video_path = os.path.abspath(video_path)
    
    print(f"\n{'='*80}")
    print(f"DEEPFAKE DETECTION PIPELINE TEST")
    print(f"{'='*80}")
    print(f"Video: {video_path}")
    print(f"File exists: {os.path.exists(video_path)}")
    
    if not os.path.exists(video_path):
        print(f"\n❌ Error: Video file not found at {video_path}")
        return 1
    
    # Run the pipeline test
    success = run_pipeline_test(video_path)
    
    print(f"\n{'='*80}")
    print("TEST SUMMARY:")
    print(f"- Video: {os.path.basename(video_path)}")
    print(f"- Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"{'='*80}\n")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
