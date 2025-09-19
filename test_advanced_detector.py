"""
Test script for the Advanced Deepfake Detection System.
"""

import os
import sys
import json
from pathlib import Path
import argparse

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))

def test_detector(video_path: str, output_dir: str = "results"):
    """
    Test the advanced deepfake detector on a video file.
    
    Args:
        video_path: Path to the video file to test.
        output_dir: Directory to save the results.
    """
    from advanced_deepfake_detector import DeepfakeDetector
    
    print("=" * 60)
    print("TESTING ADVANCED DEEPFAKE DETECTOR")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector
    print("\nInitializing detector...")
    detector = DeepfakeDetector()
    
    # Analyze video
    print(f"\nAnalyzing video: {video_path}")
    result = detector.predict(video_path)
    
    # Save results
    output_file = os.path.join(output_dir, f"detection_result_{os.path.basename(video_path)}.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Result: {'DEEPFAKE' if result.get('is_deepfake') else 'AUTHENTIC'}")
    print(f"Confidence: {result.get('confidence', 0) * 100:.2f}%")
    print(f"Analysis Time: {result.get('analysis_time', 0):.2f} seconds")
    print(f"Results saved to: {os.path.abspath(output_file)}")
    
    return result

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test the Advanced Deepfake Detection System')
    parser.add_argument('video_path', type=str, help='Path to the video file to test')
    parser.add_argument('--output_dir', type=str, default="results",
                       help='Directory to save the results')
    
    args = parser.parse_args()
    
    # Run the test
    test_detector(args.video_path, args.output_dir)

if __name__ == "__main__":
    main()
