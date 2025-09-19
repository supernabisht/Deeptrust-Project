import os
import sys
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import json

def test_video_processing(video_path):
    """Test basic video processing capabilities"""
    results = {
        'video_path': os.path.abspath(video_path),
        'file_exists': os.path.exists(video_path),
        'file_size_mb': os.path.getsize(video_path) / (1024 * 1024) if os.path.exists(video_path) else 0,
        'opencv_test': {},
        'moviepy_test': {}
    }
    
    if not results['file_exists']:
        return results
    
    # Test OpenCV
    try:
        cap = cv2.VideoCapture(video_path)
        results['opencv_test']['success'] = cap.isOpened()
        if cap.isOpened():
            results['opencv_test']['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            results['opencv_test']['fps'] = cap.get(cv2.CAP_PROP_FPS)
            results['opencv_test']['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            results['opencv_test']['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Try to read first frame
            ret, frame = cap.read()
            results['opencv_test']['first_frame'] = {
                'read_success': ret,
                'frame_shape': frame.shape if ret else None,
                'frame_dtype': str(frame.dtype) if ret else None
            }
        cap.release()
    except Exception as e:
        results['opencv_test']['error'] = str(e)
    
    # Test MoviePy
    try:
        clip = VideoFileClip(video_path)
        results['moviepy_test']['success'] = True
        results['moviepy_test']['duration'] = clip.duration
        results['moviepy_test']['fps'] = clip.fps
        results['moviepy_test']['size'] = clip.size
        results['moviepy_test']['n_frames'] = int(clip.duration * clip.fps) if clip.fps else None
        
        # Try to get first frame
        try:
            frame = clip.get_frame(0)
            results['moviepy_test']['first_frame'] = {
                'shape': frame.shape,
                'dtype': str(frame.dtype),
                'mean_intensity': float(np.mean(frame))
            }
        except Exception as e:
            results['moviepy_test']['first_frame_error'] = str(e)
            
        clip.close()
    except Exception as e:
        results['moviepy_test']['error'] = str(e)
    
    return results

def main():
    video_path = "data/real/video1.mp4"
    
    print(f"Testing video processing for: {os.path.abspath(video_path)}")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {os.path.abspath(video_path)}")
        return 1
    
    print("Running tests...")
    results = test_video_processing(video_path)
    
    # Save results to file
    output_file = "video_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTest results saved to: {os.path.abspath(output_file)}")
    print("\nSummary:")
    print(f"- File exists: {results['file_exists']}")
    print(f"- File size: {results['file_size_mb']:.2f} MB")
    
    if 'opencv_test' in results and 'success' in results['opencv_test']:
        print("\nOpenCV Test:")
        if results['opencv_test']['success']:
            print(f"- Successfully opened video")
            print(f"- Frames: {results['opencv_test']['frame_count']}")
            print(f"- FPS: {results['opencv_test']['fps']:.2f}")
            print(f"- Resolution: {results['opencv_test']['width']}x{results['opencv_test']['height']}")
        else:
            print("- Failed to open video with OpenCV")
    
    if 'moviepy_test' in results and 'success' in results['moviepy_test']:
        print("\nMoviePy Test:")
        if results['moviepy_test']['success']:
            print(f"- Successfully opened video")
            print(f"- Duration: {results['moviepy_test']['duration']:.2f} seconds")
            print(f"- FPS: {results['moviepy_test']['fps']}")
            print(f"- Resolution: {results['moviepy_test']['size'][0]}x{results['moviepy_test']['size'][1]}")
            print(f"- Estimated frames: {results['moviepy_test']['n_frames']}")
        else:
            print("- Failed to open video with MoviePy")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
