import os
import cv2

def main():
    video_path = "data/real/video1.mp4"
    
    print(f"Testing video: {os.path.abspath(video_path)}")
    print(f"File exists: {os.path.exists(video_path)}")
    
    if not os.path.exists(video_path):
        print("Error: Video file not found")
        return
    
    # Try to open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file with OpenCV")
        return
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo Properties:")
    print(f"- Frames: {frame_count}")
    print(f"- FPS: {fps}")
    print(f"- Resolution: {width}x{height}")
    
    # Try to read the first frame
    ret, frame = cap.read()
    if ret:
        print("\nSuccessfully read first frame")
        print(f"- Frame shape: {frame.shape}")
        print(f"- Data type: {frame.dtype}")
        
        # Save the first frame as an image for verification
        output_dir = "debug_output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "first_frame.jpg")
        cv2.imwrite(output_path, frame)
        print(f"\nFirst frame saved to: {os.path.abspath(output_path)}")
    else:
        print("\nFailed to read first frame")
    
    cap.release()
    print("\nTest completed successfully")

if __name__ == "__main__":
    main()
