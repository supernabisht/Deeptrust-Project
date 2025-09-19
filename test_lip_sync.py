"""
Test script for lip-sync analysis functionality.
"""

import cv2
import numpy as np
from lip_sync_analyzer import LipSyncAnalyzer

def test_lip_sync_analyzer():
    """Test the LipSyncAnalyzer with a sample image."""
    print("Testing LipSyncAnalyzer with a sample image...")
    
    # Create a sample image (blue background with a white rectangle for the face)
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:, :] = (255, 0, 0)  # Blue background
    
    # Draw a white rectangle for the face
    cv2.rectangle(img, (200, 100), (440, 340), (255, 255, 255), -1)
    
    # Draw a black rectangle for the mouth
    cv2.rectangle(img, (280, 220), (360, 260), (0, 0, 0), -1)
    
    # Initialize the analyzer
    analyzer = LipSyncAnalyzer()
    
    # Extract mouth ROI
    mouth_roi, landmarks = analyzer.extract_mouth_roi(img)
    
    if mouth_roi is not None:
        print("✓ Successfully extracted mouth ROI")
        print(f"   - ROI shape: {mouth_roi.shape}")
        
        # Display the result
        cv2.imshow("Mouth ROI", mouth_roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("✗ Failed to extract mouth ROI")
    
    print("Test completed!")

if __name__ == "__main__":
    test_lip_sync_analyzer()
