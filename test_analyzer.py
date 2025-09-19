"""
Simple test script for the LipSyncAnalyzer class.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))

# Try to import the analyzer
try:
    from lip_sync_detector.lip_sync_analyzer import LipSyncAnalyzer
    print("Successfully imported LipSyncAnalyzer")
except ImportError as e:
    print(f"Error importing LipSyncAnalyzer: {e}")
    sys.exit(1)

def test_analyzer():
    """Test the LipSyncAnalyzer with a sample image."""
    print("Testing LipSyncAnalyzer...")
    
    # Create a sample image (blue background with a white rectangle for the face)
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:, :] = (255, 0, 0)  # Blue background
    
    # Draw a white rectangle for the face
    cv2.rectangle(img, (200, 100), (440, 340), (255, 255, 255), -1)
    
    # Draw a black rectangle for the mouth
    cv2.rectangle(img, (280, 220), (360, 260), (0, 0, 0), -1)
    
    # Convert to RGB (MediaPipe expects RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Initialize the analyzer
    print("Initializing LipSyncAnalyzer...")
    analyzer = LipSyncAnalyzer()
    
    # Extract mouth ROI
    print("Extracting mouth ROI...")
    mouth_roi, landmarks = analyzer.extract_mouth_roi(img_rgb)
    
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
    test_analyzer()
