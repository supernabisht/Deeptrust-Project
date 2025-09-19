"""
Create a test video with audio for lip-sync detection testing.
"""

import cv2
import numpy as np
import os
import wave
import struct
from moviepy.editor import VideoClip, AudioClip

# Parameters
output_path = "data/fake/test_video_with_audio.mp4"
duration = 5  # seconds
fps = 30
width, height = 640, 360
sample_rate = 44100
freq = 440  # Frequency of the test tone

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Create a simple animation of a talking face
def make_frame(t):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = (255, 0, 0)  # Blue background
    
    # Draw a white face
    face_center = (width // 2, height // 2)
    face_radius = min(width, height) // 3
    cv2.circle(img, face_center, face_radius, (255, 255, 255), -1)
    
    # Draw eyes
    eye_y = height // 2 - face_radius // 3
    eye_radius = face_radius // 5
    cv2.circle(img, (width // 2 - face_radius // 2, eye_y), eye_radius, (0, 0, 0), -1)  # Left eye
    cv2.circle(img, (width // 2 + face_radius // 2, eye_y), eye_radius, (0, 0, 0), -1)  # Right eye
    
    # Animate mouth (opens and closes with a sine wave)
    mouth_open = 0.5 + 0.5 * np.sin(t * 2 * np.pi)  # Varies between 0 and 1
    mouth_width = int(face_radius * 0.8)
    mouth_height = int(face_radius * 0.4 * mouth_open)
    mouth_top = height // 2 + face_radius // 2 - mouth_height // 2
    
    # Draw mouth (as a simple ellipse)
    cv2.ellipse(
        img,
        (width // 2, mouth_top + mouth_height // 2),
        (mouth_width // 2, mouth_height // 2),
        0, 0, 180, (0, 0, 0), -1
    )
    
    # Add frame number
    cv2.putText(
        img, f"Frame {int(t * fps) + 1}", 
        (20, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, (0, 0, 0), 2
    )
    
    return img

# Create a simple audio tone that matches the mouth movement
def make_audio(t):
    # Generate a tone that varies in volume with the mouth movement
    volume = 0.5 + 0.5 * np.sin(t * 2 * np.pi)  # Same as mouth movement
    tone = np.sin(2 * np.pi * freq * t) * volume
    return np.clip(tone * 32767, -32768, 32767).astype(np.int16)

# Create the video clip
print("Creating video frames...")
video_clip = VideoClip(make_frame, duration=duration)

# Create the audio clip
print("Creating audio...")
audio_clip = AudioClip(
    make_audio,
    duration=duration,
    fps=sample_rate
)

# Set the audio to the video
video_clip = video_clip.set_audio(audio_clip)

# Write the video file
print(f"Writing video to {output_path}...")
video_clip.write_videofile(
    output_path,
    codec='libx264',
    audio_codec='aac',
    fps=fps,
    remove_temp=True,
    threads=4
)

print("Done!")
