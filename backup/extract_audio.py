
import moviepy as mp
import os

file_path = "my_video.mp4"
if os.path.exists(file_path):
    with mp.VideoFileClip(file_path) as video:
        video.audio.write_audiofile("my_voice.wav")
    print("Audio extracted successfully!")
else:
    print(f"File not found: {file_path}")
