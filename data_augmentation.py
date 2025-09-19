# data_augmentation.py
import numpy as np
import cv2
import librosa
from typing import Tuple, List
import random

class DataAugmentor:
    @staticmethod
    def augment_audio(y: np.ndarray, sr: int) -> List[Tuple[np.ndarray, int]]:
        """Generate augmented audio samples"""
        augmented = [(y, sr)]  # Original
        
        # Add noise
        noise = np.random.normal(0, 0.005, y.shape[0])
        augmented.append((y + noise, sr))
        
        # Change speed
        speed_factor = random.uniform(0.9, 1.1)
        y_speed = librosa.effects.time_stretch(y, rate=speed_factor)
        augmented.append((y_speed, sr))
        
        # Shift pitch
        n_steps = random.uniform(-1, 1)
        y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        augmented.append((y_pitch, sr))
        
        return augmented
    
    @staticmethod
    def augment_frame(frame: np.ndarray) -> List[np.ndarray]:
        """Generate augmented image frames"""
        augmented = [frame]  # Original
        
        # Flip horizontally
        augmented.append(cv2.flip(frame, 1))
        
        # Adjust brightness and contrast
        alpha = random.uniform(0.9, 1.1)
        beta = random.uniform(-10, 10)
        augmented.append(cv2.convertScaleAbs(frame, alpha=alpha, beta=beta))
        
        # Add Gaussian blur
        if frame.shape[0] > 0 and frame.shape[1] > 0:
            kernel_size = random.choice([3, 5])
            augmented.append(cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0))
        
        return augmented