import math

import cv2
import numpy as np
from scipy import io as scio
from scipy import linalg
from scipy import signal
from scipy import sparse
from skimage.util import img_as_float
from sklearn.metrics import mean_squared_error


def detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal


def process_video(frames, use_skin_mask=True):
    RGB = []
    for frame in frames:
        if use_skin_mask:
            # Ensure frame is 8-bit for OpenCV
            f_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
            # Assuming frame is in RGB format
            hsv = cv2.cvtColor(f_uint8, cv2.COLOR_RGB2HSV)
            
            # Classic HSV skin thresholds (OpenCV Hue is 0-179)
            lower_skin1 = np.array([0, 20, 50], dtype=np.uint8)
            upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
            mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
            
            lower_skin2 = np.array([170, 20, 50], dtype=np.uint8)
            upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
            mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            
            mask = mask1 | mask2
            
            skin_pixels = frame[mask == 255]
            if len(skin_pixels) > 0:
                avg_color = np.mean(skin_pixels, axis=0)
            else:
                avg_color = np.mean(frame, axis=(0, 1))
        else:
            summation = np.sum(np.sum(frame, axis=0), axis=0)
            avg_color = summation / (frame.shape[0] * frame.shape[1])
            
        RGB.append(avg_color)
        
    RGB = np.asarray(RGB)
    RGB = RGB.transpose(1, 0).reshape(1, 3, -1)
    return np.asarray(RGB)
