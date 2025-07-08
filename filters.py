"""
filters.py
Classical filter implementations for denoising ultrasound images.
"""

import numpy as np
import cv2
from scipy.ndimage import uniform_filter, gaussian_filter

# Median filter (3x3)
def median_filter(img):
    return cv2.medianBlur((img * 255).astype(np.uint8), 3) / 255.0

# Mean filter (3x3)
def mean_filter(img):
    return uniform_filter(img, size=3)

# Gaussian filter (sigma=1)
def gaussian_filter_img(img):
    return gaussian_filter(img, sigma=1)

# Bilateral filter (OpenCV)
def bilateral_filter(img):
    return cv2.bilateralFilter((img * 255).astype(np.uint8), 5, 75, 75) / 255.0

# Frost filter (simple implementation)
def frost_filter(img, win_size=3, damping=2.0):
    pad = win_size // 2
    padded = np.pad(img, pad, mode='reflect')
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i:i+win_size, j:j+win_size]
            mean = np.mean(window)
            var = np.var(window)
            if var == 0:
                out[i, j] = mean
                continue
            coeff = np.exp(-damping * np.abs(window - img[i, j]) / (mean + 1e-8))
            out[i, j] = np.sum(coeff * window) / np.sum(coeff)
    return out

# Kuan filter (simple implementation)
def kuan_filter(img, win_size=3):
    pad = win_size // 2
    padded = np.pad(img, pad, mode='reflect')
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i:i+win_size, j:j+win_size]
            mean = np.mean(window)
            var = np.var(window)
            cu = 0.523 / (mean + 1e-8)  # cu for fully developed speckle
            w = var / (var + cu**2 * mean**2 + 1e-8)
            out[i, j] = mean + w * (img[i, j] - mean)
    return out

# Lee filter (simple implementation)
def lee_filter(img, win_size=3):
    pad = win_size // 2
    padded = np.pad(img, pad, mode='reflect')
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i:i+win_size, j:j+win_size]
            mean = np.mean(window)
            var = np.var(window)
            noise_var = np.mean(var)
            w = var / (var + noise_var + 1e-8)
            out[i, j] = mean + w * (img[i, j] - mean)
    return out 