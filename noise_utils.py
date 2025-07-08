"""
noise_utils.py
Noise addition utilities for ultrasound images.
"""

import numpy as np

def add_rayleigh_noise(img, scale=0.2):
    """Add Rayleigh noise to a single image"""
    noiseRayleigh = np.random.rayleigh(scale=scale, size=img.shape)
    noisy = img * (1 + noiseRayleigh)
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy

def add_Gaussian_noise(img, scale=0.2):
    """Add Gaussian noise to a single image"""
    noiseGaussian = np.random.normal(scale=scale, size=img.shape)
    noisyGaussian = img + noiseGaussian
    noisyGaussian = np.clip(noisyGaussian, 0.0, 1.0)
    return noisyGaussian

def generate_noisy_images(images, scales=[0.10, 0.25, 0.50, 0.75]):
    """Generate 12 noisy images per input image (4 Rayleigh, 4 Gaussian, 4 combined)"""
    noised_images = []
    clean_images = []
    for img in images:
        for sigma in scales:
            # Rayleigh noise only
            rayleigh_noised = add_rayleigh_noise(img, scale=sigma)
            noised_images.append(rayleigh_noised)
            clean_images.append(img)
            # Gaussian noise only
            gaussian_noised = add_Gaussian_noise(img, scale=sigma)
            noised_images.append(gaussian_noised)
            clean_images.append(img)
            # Combined Rayleigh + Gaussian noise
            combined = add_rayleigh_noise(img, scale=sigma)
            combined = add_Gaussian_noise(combined, scale=sigma)
            noised_images.append(combined)
            clean_images.append(img)
    return np.array(noised_images), np.array(clean_images) 