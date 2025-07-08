"""
data_utils.py
Data loading and preprocessing utilities for ultrasound images.
"""

import os
import numpy as np
from PIL import Image

def load_ultrasound_data(data_path, image_size=(128, 128)):
    """Load ultrasound images and prepare for training"""
    image_files = sorted([f for f in os.listdir(data_path)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))])
    images = []
    for img_name in image_files:
        try:
            img_path = os.path.join(data_path, img_name)
            img = Image.open(img_path).convert('L')
            img = img.resize(image_size, Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            images.append(arr)
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            continue
    return np.array(images) 