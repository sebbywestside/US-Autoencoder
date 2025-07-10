"""
dataset.py
Custom PyTorch dataset for paired noisy and clean ultrasound images.
"""

import torch
from torch.utils.data import Dataset
import numpy as np

class UltrasoundDataset(Dataset):
    def __init__(self, noisy_images, clean_images):
        self.noisy_images = torch.FloatTensor(np.array(noisy_images)).unsqueeze(1)  # (N, 1, H, W)
        self.clean_images = torch.FloatTensor(np.array(clean_images)).unsqueeze(1)  # (N, 1, H, W)
    def __len__(self):
        return len(self.noisy_images)
    def __getitem__(self, idx):
        return self.noisy_images[idx], self.clean_images[idx] 