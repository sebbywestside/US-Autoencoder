"""
model.py
Autoencoder model definition for ultrasound image denoising.
"""

import torch
import torch.nn as nn

class UltrasoundAutoencoder(nn.Module):
    def __init__(self):
        super(UltrasoundAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class CNNAutoencoderLarge(nn.Module):
    def __init__(self):
        super(CNNAutoencoderLarge, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, 32, 580, 420)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                          # (B, 32, 290, 210)
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # (B, 32, 290, 210)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                          # (B, 32, 145, 105)
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # (B, 32, 145, 105)
            nn.ReLU(inplace=True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),                 # (B, 32, 290, 210)
            nn.Conv2d(32, 32, kernel_size=3, padding=1),                 # (B, 32, 290, 210)
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),                 # (B, 32, 580, 420)
            nn.Conv2d(32, 1, kernel_size=3, padding=1),                  # (B, 1, 580, 420)
            # Linear activation (no nonlinearity)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x 