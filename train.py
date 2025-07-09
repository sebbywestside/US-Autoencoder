"""
train.py
Training and evaluation functions for the autoencoder.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from data_utils import load_ultrasound_data
from noise_utils import add_rayleigh_noise
from model import UltrasoundAutoencoder
from dataset import UltrasoundDataset
import torch.optim as optim
import os

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        epoch_loss = 0.0
        for noisy, clean in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
    return train_losses

def evaluate_model(model, test_loader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    count = 0
    with torch.no_grad():
        for noisy, clean in test_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            outputs = model(noisy)
            outputs_np = outputs.cpu().numpy()
            clean_np = clean.cpu().numpy()
            for i in range(outputs_np.shape[0]):
                psnr = peak_signal_noise_ratio(clean_np[i, 0], outputs_np[i, 0], data_range=1.0)
                ssim = structural_similarity(clean_np[i, 0], outputs_np[i, 0], data_range=1.0)
                total_psnr += psnr
                total_ssim += ssim
                count += 1
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    return avg_psnr, avg_ssim

def run_rayleigh_noise_experiments(device):
    data_path = "Data/test"
    train_path = "Data/train"
    noise_levels = [0.1, 0.25, 0.5, 0.65, 0.8]
    num_epochs = 100
    batch_size = 16
    results = []

    print("Loading training images...")
    train_images = load_ultrasound_data(train_path)
    print(f"Loaded {len(train_images)} training images.")
    print("Loading test images...")
    test_images = load_ultrasound_data(data_path)
    print(f"Loaded {len(test_images)} test images.")

    for sigma in noise_levels:
        print(f"\n=== Training and testing at Rayleigh noise level {sigma} ===")
        # Generate noisy images for training
        train_noisy = [add_rayleigh_noise(img, scale=sigma) for img in train_images]
        train_clean = train_images
        test_noisy = [add_rayleigh_noise(img, scale=sigma) for img in test_images]
        test_clean = test_images

        # Create datasets and dataloaders
        train_dataset = UltrasoundDataset(train_noisy, train_clean)
        test_dataset = UltrasoundDataset(test_noisy, test_clean)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        model = UltrasoundAutoencoder().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        print("Training...")
        train_losses = train_model(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)

        # Evaluate the model
        print("Evaluating...")
        avg_psnr, avg_ssim = evaluate_model(model, test_loader, device)
        print(f"Rayleigh noise {sigma}: PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}")
        results.append({
            'noise_level': sigma,
            'psnr': avg_psnr,
            'ssim': avg_ssim
        })

        # Optionally save the model for each noise level
        model_save_path = f"ultrasound_autoencoder_rayleigh_{sigma}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved as {model_save_path}")

    # Print summary
    print("\n=== Summary of Results ===")
    for r in results:
        print(f"Rayleigh noise {r['noise_level']}: PSNR={r['psnr']:.2f} dB, SSIM={r['ssim']:.4f}")

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def main():
    device = get_device()
    print(f"Using device: {device}")
    run_rayleigh_noise_experiments(device)

if __name__ == "__main__":
    main() 