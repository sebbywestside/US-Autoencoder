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
from model import CNNAutoencoderLarge
from dataset import UltrasoundDataset
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import random

noise_levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50):
    model.train()
    loss_history = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        count = 0
        for noisy, clean in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # Compute metrics for this batch
            outputs_np = outputs.detach().cpu().numpy()
            clean_np = clean.cpu().numpy()
            for i in range(outputs_np.shape[0]):
                psnr = peak_signal_noise_ratio(clean_np[i, 0], outputs_np[i, 0], data_range=1.0)
                ssim = structural_similarity(clean_np[i, 0], outputs_np[i, 0], data_range=1.0)
                total_psnr += psnr
                total_ssim += ssim
                count += 1
        avg_loss = epoch_loss / len(train_loader)
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.6f} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
    return loss_history

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

def show_examples(model, test_images, device, noise_levels, num_examples=3):
    model.eval()
    idxs = random.sample(range(len(test_images)), min(num_examples, len(test_images)))
    fig, axes = plt.subplots(len(noise_levels), 3 * num_examples, figsize=(4 * num_examples, 4 * len(noise_levels)))
    if len(noise_levels) == 1:
        axes = axes.reshape(1, -1)
    for row, sigma in enumerate(noise_levels):
        for col, idx in enumerate(idxs):
            clean = test_images[idx]
            noisy = add_rayleigh_noise(clean, scale=sigma)
            noisy_tensor = torch.FloatTensor(noisy).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                denoised_tensor = model(noisy_tensor)
                denoised = denoised_tensor.squeeze().cpu().numpy()
            axes[row, col * 3].imshow(clean, cmap='gray')
            axes[row, col * 3].set_title(f'Clean')
            axes[row, col * 3].axis('off')
            axes[row, col * 3 + 1].imshow(noisy, cmap='gray')
            axes[row, col * 3 + 1].set_title(f'Noisy σ={sigma}')
            axes[row, col * 3 + 1].axis('off')
            axes[row, col * 3 + 2].imshow(denoised, cmap='gray')
            axes[row, col * 3 + 2].set_title('Denoised')
            axes[row, col * 3 + 2].axis('off')
    plt.tight_layout()
    plt.show()

def evaluate_model_on_noise_level(model, test_images, device, sigma, num_examples=3):
    model.eval()
    psnr_list = []
    ssim_list = []
    idxs = random.sample(range(len(test_images)), min(num_examples, len(test_images)))
    fig, axes = plt.subplots(3, num_examples, figsize=(4 * num_examples, 8))
    for i, idx in enumerate(idxs):
        clean = test_images[idx]
        noisy = add_rayleigh_noise(clean, scale=sigma)
        noisy_tensor = torch.FloatTensor(noisy).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            denoised_tensor = model(noisy_tensor)
            denoised = denoised_tensor.squeeze().cpu().numpy()
        psnr = peak_signal_noise_ratio(clean, denoised, data_range=1.0)
        ssim = structural_similarity(clean, denoised, data_range=1.0)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        axes[0, i].imshow(clean, cmap='gray')
        axes[0, i].set_title('Clean')
        axes[0, i].axis('off')
        axes[1, i].imshow(noisy, cmap='gray')
        axes[1, i].set_title(f'Noisy σ={sigma}')
        axes[1, i].axis('off')
        axes[2, i].imshow(denoised, cmap='gray')
        axes[2, i].set_title(f'Denoised\nPSNR: {psnr:.2f}\nSSIM: {ssim:.3f}')
        axes[2, i].axis('off')
    plt.suptitle(f'Denoising Results at σ={sigma}')
    plt.tight_layout()
    plt.show()
    return np.mean(psnr_list), np.mean(ssim_list)

def run_rayleigh_noise_experiments(device):
    data_path = "Data/test"
    train_path = "Data/train"
    num_epochs = 100
    batch_size = 16

    print("Loading training images...")
    train_images = load_ultrasound_data(train_path, image_size=(128, 128))
    print(f"Loaded {len(train_images)} training images.")
    print("Loading test images...")
    test_images = load_ultrasound_data(data_path, image_size=(128, 128))
    print(f"Loaded {len(test_images)} test images.")

    # For each original image, create 6 noisy versions (one for each sigma)
    train_noisy = []
    train_clean = []
    for img in train_images:
        for sigma in noise_levels:
            noisy = add_rayleigh_noise(img, scale=sigma)
            train_noisy.append(noisy)
            train_clean.append(img)
    print(f"Training set size (noisy/clean pairs): {len(train_noisy)}")

    # Prepare test set: just use the clean images (we'll add noise on the fly for display)
    test_clean = test_images

    # Create datasets and dataloaders
    train_dataset = UltrasoundDataset(train_noisy, train_clean)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, loss ONCE
    model = CNNAutoencoderLarge().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    print("Training...")
    loss_history = train_model(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)

    # Plot loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    # Show denoising examples and evaluate for all noise levels
    print("\nEvaluating and displaying denoising examples for all noise levels...")
    for sigma in noise_levels:
        print(f"\n--- Noise Level σ={sigma} ---")
        avg_psnr, avg_ssim = evaluate_model_on_noise_level(model, test_clean, device, sigma, num_examples=3)
        print(f"Test set (Rayleigh noise σ={sigma}): PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}")

    # Save one checkpoint
    model_save_path = "ultrasound_autoencoder_all_sigmas.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved as {model_save_path}")

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