"""
train.py
Training and evaluation functions for the autoencoder.
"""

import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50):
    """Train the model and return the loss history."""
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for noisy, clean in train_loader:
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
    """Evaluate the model and return average PSNR and SSIM."""
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