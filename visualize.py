"""
visualize.py
Plotting and comparison functions for denoising results.
"""

import matplotlib.pyplot as plt
import torch
from filters import median_filter, mean_filter, gaussian_filter_img, bilateral_filter, frost_filter, kuan_filter, lee_filter

def plot_results(model, test_loader, device, num_samples=4):
    """Show clean, noisy, and denoised images for a batch from the test set."""
    model.eval()
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 8))
    with torch.no_grad():
        noisy, clean = next(iter(test_loader))
        noisy, clean = noisy.to(device), clean.to(device)
        outputs = model(noisy)
        for i in range(min(num_samples, noisy.shape[0])):
            axes[0, i].imshow(clean[i, 0].cpu().numpy(), cmap='gray')
            axes[0, i].set_title('Clean')
            axes[0, i].axis('off')
            axes[1, i].imshow(noisy[i, 0].cpu().numpy(), cmap='gray')
            axes[1, i].set_title('Noisy')
            axes[1, i].axis('off')
            axes[2, i].imshow(outputs[i, 0].cpu().numpy(), cmap='gray')
            axes[2, i].set_title('Denoised')
            axes[2, i].axis('off')
    plt.tight_layout()
    plt.show()

def compare_filters(model, test_loader, device):
    """Compare the autoencoder with classic filters and display results."""
    model.eval()
    with torch.no_grad():
        noisy, clean = next(iter(test_loader))
        noisy, clean = noisy[0, 0].cpu().numpy(), clean[0, 0].cpu().numpy()
        input_tensor = torch.FloatTensor(noisy).unsqueeze(0).unsqueeze(0).to(device)
        denoised = model(input_tensor).squeeze().cpu().numpy()
    median = median_filter(noisy)
    mean = mean_filter(noisy)
    gauss = gaussian_filter_img(noisy)
    bilateral = bilateral_filter(noisy)
    frost = frost_filter(noisy)
    kuan = kuan_filter(noisy)
    lee = lee_filter(noisy)
    titles = [
        'Original', 'Noisy', 'Autoencoder', 'Median (3x3)', 'Mean',
        'Gaussian', 'Bilateral', 'Frost', 'Kuan', 'Lee'
    ]
    images = [clean, noisy, denoised, median, mean, gauss, bilateral, frost, kuan, lee]
    plt.figure(figsize=(20, 6))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 5, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show() 