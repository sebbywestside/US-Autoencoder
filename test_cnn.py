import numpy as np
import torch
from model import CNNAutoencoderLarge
from data_utils import load_ultrasound_data
from noise_utils import add_rayleigh_noise
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import random

noise_levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
image_size = (128, 128)
model_path = "/Users/sebmcmorran/SummerProject2025/Autoencoder/ultrasound_CNN.pth"
data_path = "Data/test"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load test images
print("Loading test images...")
test_images = load_ultrasound_data(data_path, image_size=image_size)
print(f"Loaded {len(test_images)} test images.")

# Load model
model = CNNAutoencoderLarge().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

results = {sigma: [] for sigma in noise_levels}

# Evaluate on all noise levels
for sigma in noise_levels:
    print(f"\nEvaluating at noise level σ={sigma}...")
    for idx, clean in enumerate(test_images):
        noisy = add_rayleigh_noise(clean, scale=sigma)
        noisy_tensor = torch.FloatTensor(noisy).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            denoised_tensor = model(noisy_tensor)
            denoised = denoised_tensor.squeeze().cpu().numpy()
        psnr = peak_signal_noise_ratio(clean, denoised, data_range=1.0)
        ssim = structural_similarity(clean, denoised, data_range=1.0)
        results[sigma].append({
            'idx': idx,
            'psnr': psnr,
            'ssim': ssim,
            'clean': clean,
            'noisy': noisy,
            'denoised': denoised
        })

# Display best results for each noise level
for sigma in noise_levels:
    best = max(results[sigma], key=lambda x: x['psnr'])
    print(f"\nBest result for σ={sigma}: PSNR={best['psnr']:.2f}, SSIM={best['ssim']:.4f}, Image idx={best['idx']}")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(best['clean'], cmap='gray')
    axes[0].set_title('Clean')
    axes[0].axis('off')
    axes[1].imshow(best['noisy'], cmap='gray')
    axes[1].set_title(f'Noisy σ={sigma}')
    axes[1].axis('off')
    axes[2].imshow(best['denoised'], cmap='gray')
    axes[2].set_title(f"Denoised\nPSNR: {best['psnr']:.2f}\nSSIM: {best['ssim']:.3f}")
    axes[2].axis('off')
    plt.suptitle(f'Best Denoising Result at σ={sigma}')
    plt.tight_layout()
    plt.show()

# Print summary statistics
print("\n=== Summary Statistics ===")
for sigma in noise_levels:
    psnrs = [r['psnr'] for r in results[sigma]]
    ssims = [r['ssim'] for r in results[sigma]]
    print(f"σ={sigma}: PSNR={np.mean(psnrs):.2f}±{np.std(psnrs):.2f} dB, SSIM={np.mean(ssims):.4f}±{np.std(ssims):.4f}") 