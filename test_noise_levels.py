import numpy as np
import torch
from data_utils import load_ultrasound_data
from noise_utils import add_rayleigh_noise, add_Gaussian_noise
from model import UltrasoundAutoencoder
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Set device
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def load_model(model_path, device):
    model = UltrasoundAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate(model, noisy_images, clean_images, device):
    psnr_list = []
    ssim_list = []
    with torch.no_grad():
        for noisy, clean in zip(noisy_images, clean_images):
            noisy_tensor = torch.FloatTensor(noisy).unsqueeze(0).unsqueeze(0).to(device)
            clean_tensor = torch.FloatTensor(clean).unsqueeze(0).unsqueeze(0).to(device)
            output = model(noisy_tensor).squeeze().cpu().numpy()
            clean_np = clean_tensor.squeeze().cpu().numpy()
            psnr = peak_signal_noise_ratio(clean_np, output, data_range=1.0)
            ssim = structural_similarity(clean_np, output, data_range=1.0)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
    return np.mean(psnr_list), np.mean(ssim_list)

def main():
    device = get_device()
    print(f"Using device: {device}")
    model_path = "/Users/sebmcmorran/SummerProject2025/Autoencoder/code/ultrasound_autoencoder_all_sigmas.pth"
    data_path = "Data/test"
    noise_levels = [0.05, 0.1, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    noise_type = ["rayleigh"]

    print("Loading test images...")
    images = load_ultrasound_data(data_path)
    print(f"Loaded {len(images)} images.")
    model = load_model(model_path, device)

    print(f"\nTesting noise type: {noise_type}")
    
    for sigma in noise_levels:
        noisy_images = [add_rayleigh_noise(img, scale=sigma) for img in images]
        avg_psnr, avg_ssim = evaluate(model, noisy_images, images, device)
        print(f"  Noise level {sigma}: PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}")

if __name__ == "__main__":
    main() 