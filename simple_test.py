import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
import random

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the PyTorch Autoencoder (same as in training)
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

def load_image(image_path, target_size=(64, 64)):
    """Load and preprocess a single image"""
    try:
        img = Image.open(image_path).convert('L')
        img = img.resize(target_size)
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array
    except Exception as e:
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, target_size)
            img_array = img.astype(np.float32) / 255.0
            return img_array
        except Exception as e2:
            print(f"Error loading {image_path}: {e2}")
            return None

def add_rayleigh_noise(image, noise_level=0.2):
    """Add Rayleigh noise to image"""
    noise = np.random.rayleigh(scale=noise_level, size=image.shape)
    noisy = image * (1 + noise)
    return np.clip(noisy, 0.0, 1.0)

def load_model(model_path='ultrasound_autoencoder.pth'):
    """Load the trained autoencoder model"""
    model = UltrasoundAutoencoder().to(device)
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print(f"Model file {model_path} not found!")
        return None

def test_image(model, image_path, noise_level=0.2):
    """Test the autoencoder on a single image with Rayleigh noise"""
    # Load image
    clean_image = load_image(image_path)
    if clean_image is None:
        return None
    
    # Add Rayleigh noise
    noisy_image = add_rayleigh_noise(clean_image, noise_level)
    
    # Convert to tensor
    noisy_tensor = torch.FloatTensor(noisy_image).unsqueeze(0).unsqueeze(0).to(device)
    
    # Denoise
    with torch.no_grad():
        denoised_tensor = model(noisy_tensor)
        denoised_image = denoised_tensor.squeeze().cpu().numpy()
    
    # Calculate metrics
    psnr = peak_signal_noise_ratio(clean_image, denoised_image, data_range=1.0)
    ssim = structural_similarity(clean_image, denoised_image, data_range=1.0)
    
    return clean_image, noisy_image, denoised_image, psnr, ssim

def test_multiple_images(model, data_path, num_images=5):
    """Test the autoencoder on multiple images with Rayleigh noise"""
    # Get list of image files
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
        image_files.extend([f for f in os.listdir(data_path) if f.lower().endswith(ext)])
    
    if not image_files:
        print(f"No image files found in {data_path}")
        return []
    
    # Randomly select images
    selected_files = random.sample(image_files, min(num_images, len(image_files)))
    
    results = []
    noise_levels = [0.1, 0.25, 0.5, 0.75]
    
    print(f"Testing on {len(selected_files)} images...")
    
    for i, image_file in enumerate(selected_files):
        print(f"Processing image {i+1}/{len(selected_files)}: {image_file}")
        image_path = os.path.join(data_path, image_file)
        
        for noise_level in noise_levels:
            result = test_image(model, image_path, noise_level)
            if result is not None:
                clean, noisy, denoised, psnr, ssim = result
                results.append({
                    'image_name': image_file,
                    'noise_level': noise_level,
                    'clean': clean,
                    'noisy': noisy,
                    'denoised': denoised,
                    'psnr': psnr,
                    'ssim': ssim
                })
    
    return results

def display_results(results):
    """Display the results"""
    if not results:
        print("No results to display")
        return
    
    # Get unique noise levels
    noise_levels = sorted(list(set([r['noise_level'] for r in results])))
    num_levels = len(noise_levels)
    
    fig, axes = plt.subplots(3, num_levels, figsize=(4*num_levels, 12))
    if num_levels == 1:
        axes = axes.reshape(3, 1)
    
    fig.suptitle(f'Autoencoder Results - Rayleigh Noise', fontsize=16)
    
    for i, noise_level in enumerate(noise_levels):
        level_results = [r for r in results if r['noise_level'] == noise_level]
        
        if level_results:
            # Use the first result for this noise level
            result = level_results[0]
            
            # Original image
            axes[0, i].imshow(result['clean'], cmap='gray')
            axes[0, i].set_title(f'Original\n{result["image_name"]}')
            axes[0, i].axis('off')
            
            # Noisy image
            axes[1, i].imshow(result['noisy'], cmap='gray')
            axes[1, i].set_title(f'Noisy (level={noise_level})')
            axes[1, i].axis('off')
            
            # Denoised image
            axes[2, i].imshow(result['denoised'], cmap='gray')
            axes[2, i].set_title(f'Denoised\nPSNR: {result["psnr"]:.2f}dB\nSSIM: {result["ssim"]:.3f}')
            axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def print_metrics_summary(results):
    """Print a summary of the metrics"""
    if not results:
        print("No results to summarize")
        return
    
    print("\n" + "="*60)
    print("METRICS SUMMARY")
    print("="*60)
    
    noise_levels = sorted(set([r['noise_level'] for r in results]))
    for level in noise_levels:
        level_results = [r for r in results if r['noise_level'] == level]
        
        psnr_values = [r['psnr'] for r in level_results]
        ssim_values = [r['ssim'] for r in level_results]
        
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        std_psnr = np.std(psnr_values)
        std_ssim = np.std(ssim_values)
        
        print(f"  Noise Level {level}:")
        print(f"    PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
        print(f"    SSIM: {avg_ssim:.3f} ± {std_ssim:.3f}")
        print(f"    Samples: {len(level_results)}")

def main():
    """Main function"""
    print("Autoencoder Testing Script")
    print("="*40)
    
    # Load the trained model
    model = load_model()
    if model is None:
        print("Could not load model. Please ensure the model file exists.")
        return
    
    # Test on test data
    test_path = "Data/test"
    if os.path.exists(test_path):
        print(f"\nTesting on {test_path}...")
        results = test_multiple_images(model, test_path, num_images=5)
        
        if results:
            # Display results
            display_results(results)
            
            # Print metrics summary
            print_metrics_summary(results)
        else:
            print("No test results obtained.")
    else:
        print(f"Test path {test_path} not found!")

if __name__ == "__main__":
    main() 