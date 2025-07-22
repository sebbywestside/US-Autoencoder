"""
main_updated.py
Updated main script to include GAN models alongside CNN and AE models.
"""

import numpy as np
import torch
from train import run_rayleigh_noise_experiments
from train_gan import run_gan_experiments
from model import CNNAutoencoderLarge, UltrasoundAutoencoder

def get_device():
    """Get the best available device"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device

def main():
    """Main function to train all models"""
    device = get_device()
    
    print("="*60)
    print("ULTRASOUND DENOISING MODEL COMPARISON")
    print("="*60)
    
    # Train traditional autoencoder models
    #print("\n1. Training Traditional Autoencoder Models...")
    #print("-" * 50)
    
    # Train Autoencoder
    #print("Training Autoencoder (AE)...")
    #run_rayleigh_noise_experiments(device, UltrasoundAutoencoder, "AE")
    
    # Train CNN Autoencoder  
    #print("\nTraining CNN Autoencoder...")
    #run_rayleigh_noise_experiments(device, CNNAutoencoderLarge, "CNN")
    
    # Train GAN models
    print("\n2. Training GAN Models...")
    print("-" * 50)
    
    # Train WGAN-GP
    print("Training WGAN-GP...")
    wgan_trainer, wgan_results = run_gan_experiments(device, "wgan_gp")
    
    # Train Denoising GAN
    print("\nTraining Denoising GAN...")
    dgan_trainer, dgan_results = run_gan_experiments(device, "denoising_gan")
    
    print("\n" + "="*60)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*60)
    
    # Print final comparison summary
    print("\nFINAL COMPARISON SUMMARY:")
    print("-" * 40)
    
    noise_levels = [0.1, 0.25, 0.50, 0.75]
    print(f"{'Model':<15} {'σ=0.1':<12} {'σ=0.25':<12} {'σ=0.5':<12} {'σ=0.75':<12}")
    print("-" * 65)
    
    # Note: For a complete comparison, you would need to modify the original
    # training functions to return results in the same format as GAN training
    print("Traditional AE  [Run model_comparison.py for detailed metrics]")
    print("CNN Large       [Run model_comparison.py for detailed metrics]")
    
    # GAN results
    for model_name, results in [("WGAN-GP", wgan_results), ("Denoising GAN", dgan_results)]:
        psnr_values = [f"{results[sigma]['psnr_mean']:.1f}" for sigma in noise_levels]
        print(f"{model_name:<15} {' dB       '.join(psnr_values)} dB")
    
    print("\nRecommendation: Run model_comparison.py for comprehensive comparison!")

if __name__ == "__main__":
    main()