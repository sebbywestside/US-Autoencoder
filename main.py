import numpy as np
import torch
from torch.utils.data import DataLoader
from data_utils import load_ultrasound_data
from noise_utils import generate_noisy_images
from filters import median_filter, mean_filter, gaussian_filter_img, bilateral_filter, frost_filter, kuan_filter, lee_filter
from model import UltrasoundAutoencoder
from dataset import UltrasoundDataset
from train import train_model, evaluate_model
from visualize import plot_results, compare_filters
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

# Set device
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    device = get_device()
    print(f"Using device: {device}")

    # Load your data
    data_path = "/Users/sebmcmorran/SummerProject2025/Autoencoder/Data/test"
    print("Loading images...")
    images = load_ultrasound_data(data_path)
    print(f"Loaded {len(images)} images")

    if len(images) == 0:
        print("No images found! Check your path and file extensions.")
        return

    # Generate noisy images
    print("Generating noisy images...")
    noised_images, clean_images = generate_noisy_images(images)
    print(f"Generated {len(noised_images)} noisy images and {len(clean_images)} clean images.")

    # Split data
    split_idx = int(0.8 * len(noised_images))
    train_noisy = noised_images[:split_idx]
    train_clean = clean_images[:split_idx]
    test_noisy = noised_images[split_idx:]
    test_clean = clean_images[split_idx:]
    print(f"Training set: {len(train_noisy)} images")
    print(f"Test set: {len(test_noisy)} images")

    # Create datasets and dataloaders
    train_dataset = UltrasoundDataset(train_noisy, train_clean)
    test_dataset = UltrasoundDataset(test_noisy, test_clean)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Test loader batches: {len(test_loader)}")

    # Initialize model
    model = UltrasoundAutoencoder().to(device)
    print("\nModel Summary:")
    print(model)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("\nStarting training...")
    train_losses = train_model(model, train_loader, criterion, optimizer, device, num_epochs=50)

    # Evaluate the model
    print("\nEvaluating model...")
    avg_psnr, avg_ssim = evaluate_model(model, test_loader, device)
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    # Show results
    print("\nVisualizing results...")
    plot_results(model, test_loader, device)

    # Compare with classic filters
    print("\nComparing with classic filters...")
    compare_filters(model, test_loader, device)

    # Save the model
    torch.save(model.state_dict(), 'ultrasound_autoencoder.pth')
    print("\nModel saved as 'ultrasound_autoencoder.pth'")

if __name__ == "__main__":
    main()

