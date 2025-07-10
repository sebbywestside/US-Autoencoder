import numpy as np
import torch
from train import run_rayleigh_noise_experiments

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    print(f"Using device: {device}")

def main():
    device = get_device()
    
    run_rayleigh_noise_experiments(device)

if __name__ == "__main__":
    main()

