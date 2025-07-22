"""
model_comparison.py
Comprehensive comparison of ultrasound denoising models across different noise levels.
Tests all three models and generates statistics and visualizations.
"""

import numpy as np
import torch
import os
import json
from model import UltrasoundAutoencoder, CNNAutoencoderLarge
from data_utils import load_ultrasound_data
from noise_utils import add_rayleigh_noise, add_Gaussian_noise
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ModelComparison:
    def __init__(self, device=None):
        self.device = device if device else self._get_device()
        self.noise_levels = [0.1, 0.25, 0.5, 0.75]
        self.image_size = (128, 128)
        self.data_path = "Data/test"
        
        # Model paths (update as needed)
        self.model_paths = {
            'CNN01075': '/Users/sebmcmorran/SummerProject2025/Autoencoder/CNN01075.pth',
            'AE001075': '/Users/sebmcmorran/SummerProject2025/Autoencoder/AE01075.pth'
        }
        
        # Initialize models
        self.models = {}
        self._load_models()
        
    def _get_device(self):
        """Get the best available device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _load_models(self):
        """Load all models specified in self.model_paths"""
        print("Loading models...")
        for name, path in self.model_paths.items():
            try:
                if "CNN" in name:
                    model = CNNAutoencoderLarge().to(self.device)
                else:
                    model = UltrasoundAutoencoder().to(self.device)
                model.load_state_dict(torch.load(path, map_location=self.device))
                model.eval()
                self.models[name] = model
                print(f"✓ {name} model loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load {name} model: {e}")
    
    def evaluate_model(self, model, model_name, test_images, noise_type='rayleigh'):
        """Evaluate a single model on all noise levels"""
        print(f"\nEvaluating {model_name} model...")
        results = {}
        
        for sigma in self.noise_levels:
            print(f"  Testing noise level σ={sigma}...")
            sigma_results = []
            
            for idx, clean in enumerate(test_images):
                # Add noise
                if noise_type == 'rayleigh':
                    noisy = add_rayleigh_noise(clean, scale=sigma)
                elif noise_type == 'gaussian':
                    noisy = add_Gaussian_noise(clean, scale=sigma)
                else:
                    # Combined noise
                    noisy = add_rayleigh_noise(clean, scale=sigma)
                    noisy = add_Gaussian_noise(noisy, scale=sigma)
                
                # Prepare tensor
                noisy_tensor = torch.FloatTensor(noisy).unsqueeze(0).unsqueeze(0).to(self.device)
                
                # Denoise
                with torch.no_grad():
                    denoised_tensor = model(noisy_tensor)
                    denoised = denoised_tensor.squeeze().cpu().numpy()
                
                # Calculate metrics
                psnr = peak_signal_noise_ratio(clean, denoised, data_range=1.0)
                ssim = structural_similarity(clean, denoised, data_range=1.0)
                
                sigma_results.append({
                    'idx': idx,
                    'psnr': psnr,
                    'ssim': ssim,
                    'clean': clean,
                    'noisy': noisy,
                    'denoised': denoised
                })
            
            results[sigma] = sigma_results
        
        return results
    
    def run_comparison(self, max_images=100):
        """Run comprehensive comparison of all models"""
        print("Loading test images...")
        test_images = load_ultrasound_data(self.data_path, image_size=self.image_size)
        
        # Limit number of images for faster testing
        if len(test_images) > max_images:
            test_images = test_images[:max_images]
            print(f"Using first {max_images} images for testing")
        
        print(f"Loaded {len(test_images)} test images")
        
        # Evaluate all models
        all_results = {}
        
        for model_name, model in self.models.items():
            # Determine noise type based on model name
            if 'Rayleigh' in model_name:
                noise_type = 'rayleigh'
            elif 'SpeckleGauss' in model_name:
                noise_type = 'combined'
            else:  # CNN model
                noise_type = 'rayleigh'  # Default for CNN
            
            results = self.evaluate_model(model, model_name, test_images, noise_type)
            all_results[model_name] = results
        
        return all_results
    
    def calculate_statistics(self, results):
        """Calculate comprehensive statistics for all models"""
        stats = {}
        
        for model_name, model_results in results.items():
            model_stats = {}
            
            for sigma in self.noise_levels:
                sigma_results = model_results[sigma]
                psnrs = [r['psnr'] for r in sigma_results]
                ssims = [r['ssim'] for r in sigma_results]
                
                model_stats[sigma] = {
                    'psnr_mean': np.mean(psnrs),
                    'psnr_std': np.std(psnrs),
                    'psnr_min': np.min(psnrs),
                    'psnr_max': np.max(psnrs),
                    'ssim_mean': np.mean(ssims),
                    'ssim_std': np.std(ssims),
                    'ssim_min': np.min(ssims),
                    'ssim_max': np.max(ssims),
                    'num_samples': len(psnrs)
                }
            
            stats[model_name] = model_stats
        
        return stats
    
    def save_results(self, results, stats, output_dir="results"):
        """Save results and statistics to files"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert float keys to strings for JSON compatibility
        stats_for_json = {}
        for model_name, model_stats in stats.items():
            stats_for_json[model_name] = {}
            for sigma, sigma_stats in model_stats.items():
                sigma_str = f"{sigma:.2f}"
                stats_for_json[model_name][sigma_str] = sigma_stats
        
        # Save statistics
        stats_file = os.path.join(output_dir, f"model_comparison_stats_{timestamp}.json")
        with open(stats_file, 'w') as f:
            json.dump(stats_for_json, f, indent=2)
        print(f"Statistics saved to {stats_file}")
        
        # Save detailed results (without images to save space)
        results_summary = {}
        for model_name, model_results in results.items():
            results_summary[model_name] = {}
            for sigma, sigma_results in model_results.items():
                sigma_str = f"{sigma:.2f}"
                results_summary[model_name][sigma_str] = [
                    {'idx': r['idx'], 'psnr': r['psnr'], 'ssim': r['ssim']}
                    for r in sigma_results
                ]
        
        results_file = os.path.join(output_dir, f"model_comparison_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"Results saved to {results_file}")
        
        return stats_file, results_file
    
    def create_visualizations(self, stats, output_dir="results"):
        """Create comprehensive visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Comparison: PSNR and SSIM vs Noise Level', fontsize=16)
        
        # PSNR plot
        ax1 = axes[0, 0]
        for model_name in stats.keys():
            psnr_means = [stats[model_name][sigma]['psnr_mean'] for sigma in self.noise_levels]
            psnr_stds = [stats[model_name][sigma]['psnr_std'] for sigma in self.noise_levels]
            ax1.errorbar(self.noise_levels, psnr_means, yerr=psnr_stds, 
                        marker='o', label=model_name, linewidth=2, capsize=5)
        ax1.set_xlabel('Noise Level (σ)')
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('Peak Signal-to-Noise Ratio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # SSIM plot
        ax2 = axes[0, 1]
        for model_name in stats.keys():
            ssim_means = [stats[model_name][sigma]['ssim_mean'] for sigma in self.noise_levels]
            ssim_stds = [stats[model_name][sigma]['ssim_std'] for sigma in self.noise_levels]
            ax2.errorbar(self.noise_levels, ssim_means, yerr=ssim_stds, 
                        marker='s', label=model_name, linewidth=2, capsize=5)
        ax2.set_xlabel('Noise Level (σ)')
        ax2.set_ylabel('SSIM')
        ax2.set_title('Structural Similarity Index')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Performance comparison table
        ax3 = axes[1, 0]
        ax3.axis('tight')
        ax3.axis('off')
        
        # Create table data
        table_data = []
        headers = ['Model', 'Avg PSNR', 'Avg SSIM', 'Best σ']
        
        for model_name in stats.keys():
            avg_psnr = np.mean([stats[model_name][sigma]['psnr_mean'] for sigma in self.noise_levels])
            avg_ssim = np.mean([stats[model_name][sigma]['ssim_mean'] for sigma in self.noise_levels])
            
            # Find best noise level
            best_sigma = max(stats[model_name].keys(), 
                           key=lambda s: stats[model_name][s]['psnr_mean'])
            
            table_data.append([
                model_name,
                f"{avg_psnr:.2f}",
                f"{avg_ssim:.4f}",
                f"{best_sigma}"
            ])
        
        table = ax3.table(cellText=table_data, colLabels=headers, 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax3.set_title('Overall Performance Summary')
        
        # Performance degradation analysis
        ax4 = axes[1, 1]
        
        # Calculate performance degradation (how much PSNR drops from lowest to highest noise)
        degradation_data = []
        model_names = list(stats.keys())
        
        for model_name in model_names:
            psnr_values = [stats[model_name][sigma]['psnr_mean'] for sigma in self.noise_levels]
            degradation = (psnr_values[0] - psnr_values[-1]) / psnr_values[0] * 100  # Percentage drop
            degradation_data.append(degradation)
        
        # Create bar chart of performance degradation
        bars = ax4.bar(model_names, degradation_data, color=['#ff7f0e', '#2ca02c', '#d62728'])
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Performance Degradation (%)')
        ax4.set_title('Performance Degradation\n(Low to High Noise)')
        
        # Add value labels on bars
        for bar, value in zip(bars, degradation_data):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, max(degradation_data) * 1.1)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(output_dir, f"model_comparison_plots_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {plot_file}")
        
        # Create additional detailed plots
        self._create_detailed_plots(stats, output_dir, timestamp)
        
        return plot_file
    
    def _create_detailed_plots(self, stats, output_dir, timestamp):
        """Create additional detailed plots"""
        # PSNR heatmap
        plt.figure(figsize=(12, 8))
        
        # Prepare data for heatmap
        model_names = list(stats.keys())
        psnr_matrix = np.zeros((len(model_names), len(self.noise_levels)))
        
        for i, model_name in enumerate(model_names):
            for j, sigma in enumerate(self.noise_levels):
                psnr_matrix[i, j] = stats[model_name][sigma]['psnr_mean']
        
        # Create heatmap
        sns.heatmap(psnr_matrix, 
                    xticklabels=[f"σ={s}" for s in self.noise_levels],
                    yticklabels=model_names,
                    annot=True, fmt='.2f', cmap='viridis',
                    cbar_kws={'label': 'PSNR (dB)'})
        plt.title('PSNR Performance Heatmap')
        plt.xlabel('Noise Level')
        plt.ylabel('Model')
        
        heatmap_file = os.path.join(output_dir, f"psnr_heatmap_{timestamp}.png")
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # SSIM heatmap
        plt.figure(figsize=(12, 8))
        ssim_matrix = np.zeros((len(model_names), len(self.noise_levels)))
        
        for i, model_name in enumerate(model_names):
            for j, sigma in enumerate(self.noise_levels):
                ssim_matrix[i, j] = stats[model_name][sigma]['ssim_mean']
        
        sns.heatmap(ssim_matrix, 
                    xticklabels=[f"σ={s}" for s in self.noise_levels],
                    yticklabels=model_names,
                    annot=True, fmt='.4f', cmap='plasma',
                    cbar_kws={'label': 'SSIM'})
        plt.title('SSIM Performance Heatmap')
        plt.xlabel('Noise Level')
        plt.ylabel('Model')
        
        ssim_heatmap_file = os.path.join(output_dir, f"ssim_heatmap_{timestamp}.png")
        plt.savefig(ssim_heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Detailed plots saved to {output_dir}")
    
    def print_summary(self, stats):
        """Print comprehensive summary of results"""
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        # Overall performance
        print("\nOVERALL PERFORMANCE:")
        print("-" * 50)
        for model_name in stats.keys():
            avg_psnr = np.mean([stats[model_name][sigma]['psnr_mean'] for sigma in self.noise_levels])
            avg_ssim = np.mean([stats[model_name][sigma]['ssim_mean'] for sigma in self.noise_levels])
            print(f"{model_name:25s}: PSNR={avg_psnr:6.2f} dB, SSIM={avg_ssim:.4f}")
        
        # Best performing model
        best_model = max(stats.keys(), 
                        key=lambda m: np.mean([stats[m][sigma]['psnr_mean'] for sigma in self.noise_levels]))
        print(f"\nBest overall model: {best_model}")
        
        # Performance by noise level
        print("\nPERFORMANCE BY NOISE LEVEL:")
        print("-" * 50)
        for sigma in self.noise_levels:
            print(f"\nNoise Level σ={sigma}:")
            for model_name in stats.keys():
                psnr = stats[model_name][sigma]['psnr_mean']
                ssim = stats[model_name][sigma]['ssim_mean']
                print(f"  {model_name:25s}: PSNR={psnr:6.2f} dB, SSIM={ssim:.4f}")
        
        # Best model for each noise level
        print("\nBEST MODEL BY NOISE LEVEL:")
        print("-" * 50)
        for sigma in self.noise_levels:
            best_model_sigma = max(stats.keys(), 
                                 key=lambda m: stats[m][sigma]['psnr_mean'])
            best_psnr = stats[best_model_sigma][sigma]['psnr_mean']
            print(f"σ={sigma:4.2f}: {best_model_sigma:25s} (PSNR={best_psnr:.2f} dB)")

def main():
    """Main function to run the model comparison"""
    print("Starting Model Comparison...")
    
    # Initialize comparison
    comparison = ModelComparison()
    
    # Run comparison
    results = comparison.run_comparison(max_images=50)  # Limit for faster testing
    
    # Calculate statistics
    stats = comparison.calculate_statistics(results)
    
    # Print summary
    comparison.print_summary(stats)
    
    # Save results
    stats_file, results_file = comparison.save_results(results, stats)
    
    # Create visualizations
    plot_file = comparison.create_visualizations(stats)
    
    print(f"\nComparison completed successfully!")
    print(f"Results saved to: {stats_file}")
    print(f"Plots saved to: {plot_file}")

if __name__ == "__main__":
    main() 