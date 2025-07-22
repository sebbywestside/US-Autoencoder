#!/usr/bin/env python3
"""
run_comparison.py
Simple script to run the model comparison and display results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_comparison import ModelComparison
import pandas as pd
import numpy as np

def display_results_table(stats):
    """Display results in a nice table format"""
    print("\n" + "="*100)
    print("MODEL COMPARISON RESULTS")
    print("="*100)
    
    # Create a comprehensive results table
    results_data = []
    noise_levels = [0.1, 0.25, 0.5, 0.75]
    
    # Debug: Print the structure of stats to understand the keys
    print("Debug: Available models and their noise level keys:")
    for model_name, model_stats in stats.items():
        print(f"  {model_name}: {list(model_stats.keys())}")
    
    for model_name in stats.keys():
        for sigma in noise_levels:
            # The keys are stored as floats, not strings
            if sigma in stats[model_name]:
                try:
                    psnr = stats[model_name][sigma]['psnr_mean']
                    ssim = stats[model_name][sigma]['ssim_mean']
                    psnr_std = stats[model_name][sigma]['psnr_std']
                    ssim_std = stats[model_name][sigma]['ssim_std']
                    
                    results_data.append({
                        'Model': model_name,
                        'Noise_Level': sigma,
                        'PSNR_Mean': f"{psnr:.2f}",
                        'PSNR_Std': f"{psnr_std:.2f}",
                        'SSIM_Mean': f"{ssim:.4f}",
                        'SSIM_Std': f"{ssim_std:.4f}"
                    })
                except KeyError as e:
                    print(f"Debug: Missing key {e} for model {model_name}, sigma {sigma}")
                    continue
            else:
                print(f"Debug: No data found for model {model_name}, sigma {sigma}")
    
    # Create DataFrame and display
    if results_data:
        df = pd.DataFrame(results_data)
        print(f"Debug: DataFrame columns: {list(df.columns)}")
        print(f"Debug: DataFrame shape: {df.shape}")
        
        # Display by noise level
        for sigma in noise_levels:
            print(f"\nNoise Level σ = {sigma:.2f}:")
            print("-" * 80)
            sigma_data = df[df['Noise_Level'] == sigma]
            if not sigma_data.empty:
                print(sigma_data[['Model', 'PSNR_Mean', 'PSNR_Std', 'SSIM_Mean', 'SSIM_Std']].to_string(index=False))
                
                # Highlight best performance
                try:
                    best_psnr_idx = sigma_data['PSNR_Mean'].astype(float).idxmax()
                    best_model = sigma_data.loc[best_psnr_idx, 'Model']
                    best_psnr = sigma_data.loc[best_psnr_idx, 'PSNR_Mean']
                    print(f"\n🏆 Best PSNR: {best_model} ({best_psnr} dB)")
                except Exception as e:
                    print(f"No valid data for this noise level: {e}")
            else:
                print("No data available for this noise level")
    else:
        print("No results data found!")

def display_summary_statistics(stats):
    """Display summary statistics"""
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    summary_data = []
    noise_levels = [0.1, 0.25, 0.5, 0.75]
    
    for model_name in stats.keys():
        psnr_values = []
        ssim_values = []
        
        for sigma in noise_levels:
            # The keys are stored as floats, not strings
            if sigma in stats[model_name]:
                psnr_values.append(stats[model_name][sigma]['psnr_mean'])
                ssim_values.append(stats[model_name][sigma]['ssim_mean'])
        
        if psnr_values:
            avg_psnr = np.mean(psnr_values)
            avg_ssim = np.mean(ssim_values)
            std_psnr = np.std(psnr_values)
            std_ssim = np.std(ssim_values)
            
            summary_data.append({
                'Model': model_name,
                'Avg_PSNR': f"{avg_psnr:.2f} ± {std_psnr:.2f}",
                'Avg_SSIM': f"{avg_ssim:.4f} ± {std_ssim:.4f}",
                'Best_PSNR': f"{max(psnr_values):.2f}",
                'Worst_PSNR': f"{min(psnr_values):.2f}",
                'Range_PSNR': f"{max(psnr_values) - min(psnr_values):.2f}"
            })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))
        
        # Find best overall model
        try:
            best_model = max(summary_data, key=lambda x: float(x['Avg_PSNR'].split()[0]))
            print(f"\n🏆 Best Overall Model: {best_model['Model']}")
            print(f"   Average PSNR: {best_model['Avg_PSNR']} dB")
            print(f"   Average SSIM: {best_model['Avg_SSIM']}")
        except:
            print("Could not determine best model")
    else:
        print("No summary data available")

def main():
    """Main function to run comparison and display results"""
    print("🚀 Starting Model Comparison...")
    print("This will test all three models across noise levels 0.05 to 0.8")
    print("Testing on ultrasound denoising models...")
    
    try:
        # Initialize comparison
        comparison = ModelComparison()
        
        # Run comparison with limited images for faster testing
        print("\n📊 Running comprehensive evaluation...")
        results = comparison.run_comparison(max_images=None)  # Limit for faster testing
        
        # Calculate statistics
        print("\n📈 Calculating statistics...")
        stats = comparison.calculate_statistics(results)
        
        # Display results
        display_results_table(stats)
        display_summary_statistics(stats)
        
        # Save results
        print("\n💾 Saving results...")
        stats_file, results_file = comparison.save_results(results, stats)
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        plot_file = comparison.create_visualizations(stats)
        
        print(f"\n✅ Comparison completed successfully!")
        print(f"📁 Results saved to: {stats_file}")
        print(f"📊 Plots saved to: {plot_file}")
        print(f"\n📋 Next steps:")
        print(f"   1. Run the MATLAB script: matlab_analysis.m")
        print(f"   2. Check the 'results' folder for detailed outputs")
        print(f"   3. Review the generated plots and statistics")
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        print("Please check that all model files exist in the 'models' directory")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 