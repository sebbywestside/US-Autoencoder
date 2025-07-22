#!/usr/bin/env python3
"""
debug_stats.py
Debug script to examine the statistics data structure.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_comparison import ModelComparison
import json

def debug_stats_structure():
    """Debug the statistics data structure"""
    print("🔍 Debugging statistics structure...")
    
    # Initialize comparison
    comparison = ModelComparison()
    
    # Run comparison with just a few images
    print("Running comparison with 5 images...")
    results = comparison.run_comparison(max_images=5)
    
    # Calculate statistics
    stats = comparison.calculate_statistics(results)
    
    # Print the structure
    print("\n📊 Statistics structure:")
    print("=" * 50)
    
    for model_name in stats.keys():
        print(f"\nModel: {model_name}")
        print(f"Keys: {list(stats[model_name].keys())}")
        
        for sigma_key in stats[model_name].keys():
            print(f"  {sigma_key}: {type(sigma_key)}")
            sigma_data = stats[model_name][sigma_key]
            print(f"    Data keys: {list(sigma_data.keys())}")
            print(f"    PSNR mean: {sigma_data['psnr_mean']}")
            print(f"    SSIM mean: {sigma_data['ssim_mean']}")
    
    # Save a sample to see the JSON structure
    sample_file = "debug_stats_sample.json"
    with open(sample_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n💾 Sample saved to: {sample_file}")
    
    return stats

if __name__ == "__main__":
    debug_stats_structure() 