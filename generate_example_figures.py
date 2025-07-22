import os
from model_comparison import ModelComparison
from data_utils import load_ultrasound_data

if __name__ == "__main__":
    # Number of example images to visualize
    num_examples = 3  # Change as desired
    output_dir = "results/example_figures"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize comparison (loads models)
    comparison = ModelComparison()

    # Load test images
    test_images = load_ultrasound_data(comparison.data_path, image_size=comparison.image_size)
    print(f"Loaded {len(test_images)} test images.")

    # For each example image, run denoising and visualize
    for example_idx in range(num_examples):
        print(f"Generating figures for example image {example_idx}...")
        # Run denoising for all models at all noise levels for this image only
        results = {}
        for model_name, model in comparison.models.items():
            # Determine noise type based on model name
            if 'Rayleigh' in model_name:
                noise_type = 'rayleigh'
            elif 'SpeckleGauss' in model_name:
                noise_type = 'combined'
            else:
                noise_type = 'rayleigh'  # Default
            # Evaluate this model on this single image
            results[model_name] = comparison.evaluate_model(model, model_name, [test_images[example_idx]], noise_type)
        # Visualize for this image
        comparison.visualize_denoising_examples(results, output_dir=output_dir, image_idx=0)
    print(f"Example figures saved to {output_dir}") 