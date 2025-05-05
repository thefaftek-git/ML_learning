"""
Image Generation Script

This script uses a trained image generator model to create new SVG wireframe images.
It can generate images from random latent vectors or interpolate between different points.
It also supports conditional image generation for models trained with multiple image types.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from model import ImageGenerator
from utils import save_as_svg, create_placeholder_svg

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate images using the trained model')
    parser.add_argument('--model', type=str, default='latest',
                        help='Path to the trained model file or "latest" to use the latest run')
    parser.add_argument('--run-guid', type=str, default=None,
                        help='Specific run GUID to use for generation')
    parser.add_argument('--model-file', type=str, default='generator_final.pt',
                        help='Model file to use within the run folder (default: generator_final.pt)')
    parser.add_argument('--count', type=int, default=5,
                        help='Number of images to generate')
    parser.add_argument('--output-dir', type=str, default='data/generated',
                        help='Directory to save generated images')
    parser.add_argument('--image-size', type=int, default=128,
                        help='Size of the square output image')
    parser.add_argument('--latent-dim', type=int, default=100,
                        help='Dimension of the latent space')
    parser.add_argument('--mode', type=str, choices=['random', 'interpolate', 'all-conditions'], default='random',
                        help='Generation mode: random samples, interpolation between points, or all conditions')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of interpolation steps (only for interpolate mode)')
    parser.add_argument('--condition', type=str, default=None,
                        help='Specify which condition to use for generation (condition ID or filename)')
    parser.add_argument('--mapping-file', type=str, default=None,
                        help='Path to the condition mapping file (JSON). If not provided, will look in the model directory.')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode to select conditions')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Base directory where model runs are stored')
    
    return parser.parse_args()

def load_condition_mapping(mapping_path):
    """
    Load the condition mapping from a JSON file.
    
    Args:
        mapping_path: Path to the condition mapping JSON file
    
    Returns:
        Dictionary mapping condition IDs to image info, or None if file doesn't exist
    """
    if not os.path.exists(mapping_path):
        print(f"Condition mapping file not found: {mapping_path}")
        print("The model may not have been trained with multiple conditions.")
        return None
    
    try:
        with open(mapping_path, 'r') as f:
            # Convert condition IDs from strings back to integers
            mapping = json.load(f)
            return {int(k): v for k, v in mapping.items()}
    except Exception as e:
        print(f"Error loading condition mapping: {e}")
        return None

def resolve_condition(condition_arg, condition_mapping):
    """
    Resolve a condition argument to a condition ID.
    
    Args:
        condition_arg: String specifying condition (can be ID or filename)
        condition_mapping: Loaded condition mapping
    
    Returns:
        Resolved condition ID (integer)
    """
    if condition_mapping is None:
        # No mapping available, default to condition 0
        return 0
    
    # If it's a number, treat as direct condition ID
    try:
        condition_id = int(condition_arg)
        if condition_id in condition_mapping:
            return condition_id
    except (ValueError, TypeError):
        pass
    
    # Try to match by filename
    for cond_id, info in condition_mapping.items():
        if condition_arg.lower() in info['filename'].lower():
            print(f"Matched condition '{condition_arg}' to '{info['filename']}' (ID: {cond_id})")
            return cond_id
    
    # No match found, default to first condition
    print(f"Could not match condition '{condition_arg}', defaulting to condition 0")
    return 0
        
def select_condition_interactive(condition_mapping):
    """
    Let the user select a condition interactively.
    
    Args:
        condition_mapping: Loaded condition mapping
        
    Returns:
        Selected condition ID
    """
    if condition_mapping is None:
        print("No condition mapping available. Using default condition (0).")
        return 0
    
    print("\nAvailable conditions:")
    for cond_id, info in condition_mapping.items():
        print(f"  [{cond_id}] {info['filename']}")
    
    while True:
        try:
            selection = input("\nEnter condition ID to generate: ")
            condition_id = int(selection)
            if condition_id in condition_mapping:
                return condition_id
            else:
                print(f"Invalid selection. Please enter a valid condition ID.")
        except ValueError:
            print("Please enter a valid number.")

def generate_random_images(generator, count, output_dir, condition_id=0, condition_mapping=None):
    """Generate a specified number of random images with the given condition."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get condition name if available
    condition_name = "default"
    if condition_mapping and condition_id in condition_mapping:
        condition_name = condition_mapping[condition_id]['filename'].split('.')[0]
    
    for i in range(count):
        # Generate random latent vector
        latent_vector = np.random.normal(0, 1, (1, generator.latent_dim))
        
        # Generate the image with the specified condition
        generated_image = generator.generate_image(latent_vector, condition_id)
        
        # Save as PNG
        output_path = os.path.join(output_dir, f"{condition_name}_random_{i+1}.png")
        plt.imsave(output_path, generated_image[:,:,0], cmap='gray')
        
        # Save as SVG
        svg_path = os.path.join(output_dir, f"{condition_name}_random_{i+1}.svg")
        save_as_svg(generated_image, svg_path)
        
        print(f"Generated image (condition: {condition_id}) saved to {output_path} and {svg_path}")

def generate_interpolated_images(generator, steps, output_dir, condition_id=0, condition_mapping=None):
    """Generate a sequence of images by interpolating between two random points."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get condition name if available
    condition_name = "default"
    if condition_mapping and condition_id in condition_mapping:
        condition_name = condition_mapping[condition_id]['filename'].split('.')[0]
    
    # Generate two random latent vectors
    start_vector = np.random.normal(0, 1, (1, generator.latent_dim))
    end_vector = np.random.normal(0, 1, (1, generator.latent_dim))
    
    # Create interpolation steps
    alphas = np.linspace(0, 1, steps)
    
    # Generate images for each interpolation step
    for i, alpha in enumerate(alphas):
        # Interpolate between the two vectors
        latent_vector = start_vector * (1 - alpha) + end_vector * alpha
        
        # Generate the image with the specified condition
        generated_image = generator.generate_image(latent_vector, condition_id)
        
        # Save as PNG
        output_path = os.path.join(output_dir, f"{condition_name}_interpolate_{i+1:02d}.png")
        plt.imsave(output_path, generated_image[:,:,0], cmap='gray')
        
        # Save as SVG
        svg_path = os.path.join(output_dir, f"{condition_name}_interpolate_{i+1:02d}.svg")
        save_as_svg(generated_image, svg_path)
        
        print(f"Generated interpolation step {i+1}/{steps} (condition: {condition_id}) saved")
    
    # Create a grid visualization of the interpolation
    plt.figure(figsize=(12, 4))
    for i, alpha in enumerate(alphas):
        if i >= 10:  # Only show up to 10 images in the grid
            break
        plt.subplot(2, 5, i+1)
        
        # Interpolate between the two vectors
        latent_vector = start_vector * (1 - alpha) + end_vector * alpha
        
        # Generate the image with the specified condition
        generated_image = generator.generate_image(latent_vector, condition_id)
        
        plt.imshow(generated_image[:,:,0], cmap='gray')
        plt.title(f"Step {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    grid_path = os.path.join(output_dir, f"{condition_name}_interpolation_grid.png")
    plt.savefig(grid_path)
    plt.close()
    print(f"Interpolation grid saved to {grid_path}")

def generate_all_conditions(generator, output_dir, condition_mapping):
    """Generate images for all available conditions using the same latent vector."""
    os.makedirs(output_dir, exist_ok=True)
    
    if condition_mapping is None:
        print("No condition mapping available. Cannot generate all conditions.")
        return
    
    # Use a fixed latent vector for consistent comparison
    latent_vector = np.random.normal(0, 1, (1, generator.latent_dim))
    
    # Create a figure to show all generated images
    num_conditions = len(condition_mapping)
    fig_cols = min(5, num_conditions)
    fig_rows = (num_conditions + fig_cols - 1) // fig_cols
    
    plt.figure(figsize=(fig_cols * 3, fig_rows * 3))
    
    for i, (condition_id, info) in enumerate(condition_mapping.items()):
        condition_name = info['filename'].split('.')[0]
        
        # Generate image with this condition
        generated_image = generator.generate_image(latent_vector, condition_id)
        
        # Save individual image
        output_path = os.path.join(output_dir, f"condition_{condition_id}_{condition_name}.png")
        plt.imsave(output_path, generated_image[:,:,0], cmap='gray')
        
        # Save as SVG
        svg_path = os.path.join(output_dir, f"condition_{condition_id}_{condition_name}.svg")
        save_as_svg(generated_image, svg_path)
        
        # Add to comparison figure
        plt.subplot(fig_rows, fig_cols, i+1)
        plt.imshow(generated_image[:,:,0], cmap='gray')
        plt.title(f"{condition_name} (ID: {condition_id})")
        plt.axis('off')
        
        print(f"Generated image for condition {condition_id} ({condition_name}) saved")
    
    # Save the comparison figure
    plt.tight_layout()
    grid_path = os.path.join(output_dir, "all_conditions_comparison.png")
    plt.savefig(grid_path)
    plt.close()
    print(f"Comparison of all conditions saved to {grid_path}")

def find_latest_run_guid(models_dir):
    """Find the GUID of the latest training run.
    
    First checks for a "latest_run.txt" file, then tries to find the newest directory.
    
    Args:
        models_dir: Base directory containing model run folders
    
    Returns:
        GUID of latest run or None if not found
    """
    # Check for latest_run.txt file first
    latest_run_file = os.path.join(models_dir, "latest_run.txt")
    if os.path.exists(latest_run_file):
        try:
            with open(latest_run_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("Latest run GUID:"):
                        guid = line.split(":", 1)[1].strip()
                        if os.path.isdir(os.path.join(models_dir, guid)):
                            print(f"Found latest run GUID from file: {guid}")
                            return guid
        except Exception as e:
            print(f"Error reading latest run file: {e}")
    
    # If no valid GUID from file, find most recently modified directory
    try:
        subdirs = [d for d in os.listdir(models_dir) 
                  if os.path.isdir(os.path.join(models_dir, d)) and len(d) > 8]  # Filter likely GUIDs
        
        if not subdirs:
            return None
            
        # Sort by modification time (newest first)
        subdirs.sort(key=lambda d: os.path.getmtime(os.path.join(models_dir, d)), reverse=True)
        latest_guid = subdirs[0]
        print(f"Found latest run GUID by timestamp: {latest_guid}")
        return latest_guid
    
    except Exception as e:
        print(f"Error finding latest run: {e}")
        return None

def main():
    """Main function for image generation."""
    args = parse_args()
    
    # Resolve the model path from arguments
    model_path = args.model
    model_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else args.models_dir
    
    # If model is 'latest' or a run GUID is specified, resolve the actual path
    if args.model == 'latest' or args.run_guid:
        guid = args.run_guid if args.run_guid else find_latest_run_guid(args.models_dir)
        
        if not guid:
            print("No valid run GUID found. Please specify a model path or run GUID.")
            return
        
        # Use the specified model file within the run folder
        model_dir = os.path.join(args.models_dir, guid)
        model_path = os.path.join(model_dir, args.model_file)
        
        print(f"Using model from run {guid}: {model_path}")
    
    # Check if the model exists, otherwise create placeholder SVGs
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        print("Creating placeholder SVG images instead...")
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create different placeholder shapes
        shapes = ["circle", "square", "triangle"]
        for i in range(min(args.count, len(shapes))):
            svg_path = os.path.join(args.output_dir, f"placeholder_{i+1}.svg")
            create_placeholder_svg(svg_path, (args.image_size, args.image_size), shapes[i])
        
        return
    
    # Determine the mapping file path if not provided
    mapping_file = args.mapping_file
    if not mapping_file:
        # Look for mapping file in the same directory as the model
        mapping_file = os.path.join(model_dir, "condition_mapping.json")
        print(f"Looking for condition mapping at: {mapping_file}")
    
    # Load the condition mapping
    condition_mapping = load_condition_mapping(mapping_file)
    
    # Initialize the model
    generator = ImageGenerator(
        image_size=(args.image_size, args.image_size),
        latent_dim=args.latent_dim,
        num_conditions=len(condition_mapping) if condition_mapping else 1
    )
    
    # Load the trained model
    print(f"Loading model from {model_path}")
    generator.load_model(model_path)
    
    # Make a subfolder for this run in the output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(model_path).replace(".pt", "")
    output_subdir = os.path.join(args.output_dir, f"{timestamp}_{model_name}")
    os.makedirs(output_subdir, exist_ok=True)
    
    # Determine which condition to use
    condition_id = 0  # Default condition
    
    if args.interactive:
        # Let user select condition interactively
        condition_id = select_condition_interactive(condition_mapping)
    elif args.condition is not None:
        # Use the specified condition
        condition_id = resolve_condition(args.condition, condition_mapping)
    
    # Generate images based on the selected mode
    if args.mode == 'random':
        print(f"Generating {args.count} random images with condition ID {condition_id}...")
        generate_random_images(generator, args.count, output_subdir, condition_id, condition_mapping)
    elif args.mode == 'interpolate':
        print(f"Generating {args.steps} interpolated images with condition ID {condition_id}...")
        generate_interpolated_images(generator, args.steps, output_subdir, condition_id, condition_mapping)
    elif args.mode == 'all-conditions':
        print(f"Generating images for all available conditions...")
        generate_all_conditions(generator, output_subdir, condition_mapping)
    
    print(f"Image generation completed successfully. Results saved to {output_subdir}")

if __name__ == "__main__":
    main()