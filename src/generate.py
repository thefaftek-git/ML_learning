"""
Image Generation Script

This script uses a trained image generator model to create new SVG wireframe images.
It can generate images from random latent vectors or interpolate between different points.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from model import ImageGenerator
from utils import save_as_svg, create_placeholder_svg

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate images using the trained model')
    parser.add_argument('--model', type=str, default='models/generator_final.pt',
                        help='Path to the trained model file')
    parser.add_argument('--count', type=int, default=5,
                        help='Number of images to generate')
    parser.add_argument('--output-dir', type=str, default='data/generated',
                        help='Directory to save generated images')
    parser.add_argument('--image-size', type=int, default=128,
                        help='Size of the square output image')
    parser.add_argument('--latent-dim', type=int, default=100,
                        help='Dimension of the latent space')
    parser.add_argument('--mode', type=str, choices=['random', 'interpolate'], default='random',
                        help='Generation mode: random samples or interpolation between points')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of interpolation steps (only for interpolate mode)')
    
    return parser.parse_args()

def generate_random_images(generator, count, output_dir):
    """Generate a specified number of random images."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(count):
        # Generate random latent vector
        latent_vector = np.random.normal(0, 1, (1, generator.latent_dim))
        
        # Generate the image
        generated_image = generator.generate_image(latent_vector)
        
        # Save as PNG
        output_path = os.path.join(output_dir, f"random_{i+1}.png")
        plt.imsave(output_path, generated_image[:,:,0], cmap='gray')
        
        # Save as SVG
        svg_path = os.path.join(output_dir, f"random_{i+1}.svg")
        save_as_svg(generated_image, svg_path)
        
        print(f"Generated image saved to {output_path} and {svg_path}")

def generate_interpolated_images(generator, steps, output_dir):
    """Generate a sequence of images by interpolating between two random points."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate two random latent vectors
    start_vector = np.random.normal(0, 1, (1, generator.latent_dim))
    end_vector = np.random.normal(0, 1, (1, generator.latent_dim))
    
    # Create interpolation steps
    alphas = np.linspace(0, 1, steps)
    
    # Generate images for each interpolation step
    for i, alpha in enumerate(alphas):
        # Interpolate between the two vectors
        latent_vector = start_vector * (1 - alpha) + end_vector * alpha
        
        # Generate the image
        generated_image = generator.generate_image(latent_vector)
        
        # Save as PNG
        output_path = os.path.join(output_dir, f"interpolate_{i+1:02d}.png")
        plt.imsave(output_path, generated_image[:,:,0], cmap='gray')
        
        # Save as SVG
        svg_path = os.path.join(output_dir, f"interpolate_{i+1:02d}.svg")
        save_as_svg(generated_image, svg_path)
        
        print(f"Generated interpolation step {i+1}/{steps} saved to {output_path} and {svg_path}")
    
    # Create a grid visualization of the interpolation
    plt.figure(figsize=(12, 4))
    for i, alpha in enumerate(alphas):
        if i >= 10:  # Only show up to 10 images in the grid
            break
        plt.subplot(2, 5, i+1)
        
        # Interpolate between the two vectors
        latent_vector = start_vector * (1 - alpha) + end_vector * alpha
        
        # Generate the image
        generated_image = generator.generate_image(latent_vector)
        
        plt.imshow(generated_image[:,:,0], cmap='gray')
        plt.title(f"Step {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'interpolation_grid.png'))
    plt.close()

def main():
    """Main function for image generation."""
    args = parse_args()
    
    # Check if the model exists, otherwise create a placeholder SVG
    if not os.path.exists(args.model):
        print(f"Model file not found at {args.model}")
        print("Creating placeholder SVG images instead...")
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create different placeholder shapes
        shapes = ["circle", "square", "triangle"]
        for i in range(min(args.count, len(shapes))):
            svg_path = os.path.join(args.output_dir, f"placeholder_{i+1}.svg")
            create_placeholder_svg(svg_path, (args.image_size, args.image_size), shapes[i])
        
        return
    
    # Initialize the model
    generator = ImageGenerator(image_size=(args.image_size, args.image_size),
                             latent_dim=args.latent_dim)
    
    # Load the trained model
    print(f"Loading model from {args.model}")
    generator.load_model(args.model)
    
    # Generate images based on the selected mode
    if args.mode == 'random':
        print(f"Generating {args.count} random images...")
        generate_random_images(generator, args.count, args.output_dir)
    else:  # interpolate
        print(f"Generating {args.steps} interpolated images...")
        generate_interpolated_images(generator, args.steps, args.output_dir)
    
    print("Image generation completed successfully.")

if __name__ == "__main__":
    main()