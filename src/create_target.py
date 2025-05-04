"""
Target SVG Creation Script

This script creates a placeholder target SVG image for training the image generator.
It's useful for testing the ML pipeline before providing your own target SVG image.
"""

import os
import argparse
from utils import create_placeholder_svg

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create a placeholder target SVG image')
    parser.add_argument('--output', type=str, default='data/target.svg',
                        help='Path to save the target SVG file')
    parser.add_argument('--size', type=int, default=128,
                        help='Size of the square image')
    parser.add_argument('--shape', type=str, choices=['circle', 'square', 'triangle'], default='circle',
                        help='Shape to include in the SVG')
    
    return parser.parse_args()

def main():
    """Main function to create the target SVG."""
    args = parse_args()
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Create the target SVG
    create_placeholder_svg(args.output, size=(args.size, args.size), shape_type=args.shape)
    
    print(f"Created target SVG at {args.output}")
    print("You can use this as a placeholder until you provide your own target SVG image.")
    print("To train the model, run: python src/train.py --target", args.output)

if __name__ == "__main__":
    main()