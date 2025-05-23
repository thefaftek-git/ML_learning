"""
Image Generator Training Script

This script trains a machine learning model to generate images that match a target image.
The model learns from scratch without any pre-existing training data.
"""

import os
import argparse
import numpy as np
import torch
import threading
import heapq
import time
import multiprocessing as mp
from multiprocessing import Process, Manager
from tqdm import tqdm
import uuid

# Set matplotlib to use a non-interactive backend before importing plt
# This helps prevent the "main thread is not in main loop" tkinter errors
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend which doesn't require a GUI
import matplotlib.pyplot as plt

from utils import load_image, preprocess_reference_image, save_as_svg
from model import ImageGenerator
from skimage.metrics import structural_similarity as ssim

# Global variable to track model checkpoints and their loss values
# Format: List of tuples (loss, epoch, path)
model_checkpoints = []
model_lock = threading.Lock()
MAX_MODELS_TO_KEEP = 100

# Special model filenames that should never be deleted
PROTECTED_MODELS = ["generator_best.pt", "generator_final.pt"]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train the image generator model')
    
    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_source = input_group.add_mutually_exclusive_group()
    input_source.add_argument('--reference', type=str, default=None,
                      help='Path to a single target reference image (JPG, PNG, or SVG)')
    input_source.add_argument('--reference-dir', type=str, default=None,
                      help='Path to a directory containing multiple reference images for training')
    
    # Set a default if neither option is provided
    parser.set_defaults(reference='reference.png')
    
    parser.add_argument('--image-selection', type=str, choices=['sequential', 'random'], default='random',
                      help='How to select images when training with multiple references: "sequential" cycles through them in order, "random" picks randomly each epoch')
    
    parser.add_argument('--output-dir', type=str, default='models',
                      help='Directory to save model checkpoints')
    parser.add_argument('--image-size', type=int, default=None,
                      help='Base size for the output image (defaults to reference image size). This is a shorthand for setting both width and height to the same value.')
    parser.add_argument('--width', type=int, default=None,
                      help='Width for the output image (defaults to reference image width)')
    parser.add_argument('--height', type=int, default=None,
                      help='Height for the output image (defaults to reference image height)')
    parser.add_argument('--preserve-aspect', action='store_true',
                      help='Preserve the aspect ratio of the reference image')
    parser.add_argument('--latent-dim', type=int, default=100,
                      help='Dimension of the latent space')
    parser.add_argument('--epochs', type=int, default=500,
                      help='Maximum number of training epochs')
    parser.add_argument('--target-accuracy', type=float, default=99.99,
                      help='Target accuracy threshold (as a percentage, e.g. 99.99) to stop training early. Calculated as 100 * (1 - loss)')
    parser.add_argument('--target-loss', type=float, default=None,
                      help='Target loss value to stop training early (alternative to target-accuracy)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--save-interval', type=int, default=100,
                      help='DEPRECATED: Use visualization-interval instead (kept for backward compatibility)')
    parser.add_argument('--preview-dir', type=str, default='data/progress',
                      help='Directory to save progress preview images')
    # Performance optimization options
    parser.add_argument('--visualization-interval', type=int, default=100,
                      help='Interval for visualizations and model saving')
    parser.add_argument('--no-async-visualization', action='store_true',
                      help='Disable asynchronous visualization generation (enabled by default)')
    parser.add_argument('--num-workers', type=int, default=None,
                      help='Number of worker threads for PyTorch (default: auto-detected based on CPU cores)')
    parser.add_argument('--use-gpu', action='store_true',
                      help='Use GPU acceleration if available')
    parser.add_argument('--mixed-precision', action='store_true',
                      help='Use mixed precision training (speeds up GPU training)')
    parser.add_argument('--no-optimize-memory', action='store_true',
                      help='Disable memory usage optimizations (enabled by default)')
    # Parallel training options
    parser.add_argument('--parallel', action='store_true',
                      help='Enable parallel training with multiple processes. Note: May require more total iterations than sequential training to achieve equivalent results.')
    parser.add_argument('--parallel-workers', type=int, default=None,
                      help='Number of parallel training processes (default: auto-detect based on CPU cores)')
    parser.add_argument('--parallel-iterations', type=int, default=50,
                      help='Number of iterations per parallel training batch')
    # New GPU optimization option
    parser.add_argument('--steps-per-epoch', type=int, default=10,
                      help='Number of training steps per epoch (default: 10, higher values increase GPU utilization)')
    # Resume training option
    parser.add_argument('--resume-guid', type=str, default=None,
                      help='GUID of a previous training run to resume from')
    parser.add_argument('--resume-checkpoint', type=str, default='generator_best.pt',
                      help='Name of the checkpoint file to load when resuming training (defaults to generator_best.pt)')
    parser.add_argument('--full-color', action='store_true',
                      help='Use a color (RGB) model instead of grayscale')
    
    return parser.parse_args()

def collect_system_info():
    """
    Collect system information for run metadata.
    
    Returns:
        Dictionary containing system information
    """
    import platform
    import sys
    
    system_info = {
        "os": platform.platform(),
        "python_version": sys.version,
        "cpu_info": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    
    # Add GPU information if available
    if torch.cuda.is_available():
        system_info["gpu_available"] = True
        system_info["gpu_name"] = torch.cuda.get_device_name(0)
        system_info["gpu_count"] = torch.cuda.device_count()
        system_info["cuda_version"] = torch.version.cuda
    else:
        system_info["gpu_available"] = False
    
    # Get memory information if psutil is available
    try:
        import psutil
        memory = psutil.virtual_memory()
        system_info["total_memory_gb"] = round(memory.total / (1024**3), 2)
        system_info["available_memory_gb"] = round(memory.available / (1024**3), 2)
    except ImportError:
        pass
    
    return system_info

def create_run_summary(run_guid, run_metadata, run_stats, image_info, system_info, output_path):
    """
    Create a comprehensive human-readable summary of the training run.
    
    Args:
        run_guid: GUID of the training run
        run_metadata: Dictionary containing run metadata
        run_stats: Dictionary containing training statistics
        image_info: List of image info dictionaries
        system_info: Dictionary containing system information
        output_path: Path to save the summary
    """
    with open(output_path, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write(f"TRAINING RUN SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Run identification
        f.write(f"Run GUID: {run_guid}\n")
        f.write(f"Timestamp: {run_metadata['timestamp']}\n")
        f.write("\n")
        
        # System information
        f.write("-" * 80 + "\n")
        f.write("SYSTEM INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"OS: {system_info.get('os', 'Unknown')}\n")
        f.write(f"Python: {system_info.get('python_version', 'Unknown').split()[0]}\n")
        f.write(f"CPU: {system_info.get('cpu_info', 'Unknown')}\n")
        f.write(f"CPU Cores: {system_info.get('cpu_count', 'Unknown')}\n")
        
        if system_info.get('gpu_available', False):
            f.write(f"GPU: {system_info.get('gpu_name', 'Unknown')}\n")
            f.write(f"CUDA Version: {system_info.get('cuda_version', 'Unknown')}\n")
        else:
            f.write("GPU: Not used\n")
            
        if 'total_memory_gb' in system_info:
            f.write(f"System Memory: {system_info['total_memory_gb']} GB total, "
                    f"{system_info['available_memory_gb']} GB available at run start\n")
        f.write("\n")
        
        # Command arguments
        f.write("-" * 80 + "\n")
        f.write("TRAINING PARAMETERS\n")
        f.write("-" * 80 + "\n")
        for arg, value in run_metadata['arguments'].items():
            if arg not in ['output_dir', 'preview_dir']:  # Skip less important params
                f.write(f"{arg}: {value}\n")
        f.write("\n")
        
        # Image dataset information
        f.write("-" * 80 + "\n")
        f.write("IMAGE DATASET\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of images: {len(image_info)}\n")
        if len(image_info) > 0:
            dims = image_info[0].get('original_dimensions', ('Unknown', 'Unknown'))
            f.write(f"Original dimensions (first image): {dims[0]}x{dims[1]}\n")
            
            # Add resizing information if applicable
            if 'image_size' in run_metadata['arguments'] and run_metadata['arguments']['image_size'] is not None:
                f.write(f"Resized to: {run_metadata['arguments']['image_size']}x{run_metadata['arguments']['image_size']}\n")
            elif 'width' in run_metadata['arguments'] and 'height' in run_metadata['arguments']:
                if run_metadata['arguments']['width'] is not None and run_metadata['arguments']['height'] is not None:
                    f.write(f"Resized to: {run_metadata['arguments']['width']}x{run_metadata['arguments']['height']}\n")
            
            f.write("\nImage mapping:\n")
            for info in image_info:
                f.write(f"  ID {info['condition_id']}: {info['filename']}\n")
        f.write("\n")
        
        # Model information
        f.write("-" * 80 + "\n")
        f.write("MODEL INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Latent dimension: {run_metadata['arguments'].get('latent_dim', 'Unknown')}\n")
        f.write(f"Batch size: {run_metadata['arguments'].get('batch_size', 'Unknown')}\n")
        f.write(f"Steps per epoch: {run_metadata['arguments'].get('steps_per_epoch', 'Unknown')}\n")
        f.write("\n")
        
        # Training statistics
        f.write("-" * 80 + "\n")
        f.write("TRAINING STATISTICS\n")
        f.write("-" * 80 + "\n")
        
        # Training time
        total_time = run_stats.get('total_training_time', 0)
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        f.write(f"Total training time: {hours}h {minutes}m {seconds}s\n")
        f.write(f"Average time per epoch: {run_stats.get('avg_epoch_time', 0):.3f} seconds\n")
        
        # Loss information
        f.write(f"Final loss: {run_stats['epoch_losses'][-1] if run_stats['epoch_losses'] else 'Unknown'}\n")
        if run_stats.get('best_loss') is not None:
            f.write(f"Best loss: {run_stats['best_loss']} (epoch {run_stats['best_epoch']})\n")
        
        # Calculate loss change rates
        if len(run_stats['epoch_losses']) > 5:
            # Calculate early, mid, and late loss changes
            early_change = run_stats['epoch_losses'][10] - run_stats['epoch_losses'][0] if len(run_stats['epoch_losses']) > 10 else None
            mid_idx = len(run_stats['epoch_losses']) // 2
            mid_change = run_stats['epoch_losses'][mid_idx] - run_stats['epoch_losses'][0] if mid_idx > 0 else None
            late_change = run_stats['epoch_losses'][-1] - run_stats['epoch_losses'][-6] if len(run_stats['epoch_losses']) > 5 else None
            
            if early_change is not None:
                f.write(f"Early loss change rate (first 10 epochs): {early_change/10:.6f} per epoch\n")
            if mid_change is not None:
                f.write(f"Mid-training loss change rate: {mid_change/mid_idx:.6f} per epoch\n")
            if late_change is not None:
                f.write(f"Late loss change rate (last 5 epochs): {late_change/5:.6f} per epoch\n")
                
            # Check for convergence
            if abs(late_change/5) < 0.0001:
                f.write("Status: Training appears to have converged (minimal loss change)\n")
            else:
                f.write("Status: Training was likely still improving when stopped\n")
        f.write("\n")
        
        # Saved files
        f.write("-" * 80 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("-" * 80 + "\n")
        run_dir = os.path.dirname(output_path)
        f.write(f"Model checkpoints and visualizations saved in:\n")
        f.write(f"- Models: {run_dir}\n")
        preview_dir = os.path.join(run_metadata['arguments']['preview_dir'], run_guid)
        f.write(f"- Visualizations: {preview_dir}\n")
        f.write("\n")
        
        # Recommendations
        f.write("-" * 80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        
        # Provide recommendations based on training outcomes
        if run_stats.get('best_epoch') is not None and run_stats['best_epoch'] < len(run_stats['epoch_losses']) * 0.8:
            f.write("- Best model occurred early in training. Consider reducing learning rate or regularization.\n")
        
        if len(run_stats['epoch_losses']) > 5:
            if run_stats['epoch_losses'][-1] > 0.1:
                f.write("- Final loss still high. Consider training for more epochs.\n")
            
            if abs(late_change/5) > 0.001:
                f.write("- Loss was still decreasing. Consider training for more epochs.\n")
                
        if len(image_info) == 1:
            f.write("- Training with more reference images might improve versatility of the model.\n")
            
        f.write("\n")
        
        # Footer
        f.write("=" * 80 + "\n")
        f.write(f"End of summary for run {run_guid}\n")
        f.write("=" * 80 + "\n")
    
    print(f"Comprehensive run summary saved to {output_path}")

def save_model_architecture_info(generator, output_path):
    """
    Save model architecture information to a file.
    
    Args:
        generator: The model instance
        output_path: Path to save the architecture information
    """
    try:
        with open(output_path, 'w') as f:
            f.write(f"Model Architecture:\n")
            f.write("=" * 80 + "\n")
            
            # Get generator architecture if available
            if hasattr(generator, 'get_architecture_info'):
                arch_info = generator.get_architecture_info()
                for key, value in arch_info.items():
                    f.write(f"{key}: {value}\n")
            else:
                # Model class doesn't have architecture info method
                # Try to get basic model attributes
                f.write(f"Input image size: {generator.image_size}\n")
                f.write(f"Latent dimension: {generator.latent_dim}\n")
                f.write(f"Device: {generator.device}\n")
                if hasattr(generator, 'model') and hasattr(generator.model, '__str__'):
                    f.write("\nModel structure:\n")
                    f.write(str(generator.model))
                    
            f.write("\n")
            
            # Add parameter count
            total_params = sum(p.numel() for p in generator.parameters())
            trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n")
            
        print(f"Model architecture information saved to {output_path}")
    except Exception as e:
        print(f"Error saving model architecture information: {e}")

def save_per_image_loss_plot(image_losses, image_info, output_dir):
    """
    Save a plot showing training loss for each reference image separately.
    
    Args:
        image_losses: Dictionary mapping condition_id to list of loss values
        image_info: List of image info dictionaries
        output_dir: Directory to save the plot
    """
    try:
        # Create a mapping from condition_id to readable name
        id_to_name = {info['condition_id']: f"{info['filename']} (ID {info['condition_id']})" 
                     for info in image_info}
        
        # Create the plot
        fig = plt.figure(figsize=(12, 6))
        
        # Plot each image's loss curve
        for condition_id, losses in image_losses.items():
            if len(losses) > 0:  # Only plot if we have data
                steps = list(range(1, len(losses) + 1))
                label = id_to_name.get(condition_id, f"Image ID {condition_id}")
                plt.plot(steps, losses, label=label, linewidth=2)
        
        plt.title('Training Loss Per Reference Image')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend(loc='upper right')
        
        # Add a horizontal line for the best overall loss if available
        all_min = float('inf')
        for losses in image_losses.values():
            if losses and min(losses) < all_min:
                all_min = min(losses)
        
        if all_min < float('inf'):
            plt.axhline(y=all_min, color='r', linestyle='--', 
                       label=f'Best Loss: {all_min:.6f}')
            plt.legend(loc='upper right')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_image_loss_curve.png'))
        plt.close(fig)
        
        # Also save individual plots for each image in their respective directories
        for condition_id, losses in image_losses.items():
            if len(losses) > 0:
                # Create a specific directory for this reference image's data
                img_specific_dir = os.path.join(output_dir, f"ref_{condition_id}")
                os.makedirs(img_specific_dir, exist_ok=True)
                
                fig = plt.figure(figsize=(8, 5))
                steps = list(range(1, len(losses) + 1))
                plt.plot(steps, losses, linewidth=2, color='blue')
                
                name = id_to_name.get(condition_id, f"Image ID {condition_id}")
                plt.title(f'Training Loss For {name}')
                plt.xlabel('Training Steps')
                plt.ylabel('Loss')
                plt.grid(True)
                
                # Add min/max/average indicators
                if losses:
                    min_loss = min(losses)
                    min_step = losses.index(min_loss) + 1
                    plt.axhline(y=min_loss, color='g', linestyle='--', 
                               label=f'Best: {min_loss:.6f} (Step {min_step})')
                    plt.axhline(y=sum(losses)/len(losses), color='r', linestyle=':',
                               label=f'Avg: {sum(losses)/len(losses):.6f}')
                    plt.legend(loc='upper right')
                
                plt.tight_layout()
                plt.savefig(os.path.join(img_specific_dir, 'loss_curve.png'))
                plt.close(fig)
                
    except Exception as e:
        print(f"Error saving per-image loss plot: {e}")
        plt.close('all')  # Close all figures in case of error

def load_reference_images(args, preview_dir):
    """
    Load reference images based on command-line arguments.
    
    This function handles both single image and directory-based image loading.
    It also checks for and loads associated annotation files.
    
    Args:
        args: Command-line arguments
        preview_dir: Directory to save preview images
        
    Returns:
        A tuple containing:
        - List of preprocessed images
        - List of image info dictionaries (each containing filename, dimensions, condition_id, etc.)
        - Common image dimensions to use (height, width)
        - Number of conditions (number of unique images)
        - Dictionary mapping tag names to indices (or None if no annotations)
        - Total number of unique tags across all annotations
    """
    images = []
    image_info = []
    all_annotations = []
    grayscale = not args.full_color
    
    # Determine if we're loading from a directory or a single file
    if args.reference_dir:
        print(f"Loading images from directory: {args.reference_dir}")
        reference_dir = os.path.join(os.getcwd(), args.reference_dir)
        
        # Check if directory exists
        if not os.path.isdir(reference_dir):
            raise ValueError(f"Reference directory '{reference_dir}' does not exist")
        
        # Get list of supported image files in the directory
        supported_extensions = ['.png', '.jpg', '.jpeg', '.svg', '.bmp', '.gif']
        image_files = []
        
        for filename in os.listdir(reference_dir):
            if any(filename.lower().endswith(ext) for ext in supported_extensions):
                image_files.append(os.path.join(reference_dir, filename))
        
        if not image_files:
            raise ValueError(f"No supported image files found in '{reference_dir}'")
            
        print(f"Found {len(image_files)} images")
        
        # Get dimensions from the first image to use as reference
        first_image_path = image_files[0]
        _, original_dimensions, _ = preprocess_reference_image(
            first_image_path, preview_dir, size=None, show_preview=False, grayscale=grayscale
        )
        original_height, original_width = original_dimensions
        
        # Calculate target dimensions
        target_dimensions = calculate_target_dimensions(
            args, original_height, original_width, preview_dir
        )
        height, width = target_dimensions
        print(f"Using common dimensions for all images: {height}x{width} (height x width)")
        
        # Create filename-to-condition_id mapping
        # Sort filenames for consistent condition IDs across runs
        sorted_filenames = sorted([os.path.basename(f) for f in image_files])
        filename_to_condition = {name: i for i, name in enumerate(sorted_filenames)}
        
        print("Condition ID mapping:")
        for filename, condition_id in filename_to_condition.items():
            print(f"  {condition_id}: {filename}")
        
        # Process all images
        for img_path in image_files:
            filename = os.path.basename(img_path)
            condition_id = filename_to_condition[filename]
            print(f"Processing: {filename} (Condition ID: {condition_id})")
            
            # Process the image using the target dimensions
            processed_img, _, annotations = preprocess_reference_image(
                img_path, preview_dir, size=target_dimensions, show_preview=False, grayscale=grayscale
            )
            
            # Store annotations if found
            if annotations:
                print(f"Found annotations for {filename}")
                all_annotations.append(annotations)
            else:
                all_annotations.append(None)
            
            # Ensure all images have the same dimensions
            if processed_img.shape[0] != height or processed_img.shape[1] != width:
                print(f"Warning: Image {filename} has different dimensions. Resizing to match others.")
                processed_img = resize_image(processed_img, (height, width))
            
            images.append(processed_img)
            image_info.append({
                'filename': filename,
                'path': img_path,
                'original_dimensions': original_dimensions,
                'condition_id': condition_id,  # Store condition ID with image info
                'has_annotations': annotations is not None  # Track if this image has annotations
            })
            
        # Use a special preview image that includes all reference images
        if len(images) > 1:
            create_reference_grid(images, image_info, preview_dir)
            
        # Number of conditions is equal to number of unique images
        num_conditions = len(images)
            
    else:
        # Single image mode
        reference_path = os.path.join(os.getcwd(), args.reference)
        print(f"Processing single reference image: {args.reference}")
        
        # Get target dimensions
        if args.image_size is None and args.width is None and args.height is None:
            # Use original dimensions from the reference image
            processed_img, original_dimensions, annotations = preprocess_reference_image(
                reference_path, preview_dir, size=None, grayscale=grayscale
            )
            height, width = original_dimensions
            print(f"Using reference image's original dimensions: {height}x{width}")
        else:
            # Calculate target dimensions based on arguments
            _, original_dimensions, _ = preprocess_reference_image(
                reference_path, preview_dir, size=None, show_preview=False, load_full=False, grayscale=grayscale
            )
            original_height, original_width = original_dimensions
            
            # Calculate target dimensions
            target_dimensions = calculate_target_dimensions(
                args, original_height, original_width, preview_dir
            )
            height, width = target_dimensions
            
            print(f"Processing reference image to dimensions: {height}x{width}")
            # Process the image using the target dimensions
            processed_img, _, annotations = preprocess_reference_image(
                reference_path, preview_dir, size=target_dimensions, grayscale=grayscale
            )
        
        # Store annotations if found
        if annotations:
            print(f"Found annotations for {os.path.basename(reference_path)}")
            all_annotations.append(annotations)
        else:
            all_annotations.append(None)
        
        # Add the single image to our lists with condition_id = 0
        images.append(processed_img)
        image_info.append({
            'filename': os.path.basename(reference_path),
            'path': reference_path,
            'original_dimensions': original_dimensions,
            'condition_id': 0,  # Single image always uses condition ID 0
            'has_annotations': annotations is not None  # Track if this image has annotations
        })
        
        # Only one condition for a single image
        num_conditions = 1
    
    # Create tag mapping if any annotations were found
    tag_mapping = None
    total_tags = 0
    
    if any(anno is not None for anno in all_annotations):
        print("Creating tag mapping from annotations...")
        tag_mapping = create_tag_mapping(all_annotations)
        total_tags = len(tag_mapping)
        
        # Save tag mapping for future reference
        tag_mapping_path = os.path.join(os.path.dirname(preview_dir), "tag_mapping.json")
        with open(tag_mapping_path, 'w') as f:
            json.dump({
                "tag_to_index": tag_mapping,
                "total_tags": total_tags
            }, f, indent=2)
        print(f"Saved tag mapping with {total_tags} tags to {tag_mapping_path}")
        
        # Update image_info with annotation details
        for i, (info, anno) in enumerate(zip(image_info, all_annotations)):
            if anno is not None:
                # Count regions and tags
                region_count = len(anno.get('regions', []))
                tag_count = len(anno.get('tags', []))
                region_tag_count = sum(len(region.get('tags', [])) for region in anno.get('regions', []))
                
                # Store counts in image_info
                image_info[i].update({
                    'region_count': region_count,
                    'global_tag_count': tag_count,
                    'region_tag_count': region_tag_count
                })
    
    return images, image_info, (height, width), num_conditions, tag_mapping, total_tags

def calculate_target_dimensions(args, original_height, original_width, preview_dir):
    """
    Calculate the target dimensions based on command-line arguments and original image dimensions.
    
    Args:
        args: Command-line arguments
        original_height: Original image height
        original_width: Original image width
        preview_dir: Directory where previews are saved
        
    Returns:
        Tuple of (height, width) dimensions to use
    """
    original_aspect_ratio = original_width / original_height
    
    # Use the user-specified size while maintaining aspect ratio
    if args.image_size is not None:
        # If image_size is provided, use square dimensions or maintain aspect ratio
        if args.preserve_aspect:
            # Maintain aspect ratio based on the reference image
            if original_width >= original_height:
                # Width-dominant image
                width = args.image_size
                height = int(width / original_aspect_ratio)
            else:
                # Height-dominant image
                height = args.image_size
                width = int(height * original_aspect_ratio)
        else:
            # Square dimensions
            width = height = args.image_size
    else:
        # Handle separate width and height parameters
        if args.width is not None and args.height is not None:
            # Both width and height specified
            width = args.width
            height = args.height
            
            # Check if the aspect ratio matches
            specified_aspect_ratio = width / height
            aspect_ratio_difference = abs(specified_aspect_ratio - original_aspect_ratio) / original_aspect_ratio
            
            if aspect_ratio_difference > 0.01 and not args.preserve_aspect:  # 1% tolerance
                print("\nWARNING: The specified dimensions do not match the aspect ratio of the reference image.")
                print(f"Original aspect ratio: {original_aspect_ratio:.3f}, Specified: {specified_aspect_ratio:.3f}")
                print("This may cause distortion in the generated images.")
                print("Use --preserve-aspect to automatically maintain the original aspect ratio.")
                print("Continuing with the specified dimensions...\n")
            elif args.preserve_aspect:
                # Adjust to preserve aspect ratio based on the larger dimension
                if width >= height:
                    # Width is the primary dimension
                    height = int(width / original_aspect_ratio)
                    print(f"Adjusting height to {height} to maintain aspect ratio")
                else:
                    # Height is the primary dimension
                    width = int(height * original_aspect_ratio)
                    print(f"Adjusting width to {width} to maintain aspect ratio")
        elif args.width is not None:
            # Only width specified, calculate height to maintain aspect ratio
            width = args.width
            height = int(width / original_aspect_ratio)
            print(f"Using calculated height of {height} to maintain aspect ratio")
        elif args.height is not None:
            # Only height specified, calculate width to maintain aspect ratio
            height = args.height
            width = int(height * original_aspect_ratio)
            print(f"Using calculated width of {width} to maintain aspect ratio")
        else:
            # No dimensions specified, use original dimensions
            height, width = original_height, original_width
    
    print(f"Using dimensions: {height}x{width} (height x width)")
    return height, width

def resize_image(image, target_size):
    """
    Resize an image to target dimensions.
    
    Args:
        image: Numpy array representing the image
        target_size: Tuple of (height, width)
        
    Returns:
        Resized image as numpy array
    """
    from skimage.transform import resize
    
    height, width = target_size
    # If image already has the target size, return it unchanged
    if image.shape[0] == height and image.shape[1] == width:
        return image
        
    # Resize the image, preserving the number of channels
    if len(image.shape) == 3:
        # Multichannel image
        resized = resize(image, (height, width, image.shape[2]), 
                         anti_aliasing=True, preserve_range=True)
    else:
        # Grayscale image
        resized = resize(image, (height, width), 
                         anti_aliasing=True, preserve_range=True)
        # Add channel dimension if it was present in the original
        if len(image.shape) == 3:
            resized = resized[..., np.newaxis]
            
    return resized.astype(image.dtype)

def create_reference_grid(images, image_info, output_dir):
    """
    Create a grid visualization of all reference images.
    
    Args:
        images: List of processed images
        image_info: List of image info dictionaries
        output_dir: Directory to save the visualization
    """
    import math
    from matplotlib.gridspec import GridSpec
    
    # Determine grid size
    n_images = len(images)
    grid_size = math.ceil(math.sqrt(n_images))
    rows = grid_size
    cols = grid_size
    
    # Create a figure to show all reference images
    plt.figure(figsize=(cols * 3, rows * 3))
    gs = GridSpec(rows, cols)
    
    for i, (img, info) in enumerate(zip(images, image_info)):
        ax = plt.subplot(gs[i // cols, i % cols])
        if img.shape[2] == 1:
            ax.imshow(img[:, :, 0], cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title(f"{info['filename']}")
        ax.axis('off')
    
    plt.tight_layout()
    grid_path = os.path.join(output_dir, 'reference_grid.png')
    plt.savefig(grid_path)
    plt.close()
    
    print(f"Created reference grid visualization: {grid_path}")

def select_target_image(images, image_info, all_annotations, epoch, selection_method='random'):
    """
    Select a target image for the current training epoch.
    
    Args:
        images: List of processed images
        image_info: List of image info dictionaries
        all_annotations: List of annotation dictionaries corresponding to images
        epoch: Current training epoch
        selection_method: How to select the image ('random' or 'sequential')
        
    Returns:
        Tuple of (selected image, image information, annotations)
    """
    if len(images) == 1:
        # If only one image, always return it
        return images[0], image_info[0], all_annotations[0] if all_annotations else None
    
    if selection_method == 'random':
        # Randomly select an image
        idx = np.random.randint(0, len(images))
        return images[idx], image_info[idx], all_annotations[idx] if all_annotations else None
    else:
        # Sequential selection - cycle through images based on epoch
        idx = epoch % len(images)
        return images[idx], image_info[idx], all_annotations[idx] if all_annotations else None

def visualize_progress(generator, epoch, latent_vector, target_image, output_dir, prefix="progress", save_comparison=False, image_info=None):
    """Save a visualization of training progress."""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Generate image with the current model
        with torch.amp.autocast('cuda', enabled=generator.mixed_precision):
            generated_image = generator.generate_image(latent_vector)
              # Apply histogram matching to match brightness/contrast of reference image
        from normalize_image import match_image_brightness_contrast
        matched_generated_image = match_image_brightness_contrast(generated_image, target_image)
        
        # Create a figure with the target and generated images side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot target image - ensure proper rendering for grayscale or color images
        if target_image.shape[2] == 1:  # Grayscale image
            ax1.imshow(target_image[:, :, 0], cmap='gray', vmin=0, vmax=1)
        else:  # RGB image
            ax1.imshow(target_image)
        
        # Create a more informative title using image_info if available
        target_title = "Target Image"
        if image_info:
            target_title = f"Target: {image_info['filename']} (ID {image_info['condition_id']})"
            
        ax1.set_title(target_title)
        ax1.axis('off')
        
        # Plot generated image - ensure proper rendering for grayscale or color images
        if matched_generated_image.shape[2] == 1:  # Grayscale image
            ax2.imshow(matched_generated_image[:, :, 0], cmap='gray', vmin=0, vmax=1)
        else:  # RGB image
            ax2.imshow(matched_generated_image)
        ax2.set_title(f"Generated (Epoch {epoch})")
        ax2.axis('off')
        
        # Save the visualization
        plt.tight_layout()
        viz_path = os.path.join(output_dir, f"{prefix}_{epoch:04d}.png")
        plt.savefig(viz_path, dpi=150)
          # If this is a final comparison, also save it to the main data folder
        if save_comparison:
            comparison_path = os.path.join(os.path.dirname(output_dir), "final_comparison.png")
            plt.savefig(comparison_path, dpi=150)
            print(f"Final comparison saved to {comparison_path}")
        
        # Explicitly close the figure to free memory and avoid tkinter issues
        plt.close(fig)
    
        # Save the generated image as SVG
        svg_path = os.path.join(output_dir, f"{prefix}_{epoch:04d}.svg")
        save_as_svg(matched_generated_image, svg_path)
        
        # Also save individual images of reference and generated for easier comparison
        # Use a more informative filename if image_info is available
        ref_filename = "reference.png"
        if image_info:
            ref_filename = f"reference_{image_info['filename']}_{image_info['condition_id']}.png"
            
        ref_path = os.path.join(output_dir, ref_filename)
        if not os.path.exists(ref_path):  # Only save reference once
            plt.figure(figsize=(5, 5))
            if target_image.shape[2] == 1:
                plt.imshow(target_image[:, :, 0], cmap='gray', vmin=0, vmax=1)
            else:
                plt.imshow(target_image)
                
            # Add title with image filename if available
            if image_info:
                plt.title(f"{image_info['filename']} (ID {image_info['condition_id']})")
                
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(ref_path, dpi=150)
            plt.close()
        
        gen_path = os.path.join(output_dir, f"generated_{epoch:04d}.png")
        plt.figure(figsize=(5, 5))
        if matched_generated_image.shape[2] == 1:
            plt.imshow(matched_generated_image[:, :, 0], cmap='gray', vmin=0, vmax=1)
        else:
            plt.imshow(matched_generated_image)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(gen_path, dpi=150)
        plt.close()
        
        # Return the matched image for further use
        return matched_generated_image
    
    except Exception as e:
        print(f"Error during visualization: {e}")
        # Make sure we clean up even if there's an error
        plt.close('all')
        return None

def visualize_progress_async(generator, epoch, latent_vector, target_image, output_dir, prefix="progress", save_comparison=False, image_info=None):
    """
    Create visualizations in a background thread to avoid interrupting training.
    Launches the visualization_worker function in a separate thread.
    
    Args:
        Same as visualize_progress
    
    Returns:
        None (visualization happens asynchronously)
    """
    # Create a thread to do the visualization work
    thread = threading.Thread(
        target=visualize_progress,
        args=(generator, epoch, latent_vector, target_image, output_dir, prefix, save_comparison, image_info),
        daemon=True  # Allow the program to exit even if this thread is running
    )
    thread.start()
    return None  # We can't return the generated image as it's created asynchronously

def save_loss_plot(losses, output_dir):
    """Save a plot of the training loss over epochs."""
    try:
        fig = plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
        # Explicitly close the figure to free memory and avoid tkinter issues
        plt.close(fig)
    except Exception as e:
        print(f"Error saving loss plot: {e}")
        plt.close('all')  # Close all figures in case of error

def register_model_checkpoint(model_path, epoch, loss):
    """
    Register a new model checkpoint and its loss value.
    This function adds the model to our tracking list and cleans up old models if needed.
    
    Args:
        model_path: Path to the saved model checkpoint
        epoch: Training epoch number
        loss: Loss value for this model checkpoint
    """
    global model_checkpoints
    
    # Don't track protected models in our cleanup system
    if os.path.basename(model_path) in PROTECTED_MODELS:
        return
    
    # Add the new model to our tracking list with a lock to avoid race conditions
    with model_lock:
        # Store as negative loss for max-heap behavior with heapq (we want lowest loss at top)
        heapq.heappush(model_checkpoints, (-loss, epoch, model_path))
        
        # Start a background thread to clean up old models if needed
        if len(model_checkpoints) > MAX_MODELS_TO_KEEP:
            threading.Thread(target=cleanup_models, daemon=True).start()

def cleanup_models():
    """
    Clean up older model checkpoints, keeping only the top MAX_MODELS_TO_KEEP models.
    This function runs in a background thread to avoid pausing training.
    """
    global model_checkpoints
    
    # Create a local copy of the model_checkpoints list to work with
    models_to_delete = []
    
    with model_lock:
        # Sort checkpoints by loss (best first)
        sorted_checkpoints = sorted(model_checkpoints)
        
        # Keep only the best MAX_MODELS_TO_KEEP models
        if len(sorted_checkpoints) > MAX_MODELS_TO_KEEP:
            models_to_delete = sorted_checkpoints[MAX_MODELS_TO_KEEP:]
            model_checkpoints = sorted_checkpoints[:MAX_MODELS_TO_KEEP]
    
    # Now delete the excess models outside the lock to minimize lock contention
    for _, _, model_path in models_to_delete:
        # Double-check the file exists and isn't a protected model
        if os.path.exists(model_path) and os.path.basename(model_path) not in PROTECTED_MODELS:
            try:
                os.remove(model_path)
                print(f"Removed old model checkpoint: {os.path.basename(model_path)}")
            except Exception as e:
                print(f"Error deleting model {model_path}: {e}")
                
            # Small sleep to avoid overwhelming the file system
            time.sleep(0.01)

def save_condition_mapping(image_info, output_path):
    """
    Save the mapping between condition IDs and image filenames to a JSON file.
    This information will be used by the generation script.
    
    Args:
        image_info: List of image info dictionaries
        output_path: Path to save the JSON file
    """
    import json
    
    # Create a mapping dictionary
    condition_mapping = {}
    for info in image_info:
        condition_id = info['condition_id']
        filename = info['filename']
        condition_mapping[condition_id] = {
            'filename': filename,
            'path': info['path']
        }
        
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(condition_mapping, f, indent=2)
        
    print(f"Saved condition mapping to {output_path}")

def create_final_comparison_grid(generator, latent_vector, images, image_info, output_dir):
    """
    Create a grid visualization comparing the model's output against all reference images.
    
    Args:
        generator: The trained generator model
        latent_vector: Fixed latent vector for consistent generation
        images: List of all reference images
        image_info: List of image info dictionaries
        output_dir: Directory to save the visualization
    """
    import math
    from matplotlib.gridspec import GridSpec
    
    # Determine grid size
    n_images = len(images)
    # One more column for the generated image
    cols = math.ceil(math.sqrt(n_images)) + 1
    rows = math.ceil(n_images / (cols - 1))
    
    # Create a figure for the grid
    plt.figure(figsize=(cols * 3, rows * 3))
    gs = GridSpec(rows, cols)
      # Generate an image with the current model using the fixed latent vector
    generated_image = generator.generate_image(latent_vector)
    
    # Apply histogram matching using the first reference image as a guide
    from normalize_image import match_image_brightness_contrast
    matched_generated_image = match_image_brightness_contrast(generated_image, images[0])
      # Place the generated image in the top-left corner
    ax = plt.subplot(gs[0, 0])
    if matched_generated_image.shape[2] == 1:
        ax.imshow(matched_generated_image[:, :, 0], cmap='gray')
    else:
        ax.imshow(matched_generated_image)
    ax.set_title("Generated Image")
    ax.axis('off')
    
    # Add all reference images to the grid
    for i, (img, info) in enumerate(zip(images, image_info)):
        row = i // (cols - 1)
        col = (i % (cols - 1)) + 1  # +1 to skip the first column
        
        ax = plt.subplot(gs[row, col])
        if img.shape[2] == 1:
            ax.imshow(img[:, :, 0], cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title(f"Ref #{info['condition_id']}: {info['filename']}")
        ax.axis('off')
    
    plt.tight_layout()
    grid_path = os.path.join(output_dir, 'final_comparison_grid.png')
    plt.savefig(grid_path)
    plt.close()
    
    print(f"Created final comparison grid: {grid_path}")

def calculate_image_similarity(image1, image2):
    """
    Calculate similarity between two images using both MSE-based similarity and SSIM (contrast-aware).
    Returns a dictionary with both metrics.
    """
    if image1 is None or image2 is None:
        return {"mse_similarity": 0.0, "ssim": 0.0}
    # Ensure images have the same shape
    if image1.shape != image2.shape:
        return {"mse_similarity": 0.0, "ssim": 0.0}
    # MSE-based similarity
    mse = np.mean((image1 - image2) ** 2)
    mse_similarity = np.exp(-mse * 10)
    # SSIM (contrast-aware)
    if image1.shape[-1] == 1:
        ssim_score = ssim(image1[:, :, 0], image2[:, :, 0], data_range=1.0)
    else:
        try:
            ssim_score = ssim(image1, image2, data_range=1.0, channel_axis=-1)
        except TypeError:
            ssim_score = ssim(image1, image2, data_range=1.0, multichannel=True)
    return {"mse_similarity": float(mse_similarity), "ssim": float(ssim_score)}

def inject_training_entropy(generator, latent_vector, epoch):
    """
    Inject entropy into the training process to escape local minima.
    
    Args:
        generator: The ImageGenerator model
        latent_vector: Current latent vector being used
        epoch: Current training epoch
        
    Returns:
        Modified latent vector with added entropy
    """
    print(f"\nWARNING: Training appears to have stalled at epoch {epoch}.")
    print("Injecting entropy to escape possible local minimum...")
    
    # Strategy 1: Add noise to the latent vector
    noise_scale = 0.2 + (0.1 * (epoch // 100))  # Increase noise with epochs
    noise = np.random.normal(0, noise_scale, latent_vector.shape)
    new_latent_vector = latent_vector + noise
    
    # Strategy 2: Modify model parameters if possible
    try:
        if hasattr(generator, 'inject_entropy'):
            generator.inject_entropy(epoch)
        else:
            # Manual intervention - temporarily increase learning rate
            if hasattr(generator, 'optimizer') and hasattr(generator.optimizer, 'param_groups'):
                for param_group in generator.optimizer.param_groups:
                    current_lr = param_group['lr']
                    # Double the learning rate temporarily
                    param_group['lr'] = current_lr * 2.0
                    print(f"Temporarily increased learning rate to {current_lr * 2.0}")
                    # Schedule to reset learning rate after 5 epochs
                    def reset_lr():
                        for pg in generator.optimizer.param_groups:
                            pg['lr'] = current_lr
                    threading.Timer(5.0, reset_lr).start()
    except Exception as e:
        print(f"Error while injecting entropy into optimizer: {e}")
    
    print("Entropy injection complete. Continuing training with modified parameters.")
    
    return new_latent_vector

def main():
    """Main training function."""
    args = parse_args()
    
    # Check if we're resuming training from a previous run
    if args.resume_guid:
        print(f"Resuming training from previous run GUID: {args.resume_guid}")
        run_guid = args.resume_guid
        run_output_dir = os.path.join(args.output_dir, run_guid)
        run_preview_dir = os.path.join(args.preview_dir, run_guid)
        
        # Check if directories exist
        if not os.path.exists(run_output_dir):
            raise ValueError(f"Cannot resume training: run directory {run_output_dir} not found.")
        
        # Load run metadata
        run_metadata_path = os.path.join(run_output_dir, "run_metadata.json")
        if not os.path.exists(run_metadata_path):
            raise ValueError(f"Cannot resume training: metadata file {run_metadata_path} not found.")
            
        # Load the metadata to get the original image dimensions and other parameters
        with open(run_metadata_path, 'r') as f:
            import json
            original_metadata = json.load(f)
            
        # Get the original training parameters
        original_args = original_metadata.get('arguments', {})
        
        # Check if we can find the model's image dimensions
        # First try to load a model checkpoint to extract dimensions
        model_path = os.path.join(run_output_dir, args.resume_checkpoint)
        if not os.path.exists(model_path):
            # Try to find any model checkpoint
            checkpoint_files = [f for f in os.listdir(run_output_dir) if f.startswith("generator_") and f.endswith(".pt")]
            if checkpoint_files:
                checkpoint_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]) if f[10:-3].isdigit() else 0, reverse=True)
                model_path = os.path.join(run_output_dir, checkpoint_files[0])
                print(f"Using the latest checkpoint found: {checkpoint_files[0]}")
            else:
                raise ValueError(f"Cannot resume training: no checkpoints found in {run_output_dir}")
        
        # Load the checkpoint to extract model parameters
        print(f"Loading model from {model_path} to extract parameters...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract image dimensions from the checkpoint if available
        if 'image_size' in checkpoint:
            model_height, model_width = checkpoint['image_size']
            print(f"Found image dimensions in checkpoint: {model_height}x{model_width}")
            
            # Override current dimensions with the original ones
            if args.image_size is not None or args.width is not None or args.height is not None:
                print("WARNING: Ignoring provided image dimensions and using the original model's dimensions")
                
            # Override the command line arguments to match the original dimensions
            args.image_size = None
            args.width = model_width
            args.height = model_height
            
            # Also copy the preserve_aspect setting to ensure consistent behavior
            if 'preserve_aspect' in original_args:
                args.preserve_aspect = original_args['preserve_aspect']
                
            print(f"Will use dimensions from original model: {model_height}x{model_width}")
        else:
            print("Warning: Could not find image dimensions in checkpoint")
            # Try to get dimensions from the original metadata
            if 'width' in original_args and 'height' in original_args and original_args['width'] and original_args['height']:
                print(f"Using dimensions from original metadata: {original_args['height']}x{original_args['width']}")
                args.width = original_args['width']
                args.height = original_args['height']
                args.image_size = None
            elif 'image_size' in original_args and original_args['image_size']:
                print(f"Using image_size from original metadata: {original_args['image_size']}")
                args.image_size = original_args['image_size']
                args.width = None
                args.height = None
            else:
                print("Warning: Could not determine original dimensions. Using provided dimensions.")
        
        # Copy other important parameters from the original run
        if 'latent_dim' in original_args and not args.latent_dim == original_args['latent_dim']:
            print(f"Using latent dimension from original model: {original_args['latent_dim']}")
            args.latent_dim = original_args['latent_dim']
        
        # Load run stats to resume from
        run_stats_path = os.path.join(run_output_dir, "run_stats.json")
        if os.path.exists(run_stats_path):
            with open(run_stats_path, 'r') as f:
                import json
                run_stats = json.load(f)
            
            # Get previous losses and epochs
            losses = run_stats.get("epoch_losses", [])
            epoch_times = run_stats.get("epoch_times", [])
            best_loss = run_stats.get("best_loss", float('inf'))
            best_epoch = run_stats.get("best_epoch", 0)
            print(f"Loaded previous training stats: {len(losses)} epochs, best loss: {best_loss}")
        else:
            # Initialize empty if stats not found
            print("No previous training stats found. Starting with empty stats.")
            losses = []
            epoch_times = []
            best_loss = float('inf')
            best_epoch = 0
            
            # Initialize run statistics dictionary
            run_stats = {
                "epoch_losses": [],
                "epoch_times": [],
                "best_loss": None,
                "best_epoch": None,
                "total_training_time": None,
                "steps_per_second": [],
                "early_stopping_triggered": False,
                "early_stopping_reason": None
            }
        
        # Ensure preview directory exists (might be missing if preview_dir changed)
        os.makedirs(run_preview_dir, exist_ok=True)
    else:
        # Generate a unique GUID for this training run
        run_guid = str(uuid.uuid4())
        print(f"Starting new training run with GUID: {run_guid}")
        
        # Create run-specific output directories using the GUID
        run_output_dir = os.path.join(args.output_dir, run_guid)
        run_preview_dir = os.path.join(args.preview_dir, run_guid)
        
        # Create the base directories if they don't exist
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.preview_dir, exist_ok=True)
        
        # Create the run-specific directories
        os.makedirs(run_output_dir, exist_ok=True)
        os.makedirs(run_preview_dir, exist_ok=True)
        
        # Initialize empty stats for new run
        losses = []
        epoch_times = []
        best_loss = float('inf')
        best_epoch = 0
        
        # Initialize the run statistics dictionary
        run_stats = {
            "epoch_losses": [],
            "epoch_times": [],
            "best_loss": None,
            "best_epoch": None,
            "total_training_time": None,
            "steps_per_second": [],  # Track steps per second for each epoch
            "early_stopping_triggered": False,
            "early_stopping_reason": None
        }
      # We'll initialize per-image loss tracking later after loading the images
    # This avoids accessing image_info before it's defined
    
    # Collect system information at the beginning of the run
    system_info = collect_system_info()
    
    # Save run metadata (or update it if resuming)
    run_metadata = {
        "guid": run_guid,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arguments": vars(args),
        "system_info": system_info
    }
    
    with open(os.path.join(run_output_dir, "run_metadata.json"), 'w') as f:
        import json
        json.dump(run_metadata, f, indent=2)
    
    # Configure PyTorch for optimal performance
    if args.num_workers is not None:
        # Set the number of threads used by PyTorch
        torch.set_num_threads(args.num_workers)
    else:
        # Auto-detect based on CPU cores
        num_cores = os.cpu_count()
        if num_cores:
            # Leave 1-2 cores free for system tasks
            optimal_threads = max(1, num_cores - 2)
            torch.set_num_threads(optimal_threads)
            print(f"Setting PyTorch to use {optimal_threads} threads (detected {num_cores} CPU cores)")
    
    # Set up device
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU acceleration: {torch.cuda.get_device_name(0)}")
        
        # Set CUDA performance optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("CUDA optimizations enabled: cudnn.benchmark = True")
        
        # Set up mixed precision training if requested
        if args.mixed_precision:
            print("Enabling mixed precision training")
            # Import autocast for mixed precision
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
    else:
        print("Using CPU for training")
        # For CPU optimization, ensure we're using MKL if available
        try:
            import mkl
            print(f"Intel MKL optimization available: {mkl.get_version_string()}")
        except ImportError:
            pass
    
    # Set visualization interval (if specified separately from save interval)
    vis_interval = args.visualization_interval or args.save_interval
    
    # Load reference images and annotations if available
    images, image_info, image_size, num_conditions, tag_mapping, total_tags = load_reference_images(args, run_preview_dir)
    num_images = len(images)
    
    # Get all annotations for the loaded images
    all_annotations = []
    for info in image_info:
        if info.get('has_annotations', False):
            annotation_path = find_annotation_file(info['path'])
            if annotation_path:
                annotations = load_annotations(annotation_path)
                all_annotations.append(annotations)
            else:
                all_annotations.append(None)
        else:
            all_annotations.append(None)
    
    # Report annotation statistics
    annotated_count = sum(1 for anno in all_annotations if anno is not None)
    if annotated_count > 0:
        print(f"Using annotations for {annotated_count} of {num_images} images")
        print(f"Found {total_tags} unique tags in annotations")
        
        # Store annotation metadata
        with open(os.path.join(run_output_dir, "annotation_metadata.json"), 'w') as f:
            import json
            json.dump({
                "annotated_images": annotated_count,
                "total_images": num_images,
                "total_unique_tags": total_tags,
            }, f, indent=2)
        
        # Set flag for annotation usage in run_stats
        run_stats["annotation_usage"] = True
    else:
        print("No annotations found for any images. Training without annotation data.")
        run_stats["annotation_usage"] = False
    if num_images > 1:
        print(f"Training with {num_images} reference images using '{args.image_selection}' selection strategy")
    else:
        print("Training with a single reference image")
        
    # Initialize per-image loss tracking in the run_stats dictionary now that we have image_info
    if "per_image_losses" not in run_stats:
        run_stats["per_image_losses"] = {info["condition_id"]: [] for info in image_info}
    
    # Create a variable to track current image losses for this training session
    current_image_losses = {info["condition_id"]: [] for info in image_info}
    
    # Save condition mapping to a JSON file
    condition_mapping_path = os.path.join(run_output_dir, "condition_mapping.json")
    save_condition_mapping(image_info, condition_mapping_path)
    
    # Select a reference image for visualization
    vis_target_image, vis_target_info, vis_annotations = select_target_image(
        images, image_info, all_annotations, 0, args.image_selection
    )
    height, width = image_size
    
    # Initialize the generator model
    # If we have annotations, use tag information to set up model parameters
    has_annotations = any(anno is not None for anno in all_annotations)
    print("Initializing generator model...")
    
    generator_args = {
        'image_size': image_size,
        'latent_dim': args.latent_dim,
        'device': device,
        'mixed_precision': args.mixed_precision,
    }
    
    # Add annotation-related parameters if available
    if has_annotations and total_tags > 0:
        generator_args['tag_dim'] = total_tags
        print(f"Configuring model with tag dimension of {total_tags}")
    
    if args.full_color:
        from model_color import ImageGeneratorColor
        generator = ImageGeneratorColor(**generator_args)
        print("Using full color (RGB) model.")
    else:
        from model import ImageGenerator
        generator = ImageGenerator(**generator_args)
        print("Using grayscale model.")
    
    # If resuming, load checkpoint model
    if args.resume_guid:
        model_path = os.path.join(run_output_dir, args.resume_checkpoint)
        if not os.path.exists(model_path):
            # Try to find the most recent checkpoint (we already did this earlier but use the result)
            checkpoint_files = [f for f in os.listdir(run_output_dir) if f.startswith("generator_") and f.endswith(".pt")]
            if checkpoint_files:
                checkpoint_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]) if f[10:-3].isdigit() else 0, reverse=True)
                model_path = os.path.join(run_output_dir, checkpoint_files[0])
                print(f"Using the latest checkpoint found: {checkpoint_files[0]}")
            else:
                raise ValueError(f"Cannot resume training: no checkpoints found in {run_output_dir}")
                
        print(f"Loading model from {model_path}")
        generator.load_model(model_path)
        print(f"Model loaded successfully")
        
        # Define a starting epoch based on the checkpoint or existing stats
        last_epoch = len(run_stats["epoch_losses"]) if run_stats["epoch_losses"] else 0
        if "generator_" in os.path.basename(model_path) and os.path.basename(model_path)[10:-3].isdigit():
            checkpoint_epoch = int(os.path.basename(model_path).split('_')[1].split('.')[0])
            print(f"Checkpoint is from epoch {checkpoint_epoch}")
        else:
            checkpoint_epoch = last_epoch
            print(f"Using epoch count from stats: {checkpoint_epoch}")
    
    # Apply memory optimizations if requested
    if not args.no_optimize_memory:
        print("Applying memory optimizations...")
        # Free unused memory cache for CUDA
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # Use more efficient memory allocation in PyTorch
        if hasattr(torch, 'memory_efficient_attention'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # Create a fixed latent vector for progress visualization
    fixed_latent_vector = np.random.normal(0, 1, (1, args.latent_dim))
      # Initial visualization
    print("Creating initial visualization...")
    visualize_func = visualize_progress if args.no_async_visualization else visualize_progress_async
    
    # Calculate the right epoch number for visualization
    viz_epoch = 0
    if args.resume_guid:
        # If resuming, use the appropriate epoch number for the visualization
        viz_epoch = checkpoint_epoch if 'checkpoint_epoch' in locals() else len(run_stats["epoch_losses"])
    
    if args.no_async_visualization:
        initial_image = visualize_func(generator, viz_epoch, fixed_latent_vector, vis_target_image, 
                                      run_preview_dir, image_info=vis_target_info)
    else:
        visualize_func(generator, viz_epoch, fixed_latent_vector, vis_target_image, 
                       run_preview_dir, image_info=vis_target_info)
        
    # If annotations exist for the visualization target, create an annotated visualization
    if vis_annotations:
        viz_path = os.path.join(run_preview_dir, 'initial_with_annotations.png')
        from utils import visualize_annotations
        visualize_annotations(vis_target_image, vis_annotations, output_path=viz_path)
    
    # Training loop
    print(f"Starting training for up to {args.epochs} epochs...")
    if args.target_accuracy:
        print(f"Will stop early if accuracy reaches {args.target_accuracy}%")
    elif args.target_loss:
        print(f"Will stop early if loss reaches {args.target_loss}")
    
    # Initialize timing statistics
    start_time = time.time()
    
    # Save last generated image for entropy comparison
    last_generated_image = None
    
    # Initialize per-image loss tracking
    per_image_losses = {info["condition_id"]: [] for info in image_info}
    step_counts = {info["condition_id"]: 0 for info in image_info}
    
    # Define a custom format for tqdm to show steps/sec instead of sec/it
    class StepsPerSecondBar(tqdm):
        """Custom tqdm progress bar that shows steps/sec instead of sec/it"""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.steps_per_epoch = args[0].steps_per_epoch if hasattr(args[0], 'steps_per_epoch') else 0
            self.last_epoch_time = time.time()
            self.steps_per_second = 0
        
        def format_meter(self, n, total, elapsed, *args, **kwargs):
            # Calculate steps per second
            if elapsed > 0:
                # Get elapsed time since last update
                current_time = time.time()
                if hasattr(self, 'last_print_n') and n > self.last_print_n:
                    epoch_time = current_time - self.last_epoch_time
                    self.steps_per_second = self.steps_per_epoch / max(epoch_time, 0.001)
                    self.last_epoch_time = current_time
                self.last_print_n = n
                
                # Format with steps/sec instead of sec/it
                if self.steps_per_second > 0:
                    kwargs['rate_fmt'] = '{:.2f} steps/sec'.format(self.steps_per_second)
            return super().format_meter(n, total, elapsed, *args, **kwargs)
    
    # Create progress bar with steps/sec display - start from the right epoch if resuming
    start_epoch = 1
    if args.resume_guid:
        start_epoch = len(run_stats["epoch_losses"]) + 1
    
    epochs_to_run = range(start_epoch, args.epochs + 1)
    progress_bar = StepsPerSecondBar(epochs_to_run, 
                                    dynamic_ncols=True,
                                    unit='epoch')
    progress_bar.steps_per_epoch = args.steps_per_epoch
    
    for epoch in progress_bar:
        epoch_start = time.time()
        
        epoch_loss = 0.0
        # Run multiple training steps per epoch to improve GPU utilization
        for step in range(args.steps_per_epoch):
            # Select target image and annotations for this step
            target_image, target_info, target_annotations = select_target_image(
                images, image_info, all_annotations, epoch * step, args.image_selection
            )
            
            # Create a batch of latent vectors for better GPU utilization
            latent_vector = np.random.normal(0, 1, (args.batch_size, args.latent_dim))
            
            # Get condition ID from the target info
            condition_id = target_info['condition_id']
            
            # Apply data augmentation if annotations are available
            if target_annotations and (step % 2 == 0):  # Only augment every other step
                from utils import augment_with_annotations
                aug_image, aug_annotations = augment_with_annotations(
                    target_image, target_annotations, tag_mapping
                )
                target_image = aug_image
            
            # Create annotation masks and tensors if annotations are available
            tag_tensor = None
            region_masks = None
            
            if has_annotations and target_annotations and hasattr(generator, 'train_with_annotations'):
                from utils import create_tag_tensor, regions_to_mask, create_region_tag_tensor
                
                # Create global tag tensor
                tag_tensor = create_tag_tensor(target_annotations, tag_mapping, total_tags)
                
                # Create region masks if regions exist
                if 'regions' in target_annotations and len(target_annotations['regions']) > 0:
                    region_masks = regions_to_mask(target_annotations['regions'], target_image.shape[:2])
                    
                    # Create region-specific tag tensor
                    region_tag_tensor = create_region_tag_tensor(
                        target_annotations, tag_mapping, total_tags, target_image.shape[:2]
                    )
                    
                    # Train with full annotation data
                    step_loss = generator.train_with_annotations(
                        target_image, condition_id, latent_vector,
                        tag_tensor=tag_tensor,
                        region_masks=region_masks,
                        region_tag_tensor=region_tag_tensor
                    )
                else:
                    # Train with just global tags
                    step_loss = generator.train_with_annotations(
                        target_image, condition_id, latent_vector,
                        tag_tensor=tag_tensor
                    )
            else:                # Standard training without annotations
                step_loss = generator.train_step(target_image, condition_id, latent_vector)
                epoch_loss += step_loss
            
            # Track loss per image
            per_image_losses[condition_id].append(float(step_loss))
            step_counts[condition_id] += 1
            
        # Calculate average loss for this epoch
        avg_epoch_loss = epoch_loss / args.steps_per_epoch
        losses.append(avg_epoch_loss)
        
        # Calculate epoch timing
        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        epoch_times.append(epoch_duration)
        
        # Update run statistics
        run_stats["epoch_losses"].append(float(avg_epoch_loss))
        run_stats["epoch_times"].append(float(epoch_duration))
          # Print progress
        if epoch % 10 == 0:
            avg_time = sum(epoch_times[-10:]) / min(10, len(epoch_times[-10:]))
            remaining = (args.epochs - epoch) * avg_time
            steps_per_second = args.steps_per_epoch / epoch_duration
            
            # Create a more informative message showing which image(s) were used for training
            images_used_str = ""
            if len(images) > 1:
                # Show a summary when multiple images are used
                if args.image_selection == 'sequential':
                    current_img_idx = epoch % len(images)
                    current_img = image_info[current_img_idx]
                    images_used_str = f" [Training with: Ref #{current_img['condition_id']}: {current_img['filename']}]"
                else:
                    images_used_str = f" [Training with {len(images)} reference images]"
                    
            print(f"Epoch {epoch}/{args.epochs}, Loss: {avg_epoch_loss:.6f}, " 
                  f"Steps: {args.steps_per_epoch}, Batch: {args.batch_size}, "
                  f"Steps/sec: {steps_per_second:.2f}, "
                  f"Time: {epoch_duration:.3f}s, Remaining: {remaining/60:.1f}m{images_used_str}")
        
        # Early stopping based on target accuracy or loss
        if args.target_accuracy is not None:
            current_accuracy = 100 * (1 - avg_epoch_loss)
            if current_accuracy >= args.target_accuracy:
                print(f"Target accuracy of {args.target_accuracy}% reached at epoch {epoch}. Stopping early.")
                run_stats["early_stopping_triggered"] = True
                run_stats["early_stopping_reason"] = f"Target accuracy of {args.target_accuracy}% reached."
                break
        if args.target_loss is not None and avg_epoch_loss <= args.target_loss:
            print(f"Target loss of {args.target_loss} reached at epoch {epoch}. Stopping early.")
            run_stats["early_stopping_triggered"] = True
            run_stats["early_stopping_reason"] = f"Target loss of {args.target_loss} reached."
            break
          # Create visualization and save model if needed
        if epoch % vis_interval == 0 or epoch == args.epochs:
            # Create visualization with the current target image
            generated_image = visualize_func(generator, epoch, fixed_latent_vector, vis_target_image, 
                                            run_preview_dir, image_info=vis_target_info)
            
            # If we have multiple images, create visualizations for each one
            if len(images) > 1 and epoch % (vis_interval * 2) == 0:
                for idx, (img, info) in enumerate(zip(images, image_info)):
                    # Skip the main visualization image since we already did it
                    if info['condition_id'] == vis_target_info['condition_id']:
                        continue
                    
                    # Create a specific directory for this reference image
                    img_specific_dir = os.path.join(run_preview_dir, f"ref_{info['condition_id']}")
                    os.makedirs(img_specific_dir, exist_ok=True)
                    
                    # Create visualization for this specific reference image
                    visualize_func(generator, epoch, fixed_latent_vector, img, img_specific_dir, 
                                  prefix=f"progress", image_info=info)
            
            # If annotations are available, create a visualization with them
            if vis_annotations and epoch % (vis_interval * 2) == 0:
                from utils import visualize_annotations
                viz_path = os.path.join(run_preview_dir, f'annotated_{epoch:04d}.png')
                visualize_annotations(vis_target_image, vis_annotations, output_path=viz_path)
            
            # Save model checkpoint
            checkpoint_path = os.path.join(run_output_dir, f"generator_{epoch:04d}.pt")
            generator.save_model(checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")
            
            # Register this model for tracking and potential cleanup
            register_model_checkpoint(checkpoint_path, epoch, avg_epoch_loss)
              # Save loss plots
            save_loss_plot(losses, run_preview_dir)
            
            # Save per-image loss plots if we have multiple images
            if len(images) > 1:
                save_per_image_loss_plot(per_image_losses, image_info, run_preview_dir)
            
            # Save per-image losses in the run_stats
            run_stats["per_image_losses"] = {k: v for k, v in per_image_losses.items()}
            run_stats["per_image_step_counts"] = {k: v for k, v in step_counts.items()}
            
            # Save intermediate run statistics
            with open(os.path.join(run_output_dir, "run_stats.json"), 'w') as f:
                import json
                json.dump(run_stats, f, indent=2)
                
            # Check if generated images have changed significantly since last visualization
            if last_generated_image is not None and generated_image is not None:
                # Calculate similarity between current and previous generated image
                similarity = calculate_image_similarity(last_generated_image, generated_image)
                
                # Add to run stats
                if "image_similarities" not in run_stats:
                    run_stats["image_similarities"] = []
                run_stats["image_similarities"].append(similarity)
                # Optionally, print SSIM for user feedback
                print(f"Image similarity (MSE-based): {similarity['mse_similarity']:.4f}, SSIM: {similarity['ssim']:.4f}")
                
                # Check if images are too similar (training might be stalled)
                # Or if it's a blank white image (common failure case)
                is_blank = np.mean(generated_image) > 0.95  # Mostly white
                
                if similarity["mse_similarity"] > 0.98 or (is_blank and epoch > vis_interval*2):  # Threshold for detecting stalled training
                    print(f"WARNING: Generated images have similarity of {similarity['mse_similarity']:.4f}, training may be stalled.")
                    
                    # Log the stalled event
                    if "entropy_injections" not in run_stats:
                        run_stats["entropy_injections"] = []
                    
                    run_stats["entropy_injections"].append({
                        "epoch": epoch,
                        "similarity": similarity,
                        "is_blank": bool(is_blank)
                    })
                    
                    # Inject entropy to escape potential local minimum
                    fixed_latent_vector = inject_training_entropy(generator, fixed_latent_vector, epoch)
                      # Force an immediate new visualization with the modified parameters
                    # to verify the entropy injection had an effect
                    new_generated_image = visualize_progress(generator, epoch, fixed_latent_vector, 
                                                          vis_target_image, run_preview_dir, 
                                                          prefix=f"entropy_{epoch}", save_comparison=False,
                                                          image_info=vis_target_info)
                    generated_image = new_generated_image
            
            # Store current generated image for next comparison
            last_generated_image = generated_image
        
        # Save the best model based on loss
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_epoch = epoch
            best_model_path = os.path.join(run_output_dir, "generator_best.pt")
            generator.save_model(best_model_path)
            
            # Update run statistics for best model
            run_stats["best_loss"] = float(best_loss)
            run_stats["best_epoch"] = epoch

    # Define best_model_path here to ensure it's always available, even with early stopping
    best_model_path = os.path.join(run_output_dir, "generator_best.pt")
        
    # Training complete, show timing information
    total_time = time.time() - start_time
    avg_epoch_time = total_time / (epoch - start_epoch + 1)  # Use actual epochs run
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Average time per epoch: {avg_epoch_time:.3f} seconds")
    
    # Update final run statistics
    if "total_training_time" in run_stats and run_stats["total_training_time"] is not None:
        # Add to existing training time if resuming
        run_stats["total_training_time"] += float(total_time)
    else:
        run_stats["total_training_time"] = float(total_time)
        
    run_stats["avg_epoch_time"] = float(avg_epoch_time)
    
    # Add per-image summary statistics
    if len(image_info) > 1:
        per_image_summary = {}
        for info in image_info:
            condition_id = info['condition_id']
            if condition_id in per_image_losses and len(per_image_losses[condition_id]) > 0:
                losses = per_image_losses[condition_id]
                per_image_summary[condition_id] = {
                    "filename": info['filename'],
                    "training_steps": len(losses),
                    "avg_loss": float(sum(losses) / len(losses)) if losses else 0,
                    "best_loss": float(min(losses)) if losses else 0,
                    "best_step": losses.index(min(losses)) + 1 if losses else 0
                }
        run_stats["per_image_summary"] = per_image_summary
    
    # Save final run statistics
    with open(os.path.join(run_output_dir, "run_stats.json"), 'w') as f:
        import json
        json.dump(run_stats, f, indent=2)
    
    # Save final model
    final_model_path = os.path.join(run_output_dir, "generator_final.pt")
    generator.save_model(final_model_path)
    print(f"Training complete! Final model saved to {final_model_path}")
    
    # Final visualization with the best model
    print("Loading best model for final visualization...")
    generator.load_model(best_model_path)
    
    # For the final visualization, create comparisons with each reference image
    print("Creating final visualizations...")
    
    # Create a grid comparison of the model's output against all reference images
    if num_images > 1:
        create_final_comparison_grid(generator, fixed_latent_vector, images, image_info, run_preview_dir)
        
        # Save final per-image loss plots
        if len(per_image_losses) > 1:
            save_per_image_loss_plot(per_image_losses, image_info, run_preview_dir)
            
            # Print summary of per-image training performance
            print("\nPer-image training performance summary:")
            print("----------------------------------------")
            for info in image_info:
                condition_id = info['condition_id']
                if condition_id in per_image_losses and len(per_image_losses[condition_id]) > 0:
                    losses = per_image_losses[condition_id]
                    avg_loss = sum(losses) / len(losses) if losses else 0
                    best_loss = min(losses) if losses else 0
                    best_step = losses.index(best_loss) + 1 if losses else 0
                    print(f"  {info['filename']} (ID {condition_id}):")
                    print(f"    - Training steps: {len(losses)}")
                    print(f"    - Average loss: {avg_loss:.6f}")
                    print(f"    - Best loss: {best_loss:.6f} (step {best_step})")
            print("----------------------------------------\n")
        
        # Create individual final visualizations for each reference image
        print(f"Creating individual final visualizations for {num_images} reference images...")
        for idx, (img, info) in enumerate(zip(images, image_info)):
            # Create a specific directory for this reference image
            img_specific_dir = os.path.join(run_preview_dir, f"ref_{info['condition_id']}")
            os.makedirs(img_specific_dir, exist_ok=True)
            
            # Create final visualization for this specific reference image
            visualize_progress(generator, args.epochs, fixed_latent_vector, img, 
                              img_specific_dir, prefix="final", image_info=info)
            
            print(f"  - Created final visualization for {info['filename']} (ID: {info['condition_id']})")
    
    # Regular final visualization with the first image
    final_image = visualize_progress(generator, args.epochs, fixed_latent_vector, 
                                   vis_target_image, run_preview_dir, prefix="final", 
                                   save_comparison=True, image_info=vis_target_info)
    
    # If annotations are available, create a final annotated visualization
    if vis_annotations:
        from utils import visualize_annotations
        viz_path = os.path.join(run_preview_dir, f'final_annotated.png')
        visualize_annotations(vis_target_image, vis_annotations, output_path=viz_path)
    
    print(f"Best loss achieved: {best_loss:.6f}")
    print(f"Progress visualizations saved to {run_preview_dir}")
    print(f"Run GUID: {run_guid}")
    print(f"Run output directory: {run_output_dir}")
    
    # Save final generated image as SVG
    final_svg_path = os.path.join(run_preview_dir, "final_generated_image.svg")
    if final_image is not None:
        save_as_svg(final_image, final_svg_path)
        print(f"Saved final generated SVG to {final_svg_path}")
    else:
        print(f"Skipping SVG save for {final_svg_path} because final_image is None.")

    # Create a symlink or copy of the latest run folder for easy access
    latest_run_link = os.path.join(args.output_dir, "latest_run")
    if os.path.exists(latest_run_link) and os.path.islink(latest_run_link):
        os.unlink(latest_run_link)
    elif os.path.exists(latest_run_link):
        import shutil
        shutil.rmtree(latest_run_link)
        
    try:
        # Try to create a symbolic link first (works on Unix systems)
        os.symlink(run_guid, latest_run_link)
    except (OSError, AttributeError):
        # On Windows, symlinks might require admin privileges or not be supported
        # Instead, create a text file with the GUID
        with open(os.path.join(args.output_dir, "latest_run.txt"), 'w') as f:
            f.write(f"Latest run GUID: {run_guid}\n")
            f.write(f"Directory: {run_output_dir}\n")
            f.write(f"Timestamp: {run_metadata['timestamp']}\n")
    
    # Final cleanup of any excess models if needed
    if len(model_checkpoints) > MAX_MODELS_TO_KEEP:
        print(f"Performing final cleanup to ensure only top {MAX_MODELS_TO_KEEP} models are kept...")
        cleanup_thread = threading.Thread(target=cleanup_models)
        cleanup_thread.start()
        cleanup_thread.join(timeout=10)  # Wait for up to 10 seconds for cleanup to complete
    
    # Save comprehensive run summary
    run_summary_path = os.path.join(run_output_dir, "run_summary.txt")
    create_run_summary(run_guid, run_metadata, run_stats, image_info, system_info, run_summary_path)
    
    # Save model architecture information
    model_architecture_path = os.path.join(run_output_dir, "model_architecture.txt")
    save_model_architecture_info(generator, model_architecture_path)
    
    # Create CSV file with epoch-by-epoch statistics
    csv_path = os.path.join(run_output_dir, "epoch_stats.csv")
    with open(csv_path, 'w') as f:
        f.write("Epoch,Loss,Duration(sec)\n")
        for i, (loss, duration) in enumerate(zip(run_stats["epoch_losses"], run_stats["epoch_times"])):
            f.write(f"{i+1},{loss},{duration}\n")
    print(f"Epoch-by-epoch statistics saved to {csv_path}")
    
    print(f"All training statistics and metadata saved to {run_output_dir}")
    print(f"To review this run, see {run_summary_path}")
    
    return run_guid

if __name__ == "__main__":
    main()
