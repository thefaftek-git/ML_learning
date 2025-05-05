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
                      help='Number of training epochs')
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
    
    return parser.parse_args()

def load_reference_images(args, preview_dir):
    """
    Load reference images based on command-line arguments.
    
    This function handles both single image and directory-based image loading.
    
    Args:
        args: Command-line arguments
        preview_dir: Directory to save preview images
        
    Returns:
        A tuple containing:
        - List of preprocessed images
        - List of image info dictionaries (each containing filename, dimensions, condition_id, etc.)
        - Common image dimensions to use (height, width)
        - Number of conditions (number of unique images)
    """
    images = []
    image_info = []
    
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
        _, original_dimensions = preprocess_reference_image(
            first_image_path, preview_dir, size=None, show_preview=False
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
            processed_img, _ = preprocess_reference_image(
                img_path, preview_dir, size=target_dimensions, show_preview=False
            )
            
            # Ensure all images have the same dimensions
            if processed_img.shape[0] != height or processed_img.shape[1] != width:
                print(f"Warning: Image {filename} has different dimensions. Resizing to match others.")
                processed_img = resize_image(processed_img, (height, width))
            
            images.append(processed_img)
            image_info.append({
                'filename': filename,
                'path': img_path,
                'original_dimensions': original_dimensions,
                'condition_id': condition_id  # Store condition ID with image info
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
            processed_img, original_dimensions = preprocess_reference_image(
                reference_path, preview_dir, size=None
            )
            height, width = original_dimensions
            print(f"Using reference image's original dimensions: {height}x{width}")
        else:
            # Calculate target dimensions based on arguments
            _, original_dimensions = preprocess_reference_image(
                reference_path, preview_dir, size=None, show_preview=False
            )
            original_height, original_width = original_dimensions
            
            # Calculate target dimensions
            target_dimensions = calculate_target_dimensions(
                args, original_height, original_width, preview_dir
            )
            height, width = target_dimensions
            
            # Process the image using the target dimensions
            processed_img, _ = preprocess_reference_image(
                reference_path, preview_dir, size=target_dimensions
            )
        
        # Add the single image to our lists with condition_id = 0
        images.append(processed_img)
        image_info.append({
            'filename': os.path.basename(reference_path),
            'path': reference_path,
            'original_dimensions': original_dimensions,
            'condition_id': 0  # Single image always uses condition ID 0
        })
        
        # Only one condition for a single image
        num_conditions = 1
    
    return images, image_info, (height, width), num_conditions

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
        ax.imshow(img[:, :, 0], cmap='gray')
        ax.set_title(f"{info['filename']}")
        ax.axis('off')
    
    plt.tight_layout()
    grid_path = os.path.join(output_dir, 'reference_grid.png')
    plt.savefig(grid_path)
    plt.close()
    
    print(f"Created reference grid visualization: {grid_path}")

def select_target_image(images, image_info, epoch, selection_method='random'):
    """
    Select a target image for the current training epoch.
    
    Args:
        images: List of processed images
        image_info: List of image info dictionaries
        epoch: Current training epoch
        selection_method: How to select the image ('random' or 'sequential')
        
    Returns:
        Tuple of (selected image, image information)
    """
    if len(images) == 1:
        # If only one image, always return it
        return images[0], image_info[0]
    
    if selection_method == 'random':
        # Randomly select an image
        idx = np.random.randint(0, len(images))
        return images[idx], image_info[idx]
    else:
        # Sequential selection - cycle through images based on epoch
        idx = epoch % len(images)
        return images[idx], image_info[idx]

def visualize_progress(generator, epoch, latent_vector, target_image, output_dir, prefix="progress", save_comparison=False):
    """Save a visualization of training progress."""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Generate image with the current model
        generated_image = generator.generate_image(latent_vector)
        
        # Create a figure with the target and generated images side by side
        fig = plt.figure(figsize=(10, 5))
        
        # Plot target image
        plt.subplot(1, 2, 1)
        plt.imshow(target_image[:, :, 0], cmap='gray')
        plt.title("Target Image")
        plt.axis('off')
        
        # Plot generated image
        plt.subplot(1, 2, 2)
        plt.imshow(generated_image[:, :, 0], cmap='gray')
        plt.title(f"Generated (Epoch {epoch})")
        plt.axis('off')
        
        # Save the visualization
        plt.tight_layout()
        viz_path = os.path.join(output_dir, f"{prefix}_{epoch:04d}.png")
        plt.savefig(viz_path)
        
        # If this is a final comparison, also save it to the main data folder
        if save_comparison:
            comparison_path = os.path.join(os.path.dirname(output_dir), "final_comparison.png")
            plt.savefig(comparison_path)
            print(f"Final comparison saved to {comparison_path}")
        
        # Explicitly close the figure to free memory and avoid tkinter issues
        plt.close(fig)
        
        # Also save the generated image as SVG
        svg_path = os.path.join(output_dir, f"{prefix}_{epoch:04d}.svg")
        save_as_svg(generated_image, svg_path)
        
        return generated_image
    
    except Exception as e:
        print(f"Error during visualization: {e}")
        # Make sure we clean up even if there's an error
        plt.close('all')
        return None

def visualize_progress_async(generator, epoch, latent_vector, target_image, output_dir, prefix="progress", save_comparison=False):
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
        args=(generator, epoch, latent_vector, target_image, output_dir, prefix, save_comparison),
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

def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.preview_dir, exist_ok=True)
    
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
    
    # Load reference images
    images, image_info, image_size, num_conditions = load_reference_images(args, args.preview_dir)
    num_images = len(images)
    
    if num_images > 1:
        print(f"Training with {num_images} reference images using '{args.image_selection}' selection strategy")
    else:
        print("Training with a single reference image")
    
    # Save condition mapping to a JSON file
    condition_mapping_path = os.path.join(args.output_dir, "condition_mapping.json")
    save_condition_mapping(image_info, condition_mapping_path)
    
    # Select a reference image for visualization
    vis_target_image = images[0]  # Use the first image for visualization
    height, width = image_size
    
    # Initialize the generator model
    print("Initializing generator model...")
    generator = ImageGenerator(
        image_size=image_size,
        latent_dim=args.latent_dim,
        device=device,
        mixed_precision=args.mixed_precision
    )
    
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
    if args.no_async_visualization:
        initial_image = visualize_func(generator, 0, fixed_latent_vector, vis_target_image, args.preview_dir)
    else:
        visualize_func(generator, 0, fixed_latent_vector, vis_target_image, args.preview_dir)
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    losses = []
    best_loss = float('inf')
    
    # Scan for existing model checkpoints in the output directory
    print(f"Scanning for existing model checkpoints in {args.output_dir}...")
    for filename in os.listdir(args.output_dir):
        if filename.startswith("generator_") and filename.endswith(".pt") and filename not in PROTECTED_MODELS:
            try:
                # Extract epoch number from filename (format: generator_XXXX.pt)
                epoch_str = filename.replace("generator_", "").replace(".pt", "")
                epoch_num = int(epoch_str)
                
                # Load the model to get its loss (could be expensive but only done once at startup)
                model_path = os.path.join(args.output_dir, filename)
                print(f"Registering existing model: {filename}")
                
                # Since we don't have the loss value for old models, use epoch number as proxy
                # Higher epoch generally means better model (but not always)
                register_model_checkpoint(model_path, epoch_num, 1.0 / (epoch_num + 1))
            except Exception as e:
                print(f"Error processing existing model {filename}: {e}")
    
    # Initialize timing statistics
    start_time = time.time()
    epoch_times = []
    
    for epoch in tqdm(range(1, args.epochs + 1)):
        epoch_start = time.time()
        
        epoch_loss = 0.0
        # Run multiple training steps per epoch to improve GPU utilization
        for step in range(args.steps_per_epoch):
            # Select target image for this step
            target_image, target_info = select_target_image(images, image_info, epoch * step, args.image_selection)
            
            # Create a batch of latent vectors for better GPU utilization
            latent_vector = np.random.normal(0, 1, (args.batch_size, args.latent_dim))
            
            # Train with the full batch - pass the condition ID from the target info
            condition_id = target_info['condition_id']
            step_loss = generator.train_step(target_image, condition_id, latent_vector)
            epoch_loss += step_loss
            
        # Calculate average loss for this epoch
        avg_epoch_loss = epoch_loss / args.steps_per_epoch
        losses.append(avg_epoch_loss)
        
        # Calculate epoch timing
        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        epoch_times.append(epoch_duration)
        
        # Print progress
        if epoch % 10 == 0:
            avg_time = sum(epoch_times[-10:]) / min(10, len(epoch_times[-10:]))
            remaining = (args.epochs - epoch) * avg_time
            print(f"Epoch {epoch}/{args.epochs}, Loss: {avg_epoch_loss:.6f}, " 
                  f"Steps: {args.steps_per_epoch}, Batch: {args.batch_size}, "
                  f"Time: {epoch_duration:.3f}s, Remaining: {remaining/60:.1f}m")
        
        # Create visualization and save model if needed
        if epoch % vis_interval == 0 or epoch == args.epochs:
            # Create visualization with the current target image
            visualize_func(generator, epoch, fixed_latent_vector, vis_target_image, args.preview_dir)
            
            # Save model checkpoint
            checkpoint_path = os.path.join(args.output_dir, f"generator_{epoch:04d}.pt")
            generator.save_model(checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")
            
            # Register this model for tracking and potential cleanup
            register_model_checkpoint(checkpoint_path, epoch, avg_epoch_loss)
            
            # Save loss plot
            save_loss_plot(losses, args.preview_dir)
        
        # Save the best model based on loss
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_path = os.path.join(args.output_dir, "generator_best.pt")
            generator.save_model(best_model_path)
    
    # Training complete, show timing information
    total_time = time.time() - start_time
    avg_epoch_time = total_time / args.epochs
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Average time per epoch: {avg_epoch_time:.3f} seconds")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "generator_final.pt")
    generator.save_model(final_model_path)
    print(f"Training complete! Final model saved to {final_model_path}")
    
    # Final visualization with the best model
    print("Loading best model for final visualization...")
    generator.load_model(best_model_path)
    
    # For the final visualization, create comparisons with each reference image
    print("Creating final visualizations...")
    
    # Create a grid comparison of the model's output against all reference images
    if num_images > 1:
        create_final_comparison_grid(generator, fixed_latent_vector, images, image_info, args.preview_dir)
    
    # Regular final visualization with the first image
    final_image = visualize_progress(generator, args.epochs, fixed_latent_vector, 
                                   vis_target_image, args.preview_dir, prefix="final", save_comparison=True)
    
    print(f"Best loss achieved: {best_loss:.6f}")
    print(f"Progress visualizations saved to {args.preview_dir}")
    
    # Save the final best image as SVG
    final_svg_path = os.path.join(os.getcwd(), 'data', 'final_output.svg')
    os.makedirs(os.path.dirname(final_svg_path), exist_ok=True)
    save_as_svg(final_image, final_svg_path)
    print(f"Final SVG output saved to {final_svg_path}")
    
    # Final cleanup of any excess models if needed
    if len(model_checkpoints) > MAX_MODELS_TO_KEEP:
        print(f"Performing final cleanup to ensure only top {MAX_MODELS_TO_KEEP} models are kept...")
        cleanup_thread = threading.Thread(target=cleanup_models)
        cleanup_thread.start()
        cleanup_thread.join(timeout=10)  # Wait for up to 10 seconds for cleanup to complete

if __name__ == "__main__":
    main()