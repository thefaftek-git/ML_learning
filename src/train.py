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
from tqdm import tqdm

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
    parser.add_argument('--reference', type=str, default='reference.png',
                      help='Path to the target reference image (JPG, PNG, or SVG)')
    parser.add_argument('--output-dir', type=str, default='models',
                      help='Directory to save model checkpoints')
    parser.add_argument('--image-size', type=int, default=None,
                      help='Base size for the output image (defaults to reference image size)')
    parser.add_argument('--preserve-aspect', action='store_true',
                      help='Preserve the aspect ratio of the reference image')
    parser.add_argument('--latent-dim', type=int, default=100,
                      help='Dimension of the latent space')
    parser.add_argument('--epochs', type=int, default=2000,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                      help='Batch size for training')
    parser.add_argument('--save-interval', type=int, default=100,
                      help='Interval for saving model checkpoints and samples')
    parser.add_argument('--preview-dir', type=str, default='data/progress',
                      help='Directory to save progress preview images')
    # Performance optimization options
    parser.add_argument('--visualization-interval', type=int, default=None,
                      help='Interval for visualizations (separate from model saving, set higher to speed up training)')
    parser.add_argument('--async-visualization', action='store_true',
                      help='Generate visualizations in background threads to avoid pausing training')
    parser.add_argument('--num-workers', type=int, default=None,
                      help='Number of worker threads for PyTorch (default: auto-detected based on CPU cores)')
    parser.add_argument('--use-gpu', action='store_true',
                      help='Use GPU acceleration if available')
    parser.add_argument('--mixed-precision', action='store_true',
                      help='Use mixed precision training (speeds up GPU training)')
    parser.add_argument('--optimize-memory', action='store_true',
                      help='Apply memory usage optimizations')
    
    return parser.parse_args()

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
    
    # Preprocess the reference image
    print(f"Processing reference image: {args.reference}")
    reference_path = os.path.join(os.getcwd(), args.reference)
    
    # Get the processed image and original dimensions
    if args.image_size is None:
        # Use original dimensions from the reference image
        target_image, original_dimensions = preprocess_reference_image(
            reference_path, args.preview_dir, size=None
        )
        image_size = original_dimensions
        print(f"Using reference image's original dimensions: {image_size[0]}x{image_size[1]}")
    else:
        # Use the user-specified size
        size = (args.image_size, args.image_size)
        target_image, _ = preprocess_reference_image(
            reference_path, args.preview_dir, size=size
        )
        image_size = size
        print(f"Using user-specified dimensions: {image_size[0]}x{image_size[1]}")
    
    # Initialize the generator model
    print("Initializing generator model...")
    generator = ImageGenerator(
        image_size=image_size,
        latent_dim=args.latent_dim,
        device=device,
        mixed_precision=args.mixed_precision
    )
    
    # Apply memory optimizations if requested
    if args.optimize_memory:
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
    visualize_func = visualize_progress_async if args.async_visualization else visualize_progress
    if not args.async_visualization:
        initial_image = visualize_func(generator, 0, fixed_latent_vector, target_image, args.preview_dir)
    else:
        visualize_func(generator, 0, fixed_latent_vector, target_image, args.preview_dir)
    
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
        
        # In each epoch we'll train with a random latent vector
        latent_vector = np.random.normal(0, 1, (args.batch_size, args.latent_dim))
        
        # Train for one step
        loss = generator.train_step(target_image, latent_vector)
        losses.append(loss)
        
        # Calculate epoch timing
        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        epoch_times.append(epoch_duration)
        
        # Print progress
        if epoch % 10 == 0:
            avg_time = sum(epoch_times[-10:]) / min(10, len(epoch_times[-10:]))
            remaining = (args.epochs - epoch) * avg_time
            print(f"Epoch {epoch}/{args.epochs}, Loss: {loss:.6f}, " 
                  f"Time: {epoch_duration:.3f}s, Remaining: {remaining/60:.1f}m")
        
        # Create visualization if needed
        if epoch % vis_interval == 0 or epoch == args.epochs:
            visualize_func(generator, epoch, fixed_latent_vector, target_image, args.preview_dir)
        
        # Save model checkpoint and additional data
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            # Save model checkpoint
            checkpoint_path = os.path.join(args.output_dir, f"generator_{epoch:04d}.pt")
            generator.save_model(checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")
            
            # Register this model for tracking and potential cleanup
            register_model_checkpoint(checkpoint_path, epoch, loss)
            
            # Save loss plot
            save_loss_plot(losses, args.preview_dir)
        
        # Save the best model based on loss
        if loss < best_loss:
            best_loss = loss
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
    final_image = visualize_progress(generator, args.epochs, fixed_latent_vector, 
                                   target_image, args.preview_dir, prefix="final", save_comparison=True)
    
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