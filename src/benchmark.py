"""
ML System Benchmarking Script

This script benchmarks different batch sizes and steps-per-epoch configurations
to help determine the optimal settings for your training workload.
"""

import os
import time
import json
import argparse
import numpy as np
import torch
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

# Import from your existing codebase
from utils import load_image, preprocess_reference_image

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Benchmark different batch sizes and steps per epoch')
    
    # Basic benchmark settings
    parser.add_argument('--reference', type=str, default='reference.png',
                      help='Path to a target reference image (JPG, PNG, or SVG)')
    parser.add_argument('--image-size', type=int, default=256,
                      help='Size for the output image')
    parser.add_argument('--latent-dim', type=int, default=100,
                      help='Dimension of the latent space')
    parser.add_argument('--use-gpu', action='store_true', default=True,
                      help='Use GPU acceleration if available')
    parser.add_argument('--mixed-precision', action='store_true',
                      help='Use mixed precision training (speeds up GPU training)')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                      help='Directory to save benchmark results')
    parser.add_argument('--full-color', action='store_true',
                      help='Use a color (RGB) model instead of grayscale for benchmarking')
    parser.add_argument('--resume', action='store_true',
                      help='Resume benchmarking from the last saved progress in the output directory')

    # Benchmark-specific options
    parser.add_argument('--test-epochs', type=int, default=10,
                      help='Number of test epochs for each configuration')
    parser.add_argument('--warm-up-epochs', type=int, default=2,
                      help='Number of warm-up epochs before measurement (to stabilize GPU usage)')
    
    # Batch size options
    parser.add_argument('--min-batch', type=int, default=1,
                      help='Minimum batch size to test')
    parser.add_argument('--max-batch', type=int, default=128,
                      help='Maximum batch size to test')
    parser.add_argument('--batch-step', type=int, default=None,
                      help='Step size for batch testing (if None, will use exponential scale)')
    
    # Steps per epoch options
    parser.add_argument('--min-steps', type=int, default=1,
                      help='Minimum steps per epoch to test')
    parser.add_argument('--max-steps', type=int, default=50,
                      help='Maximum steps per epoch to test')
    parser.add_argument('--steps-step', type=int, default=None,
                      help='Step size for steps testing (if None, will use exponential scale)')
    
    # Thread options
    parser.add_argument('--test-threads', action='store_true',
                      help='Whether to test different thread configurations')
    parser.add_argument('--min-threads', type=int, default=1,
                      help='Minimum number of threads to test')
    parser.add_argument('--max-threads', type=int, default=64,
                      help='Maximum number of threads to test (default is system CPU count)')
    parser.add_argument('--thread-step', type=int, default=None,
                      help='Step size for thread testing (if None, will use exponential scale)')
    
    return parser.parse_args()

def get_gpu_metrics():
    """Get GPU metrics including utilization and memory usage."""
    if not torch.cuda.is_available():
        return {'gpu_utilization': 0, 'gpu_memory_used': 0, 'gpu_memory_total': 0}
    
    try:
        # Initialize metrics
        metrics = {}
        
        # Get memory information
        memory_used = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
        
        metrics['gpu_memory_used'] = memory_used
        metrics['gpu_memory_total'] = memory_total
        metrics['gpu_memory_percent'] = (memory_used / memory_total) * 100
        
        # Try to get GPU utilization if nvml is available
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics['gpu_utilization'] = util.gpu
        except (ImportError, Exception) as e:
            # If pynvml is not available or fails, we can't get utilization
            metrics['gpu_utilization'] = None
            
        return metrics
    except Exception as e:
        print(f"Error getting GPU metrics: {e}")
        return {'gpu_utilization': None, 'gpu_memory_used': None, 'gpu_memory_total': None}

def get_cpu_metrics():
    """Get CPU metrics including utilization and memory usage."""
    try:
        metrics = {}
        # Get CPU utilization
        metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        
        # Get memory information
        memory = psutil.virtual_memory()
        metrics['ram_used'] = memory.used / (1024 ** 3)  # Convert to GB
        metrics['ram_total'] = memory.total / (1024 ** 3)  # Convert to GB
        metrics['ram_percent'] = memory.percent
        
        return metrics
    except Exception as e:
        print(f"Error getting CPU metrics: {e}")
        return {'cpu_percent': None, 'ram_used': None, 'ram_total': None, 'ram_percent': None}

def run_benchmark(batch_size, steps_per_epoch, image, args, device, num_threads=None, ModelClass=None):
    """
    Run a training benchmark with the specified configuration.
    
    Args:
        batch_size: Batch size to test
        steps_per_epoch: Steps per epoch to test
        image: Reference image to use for training
        args: Command line arguments
        device: PyTorch device to use
        num_threads: Number of threads to use for intra-op parallelism (None = default)
        ModelClass: Model class to use for instantiation
        
    Returns:
        Dictionary containing benchmark results
    """
    # Configure thread settings if specified
    previous_num_threads = None
    if num_threads is not None:
        print(f"Setting thread count to: {num_threads}")
        # Save previous thread setting to restore later
        previous_num_threads = torch.get_num_threads()
        # Set new thread count
        torch.set_num_threads(num_threads)
    
    print(f"Benchmarking batch_size={batch_size}, steps_per_epoch={steps_per_epoch}" + 
          (f", threads={num_threads}" if num_threads is not None else ""))
    
    # Initialize the model
    generator = ModelClass(
        image_size=image.shape[:2],
        latent_dim=args.latent_dim,
        device=device,
        mixed_precision=args.mixed_precision
    )

    # Create a fixed latent vector for consistency
    fixed_latent_vector = np.random.normal(0, 1, (1, args.latent_dim))
    
    # Some variables to track metrics
    epoch_times = []
    step_times = []
    losses = []
    gpu_utils = []
    gpu_mems = []
    cpu_utils = []
    ram_utils = []
    
    # Run warm-up epochs
    print(f"Running {args.warm_up_epochs} warm-up epochs...")
    for epoch in range(args.warm_up_epochs):
        # Run training steps
        for step in range(steps_per_epoch):
            # Create a batch of latent vectors
            latent_vector = np.random.normal(0, 1, (batch_size, args.latent_dim))
            
            # Train step
            step_loss = generator.train_step(image, 0, latent_vector)
    
    # Empty CUDA cache between tests
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run benchmark epochs
    print(f"Running {args.test_epochs} benchmark epochs...")
    for epoch in range(args.test_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        # Run training steps
        for step in range(steps_per_epoch):
            step_start = time.time()
            
            # Create a batch of latent vectors
            latent_vector = np.random.normal(0, 1, (batch_size, args.latent_dim))
            
            # Train step
            step_loss = generator.train_step(image, 0, latent_vector)
            epoch_loss += step_loss
            
            # Measure step time
            step_end = time.time()
            step_times.append(step_end - step_start)
            
            # Get resource utilization
            if step == steps_per_epoch // 2:  # Measure in the middle of the epoch
                gpu_metrics = get_gpu_metrics()
                cpu_metrics = get_cpu_metrics()
                
                if gpu_metrics['gpu_utilization'] is not None:
                    gpu_utils.append(gpu_metrics['gpu_utilization'])
                if gpu_metrics['gpu_memory_percent'] is not None:
                    gpu_mems.append(gpu_metrics['gpu_memory_percent'])
                    
                cpu_utils.append(cpu_metrics['cpu_percent'])
                ram_utils.append(cpu_metrics['ram_percent'])
        
        # Calculate metrics for this epoch
        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        epoch_times.append(epoch_duration)
        
        avg_loss = epoch_loss / steps_per_epoch
        losses.append(avg_loss)
        
        print(f"  Epoch {epoch+1}/{args.test_epochs}, Loss: {avg_loss:.6f}, Time: {epoch_duration:.3f}s")
    
    # After benchmark epochs, generate an image for accuracy/SSIM evaluation
    with torch.no_grad():
        latent_vector = np.random.normal(0, 1, (1, args.latent_dim))
        generated_image = generator.generate_image(latent_vector)
        # Ensure both images are in [0,1] and same shape
        ref_img = image
        gen_img = generated_image
        if ref_img.shape != gen_img.shape:
            # Resize generated image to match reference if needed
            from skimage.transform import resize
            gen_img = resize(gen_img, ref_img.shape, preserve_range=True, anti_aliasing=True)        # Ensure images have proper dimensionality and handle both grayscale and color
        # First check if the shapes actually match, not just the dimensions
        if np.array_equal(ref_img.shape, gen_img.shape):
            # Handle different versions of scikit-image SSIM function
            if len(ref_img.shape) == 2 or ref_img.shape[-1] == 1:
                # Grayscale images
                if len(ref_img.shape) == 3 and ref_img.shape[-1] == 1:
                    ref_img_2d = ref_img[:, :, 0]  # Convert to 2D
                    gen_img_2d = gen_img[:, :, 0]  # Convert to 2D
                else:
                    ref_img_2d = ref_img
                    gen_img_2d = gen_img
                ssim_score = ssim(ref_img_2d, gen_img_2d, data_range=1.0)
            else:
                # Color images
                try:
                    # Newer scikit-image versions use channel_axis
                    ssim_score = ssim(ref_img, gen_img, data_range=1.0, channel_axis=-1)
                except TypeError:
                    try:
                        # Older scikit-image versions use multichannel
                        ssim_score = ssim(ref_img, gen_img, data_range=1.0, multichannel=True)
                    except Exception as e:
                        print(f"Error computing SSIM: {e}. Using default score.")
                        ssim_score = 0.0
        else:
            print(f"Shape mismatch: ref_img {ref_img.shape} vs gen_img {gen_img.shape}. Using default SSIM score.")
            ssim_score = 0.0
    
    # Calculate overall metrics
    avg_epoch_time = np.mean(epoch_times)
    avg_step_time = np.mean(step_times)
    avg_loss = np.mean(losses)
    
    # Calculate steps per second (instead of samples per second)
    steps_per_second = steps_per_epoch / avg_epoch_time
    
    # For backward compatibility, still calculate samples per second
    samples_per_second = batch_size * steps_per_epoch / avg_epoch_time
    
    # GPU utilization metrics
    avg_gpu_util = np.mean(gpu_utils) if gpu_utils else None
    avg_gpu_mem = np.mean(gpu_mems) if gpu_mems else None
    
    # CPU utilization metrics
    avg_cpu_util = np.mean(cpu_utils) if cpu_utils else None
    avg_ram_util = np.mean(ram_utils) if ram_utils else None
    
    # Calculate efficiency metrics
    # Use steps per second as the primary throughput metric
    throughput = steps_per_second
    
    # Lower is better - how much time to process one step
    latency = avg_step_time
    
    # Calculate a composite score (higher is better)
    # This balances throughput with GPU efficiency
    composite_score = None
    if avg_gpu_util is not None:
        # Score that rewards higher throughput and higher GPU utilization
        # while penalizing excessive memory usage
        composite_score = (throughput * avg_gpu_util) / max(5, avg_gpu_mem)
    
    # Gather results
    result = {
        'batch_size': batch_size,
        'steps_per_epoch': steps_per_epoch,
        'num_threads': num_threads if num_threads is not None else torch.get_num_threads(),
        'avg_epoch_time': avg_epoch_time,
        'avg_step_time': avg_step_time,
        'avg_loss': float(avg_loss),
        'steps_per_second': steps_per_second,  # New metric
        'samples_per_second': samples_per_second,  # Keep for backward compatibility
        'gpu_utilization': avg_gpu_util,
        'gpu_memory_percent': avg_gpu_mem,
        'cpu_utilization': avg_cpu_util,
        'ram_utilization': avg_ram_util,
        'throughput': throughput,  # Now represents steps/sec instead of samples/sec
        'latency': latency,  # Now represents time per step instead of time per sample
        'composite_score': composite_score,
        'ssim': float(ssim_score),
    }
    
    # Restore previous thread setting if we changed it
    if previous_num_threads is not None:
        torch.set_num_threads(previous_num_threads)
    
    return result

def main():
    """Main benchmark function."""
    args = parse_args()
    
    # Import the correct model based on color flag
    if args.full_color:
        from model_color import ImageGeneratorColor
        ModelClass = ImageGeneratorColor
        print("Benchmarking with full color (RGB) model.")
    else:
        from model import Generator  # Changed from ImageGenerator
        ModelClass = Generator       # Changed from ImageGenerator
        print("Benchmarking with grayscale model.")
    
    # Create output directory
    if args.resume:
        # Find the most recent benchmark_* directory in output_dir
        if os.path.exists(args.output_dir):
            subdirs = [d for d in os.listdir(args.output_dir) if d.startswith('benchmark_')]
            if subdirs:
                subdirs.sort(reverse=True)
                output_dir = os.path.join(args.output_dir, subdirs[0])
                print(f"Resuming in most recent output directory: {output_dir}")
            else:
                # No previous runs, create a new one
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join(args.output_dir, f"benchmark_{timestamp}")
                os.makedirs(output_dir, exist_ok=True)
        else:
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(args.output_dir, f"benchmark_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, f"benchmark_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Benchmark results will be saved to: {output_dir}")
    
    # Set up device
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print("Using CPU for benchmarking")
    
    # Load reference image
    reference_path = os.path.join(os.getcwd(), args.reference)
    print(f"Processing reference image: {args.reference}")
    
    # Process the reference image
    processed_img, original_dimensions, _ = preprocess_reference_image(
        reference_path, 
        output_dir, 
        size=(args.image_size, args.image_size),
        show_preview=False
    )
    
    print(f"Reference image loaded with dimensions: {processed_img.shape}")
    
    # Generate batch sizes to test
    if args.batch_step:
        batch_sizes = list(range(args.min_batch, args.max_batch + 1, args.batch_step))
    else:
        # Use exponential scale: 1, 2, 4, 8, 16, 32, 64, 128...
        batch_sizes = [2**i for i in range(int(np.log2(args.min_batch)), int(np.log2(args.max_batch))+1)]
    
    # Generate steps per epoch to test
    if args.steps_step:
        steps_list = list(range(args.min_steps, args.max_steps + 1, args.steps_step))
    else:
        # Use logarithmic scale for steps too
        steps_list = []
        i = args.min_steps
        while i <= args.max_steps:
            steps_list.append(i)
            i = i * 2 if i < 8 else i + 8    # Start with a quick check to find the maximum batch size
    print("Finding maximum batch size that fits in GPU memory...")
    max_working_batch = 0
    
    # Start from smallest batch size and work up instead of starting from the largest
    for test_batch in sorted(batch_sizes):
        try:
            print(f"Testing batch size {test_batch}...")
            # Try to create a model and run a single step
            generator = ModelClass(
                image_size=processed_img.shape[:2],
                latent_dim=args.latent_dim,
                device=device,
                mixed_precision=args.mixed_precision
            )
            
            latent_vector = np.random.normal(0, 1, (test_batch, args.latent_dim))
            _ = generator.train_step(processed_img, 0, latent_vector)
            
            # If we made it here, this batch size works
            max_working_batch = test_batch
            print(f"Working batch size found: {max_working_batch}")
            
            # Clean up
            del generator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Continue testing larger sizes to find the maximum
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"Batch size {test_batch} causes CUDA out of memory error")
                # Clean up and don't test any larger sizes
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Break the loop as we've found our limit
                break            
            else:
                # If it's another error, re-raise it
                raise
    
    if max_working_batch == 0:
        print("Could not find a working batch size. Trying with minimal batch size of 1...")
        # Add a minimal batch size to attempt the benchmark
        max_working_batch = 1
        # If that still doesn't work, we'll exit later

    # Filter batch sizes to only those that work
    working_batch_sizes = [b for b in batch_sizes if b <= max_working_batch]
    
    # Ensure we have at least one working batch size
    if not working_batch_sizes:
        working_batch_sizes = [max_working_batch]
    
    # For efficiency, use a smaller set of batch sizes that are well-distributed
    if len(working_batch_sizes) > 5:
        sampled_indices = np.linspace(0, len(working_batch_sizes)-1, 5, dtype=int)
        working_batch_sizes = [working_batch_sizes[i] for i in sampled_indices]
    
    print(f"Testing batch sizes: {working_batch_sizes}")
    print(f"Testing steps per epoch values: {steps_list}")

    # Store benchmark results
    progress_path = os.path.join(output_dir, 'benchmark_progress.json')
    completed_configs = set()
    results = []
    if args.resume and os.path.exists(progress_path):
        print(f"Resuming from previous progress at {progress_path}")
        with open(progress_path, 'r') as f:
            progress_data = json.load(f)
            results = progress_data.get('results', [])
            for entry in results:
                key = (entry.get('batch_size'), entry.get('steps_per_epoch'), entry.get('num_threads'))
                completed_configs.add(key)

    # Thread sweep logic
    if args.test_threads:
        # Generate thread counts to test
        if args.thread_step:
            thread_counts = list(range(args.min_threads, args.max_threads + 1, args.thread_step))
        else:
            # Use exponential scale: 1, 2, 4, 8, ...
            thread_counts = [2**i for i in range(int(np.log2(args.min_threads)), int(np.log2(args.max_threads))+1)]
        print(f"Testing thread counts: {thread_counts}")
        total_configs = len(thread_counts) * len(working_batch_sizes) * len(steps_list)
        print(f"Running {total_configs} benchmark configurations (thread sweep)...")
        for num_threads in thread_counts:
            for batch_size in working_batch_sizes:
                for steps in steps_list:
                    config_key = (batch_size, steps, num_threads)
                    if config_key in completed_configs:
                        print(f"Skipping already completed config: batch_size={batch_size}, steps={steps}, threads={num_threads}")
                        continue
                    try:
                        result = run_benchmark(batch_size, steps, processed_img, args, device, num_threads=num_threads, ModelClass=ModelClass)
                        results.append(result)
                        with open(progress_path, 'w') as f:
                            json.dump({'results': results}, f, indent=2)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"Error benchmarking batch_size={batch_size}, steps={steps}, threads={num_threads}: {e}")
                        results.append({
                            'batch_size': batch_size,
                            'steps_per_epoch': steps,
                            'num_threads': num_threads,
                            'error': str(e)
                        })
                        with open(progress_path, 'w') as f:
                            json.dump({'results': results}, f, indent=2)
    else:
        # Run benchmarks for each configuration (no thread sweep)
        total_configs = len(working_batch_sizes) * len(steps_list)
        print(f"Running {total_configs} benchmark configurations...")
        for batch_size in working_batch_sizes:
            for steps in steps_list:
                config_key = (batch_size, steps, None)
                if config_key in completed_configs:
                    print(f"Skipping already completed config: batch_size={batch_size}, steps={steps}, threads=default")
                    continue
                try:
                    # Run the benchmark
                    result = run_benchmark(batch_size, steps, processed_img, args, device, ModelClass=ModelClass)
                    results.append(result)
                    with open(progress_path, 'w') as f:
                        json.dump({'results': results}, f, indent=2)
                    # Clean up GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error benchmarking batch_size={batch_size}, steps={steps}: {e}")
                    # Store failed result
                    results.append({
                        'batch_size': batch_size,
                        'steps_per_epoch': steps,
                        'error': str(e)
                    })
                    with open(progress_path, 'w') as f:
                        json.dump({'results': results}, f, indent=2)
    
    # Convert results to a pandas DataFrame
    df = pd.DataFrame(results)
    
    # Save raw results
    df.to_csv(os.path.join(output_dir, "benchmark_results.csv"), index=False)
    
    # Save results as JSON
    with open(os.path.join(output_dir, "benchmark_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Find the best configurations based on different metrics
    if 'error' not in df.columns or not df['error'].any():
        # Best throughput (steps/sec)
        best_throughput_idx = df['throughput'].idxmax()
        best_throughput = df.iloc[best_throughput_idx]
        
        # Best GPU utilization
        if 'gpu_utilization' in df and not df['gpu_utilization'].isna().all():
            best_util_idx = df['gpu_utilization'].idxmax()
            best_util = df.iloc[best_util_idx]
        else:
            best_util = None
            
        # Best composite score (if available)
        best_composite = None
        if 'composite_score' in df and not df['composite_score'].isna().all():
            best_composite_idx = df['composite_score'].idxmax()
            best_composite = df.iloc[best_composite_idx]
        
        # Generate recommendations
        recommendations = {
            'best_throughput': {
                'batch_size': int(best_throughput['batch_size']),
                'steps_per_epoch': int(best_throughput['steps_per_epoch']),
                'throughput': float(best_throughput['throughput']),
                'gpu_utilization': float(best_throughput['gpu_utilization']) if best_throughput['gpu_utilization'] is not None else None,
                'gpu_memory_percent': float(best_throughput['gpu_memory_percent']) if best_throughput['gpu_memory_percent'] is not None else None
            }
        }
        
        if best_util is not None:
            recommendations['best_gpu_utilization'] = {
                'batch_size': int(best_util['batch_size']),
                'steps_per_epoch': int(best_util['steps_per_epoch']),
                'throughput': float(best_util['throughput']),
                'gpu_utilization': float(best_util['gpu_utilization']) if best_util['gpu_utilization'] is not None else None,
                'gpu_memory_percent': float(best_util['gpu_memory_percent']) if best_util['gpu_memory_percent'] is not None else None
            }
            
        if best_composite is not None:
            recommendations['best_composite_score'] = {
                'batch_size': int(best_composite['batch_size']),
                'steps_per_epoch': int(best_composite['steps_per_epoch']),
                'throughput': float(best_composite['throughput']),
                'gpu_utilization': float(best_composite['gpu_utilization']) if best_composite['gpu_utilization'] is not None else None,
                'gpu_memory_percent': float(best_composite['gpu_memory_percent']) if best_composite['gpu_memory_percent'] is not None else None,
                'composite_score': float(best_composite['composite_score'])
            }
        
        # Save recommendations
        with open(os.path.join(output_dir, "recommendations.json"), 'w') as f:
            json.dump(recommendations, f, indent=2)
            
        # Create a markdown summary
        with open(os.path.join(output_dir, "benchmark_summary.md"), 'w') as f:
            f.write("# ML Learning Benchmark Results\n\n")
            f.write(f"Run on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # System info
            f.write("## System Information\n\n")
            if device.type == "cuda":
                f.write(f"- GPU: {torch.cuda.get_device_name(0)}\n")
            else:
                f.write("- CPU Only\n")
                
            # Reference image
            f.write(f"- Reference Image: {args.reference}\n")
            f.write(f"- Image Size: {args.image_size}x{args.image_size}\n\n")
            
            # Best configurations
            f.write("## Recommendations\n\n")
            
            f.write("### Best Throughput Configuration\n\n")
            f.write(f"- Batch Size: **{recommendations['best_throughput']['batch_size']}**\n")
            f.write(f"- Steps Per Epoch: **{recommendations['best_throughput']['steps_per_epoch']}**\n")
            f.write(f"- Steps Per Second: {recommendations['best_throughput']['throughput']:.2f}\n")
            if recommendations['best_throughput']['gpu_utilization'] is not None:
                f.write(f"- GPU Utilization: {recommendations['best_throughput']['gpu_utilization']:.1f}%\n")
            if recommendations['best_throughput']['gpu_memory_percent'] is not None:
                f.write(f"- GPU Memory Usage: {recommendations['best_throughput']['gpu_memory_percent']:.1f}%\n")
            f.write("\n")
            
            if 'best_gpu_utilization' in recommendations:
                f.write("### Best GPU Utilization Configuration\n\n")
                f.write(f"- Batch Size: **{recommendations['best_gpu_utilization']['batch_size']}**\n")
                f.write(f"- Steps Per Epoch: **{recommendations['best_gpu_utilization']['steps_per_epoch']}**\n")
                f.write(f"- Steps Per Second: {recommendations['best_gpu_utilization']['throughput']:.2f}\n")
                f.write(f"- GPU Utilization: {recommendations['best_gpu_utilization']['gpu_utilization']:.1f}%\n")
                if recommendations['best_gpu_utilization']['gpu_memory_percent'] is not None:
                    f.write(f"- GPU Memory Usage: {recommendations['best_gpu_utilization']['gpu_memory_percent']:.1f}%\n")
                f.write("\n")
                
            if 'best_composite_score' in recommendations:
                f.write("### Best Balanced Configuration\n\n")
                f.write(f"- Batch Size: **{recommendations['best_composite_score']['batch_size']}**\n")
                f.write(f"- Steps Per Epoch: **{recommendations['best_composite_score']['steps_per_epoch']}**\n")
                f.write(f"- Steps Per Second: {recommendations['best_composite_score']['throughput']:.2f}\n")
                f.write(f"- GPU Utilization: {recommendations['best_composite_score']['gpu_utilization']:.1f}%\n")
                if recommendations['best_composite_score']['gpu_memory_percent'] is not None:
                    f.write(f"- GPU Memory Usage: {recommendations['best_composite_score']['gpu_memory_percent']:.1f}%\n")
                f.write(f"- Composite Score: {recommendations['best_composite_score']['composite_score']:.2f}\n\n")
                
            f.write("## How to Use These Results\n\n")
            f.write("To use the recommended configuration, run your training with these parameters:\n\n")
            
            # Use the balanced configuration if available, otherwise use throughput
            if 'best_composite_score' in recommendations:
                rec = recommendations['best_composite_score']
            else:
                rec = recommendations['best_throughput']
                
            f.write("```bash\n")
            f.write(f"python src/train.py --use-gpu --batch-size {rec['batch_size']} --steps-per-epoch {rec['steps_per_epoch']} --image-size {args.image_size}\n")
            f.write("```\n\n")
            
            # Runtime comparison
            f.write("## Full Results\n\n")
            f.write("See `benchmark_results.csv` for the complete benchmark data.\n")
        
        # Generate visualization plots
        try:
            # Create throughput heatmap
            plt.figure(figsize=(10, 8))
            pivot_df = df.pivot(index="batch_size", columns="steps_per_epoch", values="throughput")
            plt.pcolormesh(pivot_df.columns, pivot_df.index, pivot_df.values, cmap='viridis')
            plt.colorbar(label="Steps per second")
            plt.xlabel("Steps per Epoch")
            plt.ylabel("Batch Size")
            plt.title("Training Throughput (steps/sec)")
            plt.savefig(os.path.join(output_dir, "throughput_heatmap.png"), dpi=120)
            plt.close()
            
            # Create GPU utilization plot
            if 'gpu_utilization' in df and not df['gpu_utilization'].isna().all():
                plt.figure(figsize=(10, 8))
                pivot_df = df.pivot(index="batch_size", columns="steps_per_epoch", values="gpu_utilization")
                plt.pcolormesh(pivot_df.columns, pivot_df.index, pivot_df.values, cmap='inferno')
                plt.colorbar(label="GPU Utilization %")
                plt.xlabel("Steps per Epoch")
                plt.ylabel("Batch Size")
                plt.title("GPU Utilization")
                plt.savefig(os.path.join(output_dir, "gpu_utilization_heatmap.png"), dpi=120)
                plt.close()
                
            # Create composite score plot
            if 'composite_score' in df and not df['composite_score'].isna().all():
                plt.figure(figsize=(10, 8))
                pivot_df = df.pivot(index="batch_size", columns="steps_per_epoch", values="composite_score")
                plt.pcolormesh(pivot_df.columns, pivot_df.index, pivot_df.values, cmap='plasma')
                plt.colorbar(label="Composite Score")
                plt.xlabel("Steps per Epoch")
                plt.ylabel("Batch Size")
                plt.title("Balanced Performance Score (higher is better)")
                plt.savefig(os.path.join(output_dir, "composite_score_heatmap.png"), dpi=120)
                plt.close()
                
            # Batch size vs metrics plot
            plt.figure(figsize=(10, 6))
            for steps in steps_list:
                subset = df[df['steps_per_epoch'] == steps]
                if not subset.empty:
                    plt.plot(subset['batch_size'], subset['throughput'], 'o-', label=f'Steps={steps}')
            plt.xlabel('Batch Size')
            plt.ylabel('Steps per Second')
            plt.title('Effect of Batch Size on Training Throughput')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(output_dir, "batch_size_throughput.png"), dpi=120)
            plt.close()
                
        except Exception as e:
            print(f"Error creating visualization plots: {e}")
            
    # Print recommendations
    print("\n" + "="*60)
    print("BENCHMARK RESULTS & RECOMMENDATIONS")
    print("="*60)
    
    if 'error' not in df.columns or not df['error'].any():
        print("\nBest throughput configuration:")
        print(f"  Batch size: {recommendations['best_throughput']['batch_size']}")
        print(f"  Steps per epoch: {recommendations['best_throughput']['steps_per_epoch']}")
        print(f"  Steps per second: {recommendations['best_throughput']['throughput']:.2f}")
        
        if 'best_composite_score' in recommendations:
            print("\nBest balanced configuration (recommended):")
            print(f"  Batch size: {recommendations['best_composite_score']['batch_size']}")
            print(f"  Steps per epoch: {recommendations['best_composite_score']['steps_per_epoch']}")
            print(f"  Steps per second: {recommendations['best_composite_score']['throughput']:.2f}")
            print(f"  GPU utilization: {recommendations['best_composite_score']['gpu_utilization']:.1f}%")
        
        print("\nTo use the recommended configuration, run:")
        if 'best_composite_score' in recommendations:
            rec = recommendations['best_composite_score']
        else:
            rec = recommendations['best_throughput']
            
        print(f"  python src/train.py --use-gpu --batch-size {rec['batch_size']} --steps-per-epoch {rec['steps_per_epoch']} --image-size {args.image_size}")
    
    print(f"\nDetailed results saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()