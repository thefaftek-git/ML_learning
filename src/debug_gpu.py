"""
ML System Benchmarking Script - GPU Memory Testing Version

This is a simplified version of the benchmarking script that focuses on 
properly handling GPU device compatibility. It attempts to test if a model
can properly run on your GPU with the specified parameters.
"""

import torch
import torch.optim as optim
import argparse
import numpy as np
import sys
import os
from utils import preprocess_reference_image

# Set up logging to a file
log_file = "gpu_debug_log.txt"
with open(log_file, "w") as f:
    f.write("Starting GPU debugging\n")

def log(message):
    """Write message to both stdout and the log file"""
    print(message)
    with open(log_file, "a") as f:
        f.write(f"{message}\n")

def parse_args():
    parser = argparse.ArgumentParser(description='Test GPU compatibility')
    parser.add_argument('--reference', type=str, default='reference.png',
                      help='Path to a target reference image')
    parser.add_argument('--image-size', type=int, default=16,
                      help='Size for the output image (smaller = less memory)')
    parser.add_argument('--latent-dim', type=int, default=32,
                      help='Dimension of the latent space (smaller = less memory)')
    parser.add_argument('--batch-size', type=int, default=1,
                      help='Batch size to test')
    parser.add_argument('--gpu-id', type=int, default=None,
                        help='Specify the ID of the GPU to use for debugging (e.g., 0, 1, 2). If not specified, defaults to GPU 0 if CUDA is available.')
    return parser.parse_args()

def move_model_to_device(model, device):
    """Explicitly move all model parameters to the specified device"""
    for param in model.parameters():
        if param.device != device:
            # Create a new copy of the parameter on the right device
            param.data = param.data.to(device)
    return model

def main():
    args = parse_args()

    # Updated GPU device selection logic
    if torch.cuda.is_available():
        if args.gpu_id is not None:
            if 0 <= args.gpu_id < torch.cuda.device_count():
                device = torch.device(f"cuda:{args.gpu_id}")
                log(f"Using specified GPU ID: {args.gpu_id} ({torch.cuda.get_device_name(args.gpu_id)})")
            else:
                log(f"Error: Invalid GPU ID {args.gpu_id}. GPU IDs must be between 0 and {torch.cuda.device_count() - 1}.")
                log("Available GPUs are:")
                for i in range(torch.cuda.device_count()):
                    log(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                log("Exiting due to invalid GPU ID.")
                sys.exit(1) # Exit the script
        else:
            device = torch.device("cuda:0") # Default to GPU 0
            log("No specific GPU ID provided, defaulting to GPU 0 for debugging.")
    else:
        device = torch.device("cpu")
        log("CUDA is not available. Using CPU for debugging.")
    
    log(f"Selected device: {device}")

    # Load reference image and preprocess
    img, _, _ = preprocess_reference_image(
        args.reference, ".", size=(args.image_size, args.image_size), show_preview=False
    )
    
    # Convert to tensor and ensure proper shape
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img, dtype=torch.float32)
    
    if img.dim() == 3 and img.shape[2] == 1: # Assuming channel-last from preprocess_reference_image
        img = img.permute(2, 0, 1).unsqueeze(0) # HWC to CHW then NCHW
    elif img.dim() == 2: # Grayscale H W
        img = img.unsqueeze(0).unsqueeze(0) # N C H W (1, 1, H, W)
    elif img.dim() == 3 and img.shape[0] == 3: # Assuming CHW from elsewhere
        img = img.unsqueeze(0) # NCHW
    elif img.dim() !=4 :
        log(f"Unexpected image dimensions: {img.shape}. Attempting to proceed but may fail.")
        # Fallback: try to force into NCHW if it's a plausible image size
        if img.ndim == 3: # e.g. 3, H, W
             img = img.unsqueeze(0)


    # Move image to device
    img = img.to(device)
    log(f"Reference image tensor moved to {img.device} with shape {img.shape}")

    # Import the model
    from model import ImageGenerator
    log(f"Creating model for image size: {args.image_size}x{args.image_size}")
    
    try:
        # Create generator with explicit image size
        generator = ImageGenerator(
            image_size=(args.image_size, args.image_size), # Pass as tuple
            latent_dim=args.latent_dim,
            device=device
        )
        
        log("Model created successfully")
        # Check a parameter's device to confirm
        if hasattr(generator, 'generator') and next(generator.generator.parameters(), None) is not None:
             log(f"Generator parameter device check: {next(generator.generator.parameters()).device}")
        else:
             log("Generator model appears empty or not initialized correctly.")
             raise ValueError("Generator model not properly initialized.")

        # Check and explicitly move all parameters to device (redundant if model init is correct, but good for debug)
        log("Ensuring ALL model parameters are on the correct device...")
        generator.generator = move_model_to_device(generator.generator, device)
        
        # Create new optimizer AFTER moving parameters
        generator.optimizer = optim.Adam(
            generator.generator.parameters(),
            lr=0.0002, 
            betas=(0.5, 0.999)
        )
        
        # Generate latent vector on device
        latent_vector = torch.randn(args.batch_size, args.latent_dim, device=device)
        log(f"Latent vector created on {latent_vector.device} with shape {latent_vector.shape}")
        
        # Try a training step
        log("Attempting a training step...")
        # Ensure the image is correctly batched if batch_size > 1
        if args.batch_size > 1 and img.shape[0] == 1:
            img_batch = img.repeat(args.batch_size, 1, 1, 1)
            log(f"Duplicated single reference image to batch size {args.batch_size}")
        else:
            img_batch = img

        loss = generator.train_step(img_batch, 0, latent_vector) # Pass condition_id as 0
        
        log(f"SUCCESS! Training step completed with loss: {loss}")
        log("Your model appears to work correctly with the selected GPU and parameters.")
    
    except Exception as e:
        log(f"ERROR during main execution: {e}")
        log(f"Exception type: {type(e).__name__}")
        import traceback
        log(f"Traceback:\n{traceback.format_exc()}")
        log("Attempting further debugging steps if possible...")
        
        try:
            # Create model on CPU first for comparison
            log("Creating model on CPU for comparison...")
            cpu_device = torch.device('cpu')
            # Ensure image is on CPU for this test
            cpu_img = img.cpu()
            if args.batch_size > 1 and cpu_img.shape[0] == 1:
                 cpu_img_batch = cpu_img.repeat(args.batch_size, 1, 1, 1)
            else:
                 cpu_img_batch = cpu_img

            cpu_generator = ImageGenerator(
                image_size=(args.image_size, args.image_size),
                latent_dim=args.latent_dim,
                device=cpu_device
            )
            log("CPU model created.")
            
            # Try invoking the forward method directly on CPU
            log("Testing generator forward method on CPU...")
            with torch.no_grad():
                cpu_generator.generator.eval()
                cpu_latent = torch.randn(args.batch_size, args.latent_dim, device=cpu_device)
                # Assuming condition_id 0, and condition handling is inside the model or not strictly needed for forward
                condition_tensor_cpu = torch.zeros(args.batch_size, dtype=torch.long, device=cpu_device)

                # Check if the model's forward takes condition
                import inspect
                sig = inspect.signature(cpu_generator.generator.forward)
                if 'condition' in sig.parameters:
                    cpu_output = cpu_generator.generator(cpu_latent, condition_tensor_cpu)
                else:
                    cpu_output = cpu_generator.generator(cpu_latent)
                log(f"CPU forward pass worked! Output shape: {cpu_output.shape}")
            
            # If the original error was on GPU, try to re-instantiate on GPU and move parts
            if device.type == 'cuda':
                log("Re-attempting model instantiation directly on GPU for detailed check...")
                gpu_generator_debug = ImageGenerator(
                    image_size=(args.image_size, args.image_size),
                    latent_dim=args.latent_dim,
                    device=device # Explicitly pass the target device
                )
                log("Model instantiated on GPU.")
                
                # Check parameters
                cpu_params = []
                gpu_params_count = 0
                for name, param in gpu_generator_debug.generator.named_parameters():
                    if param.device.type == 'cpu':
                        cpu_params.append(name)
                    elif param.device.type == 'cuda':
                        gpu_params_count +=1
                
                if cpu_params:
                    log(f"WARNING: The following parameters were found on CPU after GPU instantiation: {cpu_params}")
                else:
                    log(f"All {gpu_params_count} parameters appear to be on GPU after instantiation.")

                # Try a GPU forward pass
                log("Testing forward pass on GPU with re-instantiated model...")
                with torch.no_grad():
                    gpu_generator_debug.generator.eval()
                    gpu_latent_debug = torch.randn(args.batch_size, args.latent_dim, device=device)
                    condition_tensor_gpu = torch.zeros(args.batch_size, dtype=torch.long, device=device)
                    if 'condition' in sig.parameters: # Using sig from CPU model check
                        gpu_output_debug = gpu_generator_debug.generator(gpu_latent_debug, condition_tensor_gpu)
                    else:
                        gpu_output_debug = gpu_generator_debug.generator(gpu_latent_debug)
                    log(f"GPU forward pass with re-instantiated model worked! Output shape: {gpu_output_debug.shape}")
                
        except Exception as sub_e:
            log(f"ERROR during debugging attempt: {sub_e}")
            log(f"Sub-exception type: {type(sub_e).__name__}")
            log(f"Sub-Traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
