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
      # Use CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Using device: {device}")
    
    # Load reference image and preprocess
    img, _, _ = preprocess_reference_image(
        args.reference, ".", size=(args.image_size, args.image_size), show_preview=False
    )
    
    # Convert to tensor and ensure proper shape
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img, dtype=torch.float32)
    
    if img.dim() == 3 and img.shape[2] == 1:
        img = img.permute(2, 0, 1).unsqueeze(0)
    elif img.dim() == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    
    # Move image to device
    img = img.to(device)
    
    # Import the model
    from model import ImageGenerator
      log(f"Creating model with image size: {img.shape}")
    
    try:
        # Create generator with explicit image size
        generator = ImageGenerator(
            image_size=(img.shape[2], img.shape[3]),
            latent_dim=args.latent_dim,
            device=device
        )
        
        log("Model created successfully")
        log(f"Generator device: {next(generator.generator.parameters()).device}")
        
        # Check and explicitly move all parameters to device
        log("Ensuring ALL parameters are on the correct device...")
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
        loss = generator.train_step(img, 0, latent_vector)
        
        log(f"SUCCESS! Training step completed with loss: {loss}")
        log("Your model works correctly with the GPU.")
    
    except Exception as e:
        log(f"ERROR: {e}")
        log("Let's try to debug the model...")
        
        try:
            # Create model on CPU first
            log("Creating model on CPU...")
            cpu_device = torch.device('cpu')
            generator = ImageGenerator(
                image_size=(img.shape[2], img.shape[3]),
                latent_dim=args.latent_dim,
                device=cpu_device
            )
            
            # Manually move everything to GPU
            log("Manually moving everything to GPU...")
            generator.device = device
            
            # Create CPU versions for testing
            cpu_img = img.cpu()
            cpu_latent = torch.randn(args.batch_size, args.latent_dim)
            
            # Try invoking the forward method directly on CPU
            log("Testing generator forward method on CPU...")
            with torch.no_grad():
                generator.generator.eval()
                condition = torch.zeros(args.batch_size, dtype=torch.long)
                cpu_output = generator.generator(cpu_latent, condition)
                log(f"CPU forward pass worked! Output shape: {cpu_output.shape}")
            
            # Move the generator model to GPU manually
            log("Moving generator to GPU manually...")
            for name, module in generator.generator.named_children():
                module.to(device)
                log(f"Moved {name} to {device}")
            
            # Check if any parameters are still on CPU
            cpu_params = []
            for name, param in generator.generator.named_parameters():
                if param.device.type == 'cpu':
                    cpu_params.append(name)
            
            if cpu_params:
                log(f"The following parameters are still on CPU: {cpu_params}")
            else:
                log("All parameters successfully moved to GPU")
                
                # Try a GPU forward pass
                log("Testing forward pass on GPU...")
                gpu_latent = torch.randn(args.batch_size, args.latent_dim, device=device)
                condition = torch.zeros(args.batch_size, dtype=torch.long, device=device)
                gpu_output = generator.generator(gpu_latent, condition)
                log(f"GPU forward pass worked! Output shape: {gpu_output.shape}")
                
        except Exception as sub_e:
            log(f"Debugging error: {sub_e}")

if __name__ == "__main__":
    main()
