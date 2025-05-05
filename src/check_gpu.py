"""
GPU Diagnostic Script

This script checks if CUDA is available and provides detailed information
about the GPU configuration for PyTorch.
"""

import torch
import sys
import platform

def print_section(title):
    """Print a section header to make output more readable."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def main():
    print_section("System Information")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Operating system: {platform.system()} {platform.release()}")
    
    print_section("CUDA Availability")
    if torch.cuda.is_available():
        print("✅ CUDA is AVAILABLE")
        
        # Check CUDA version
        print(f"CUDA version: {torch.version.cuda}")
        
        # Get device count
        device_count = torch.cuda.device_count()
        print(f"Number of available CUDA devices: {device_count}")
        
        # Display information for each device
        for i in range(device_count):
            print_section(f"GPU #{i} Information")
            print(f"Device name: {torch.cuda.get_device_name(i)}")
            print(f"Device capability: {torch.cuda.get_device_capability(i)}")
            
            # Get memory information
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # Convert to GB
            print(f"Total memory: {total_mem:.2f} GB")
            
            # Test creating a tensor on the GPU
            try:
                print("Testing tensor creation on GPU...")
                x = torch.ones(1000, 1000).to(f"cuda:{i}")
                y = x + x
                print("✅ Successfully created and computed tensor operations on GPU")
                
                # Clean up
                del x, y
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"❌ Error creating tensor on GPU: {e}")
        
        # Check if cuDNN is available and its version
        if hasattr(torch.backends, 'cudnn'):
            print_section("cuDNN Information")
            if torch.backends.cudnn.is_available():
                print(f"cuDNN is available: {torch.backends.cudnn.is_available()}")
                print(f"cuDNN version: {torch.backends.cudnn.version()}")
                print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
            else:
                print("cuDNN is NOT available")
    else:
        print("❌ CUDA is NOT available")
        print("This means PyTorch cannot use your GPU for computation.")
        
        # Check if CUDA is installed but not available to PyTorch
        try:
            import subprocess
            nvidia_info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
            print("\nNOTE: NVIDIA driver appears to be installed (nvidia-smi works)," 
                  "but PyTorch can't access it.")
            print("This might be due to:")
            print("1. PyTorch being installed without CUDA support")
            print("2. Incompatible CUDA versions")
            print("3. Environment/path issues")
        except:
            print("\nNVIDIA driver appears to be NOT installed or not working properly")
            print("Ensure you have a compatible GPU and have installed the NVIDIA drivers")
    
    print_section("Testing Memory Allocation")
    # Try to create tensors of increasing size
    sizes = [
        (1000, 1000),      # ~4MB
        (5000, 5000),      # ~100MB
        (10000, 10000),    # ~400MB
        (20000, 20000),    # ~1.6GB (this might fail on GPUs with limited memory)
    ]
    
    for size in sizes:
        try:
            if torch.cuda.is_available():
                print(f"Testing creation of tensor with size {size}...")
                x = torch.ones(size).to('cuda')
                y = x + x  # Do a simple operation
                # Calculate memory used
                mem_used_mb = x.element_size() * x.nelement() / (1024 * 1024)
                print(f"✅ Success: Created tensor using {mem_used_mb:.2f} MB of GPU memory")
                del x, y  # Free memory
                torch.cuda.empty_cache()
            else:
                print("Skipping memory test (CUDA not available)")
                break
        except RuntimeError as e:
            print(f"❌ Failed to create tensor of size {size}: {e}")
            break
    
    print_section("Summary")
    if torch.cuda.is_available():
        print("Your GPU appears to be configured correctly for PyTorch.")
        print("If you're still experiencing low GPU utilization in your model, it may be due to:")
        print("1. Inefficient data loading or processing")
        print("2. Small batch sizes")
        print("3. Model architecture limitations")
        print("4. CPU bottlenecks in your training pipeline")
    else:
        print("PyTorch cannot access your GPU. Please check your installation.")
        print("To install PyTorch with GPU support, visit: https://pytorch.org/get-started/locally/")


if __name__ == "__main__":
    main()