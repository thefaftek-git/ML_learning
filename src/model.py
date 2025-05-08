"""
Image Generator Model

This module defines a machine learning model that generates 2D wireframe images.
The model is designed to learn from scratch without pre-existing training data.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Generator(nn.Module):
    """
    A PyTorch neural network model for generating 2D wireframe images.
    
    This implementation uses transposed convolutional layers to generate
    images from a latent vector input and conditional embedding.
    """
    
    def __init__(self, latent_dim=100, channels=1, output_size=256, num_conditions=3, tag_dim=0):
        """
        Initialize the generator architecture.
        
        Args:
            latent_dim: Dimension of the latent space input vector
            channels: Number of channels in the output image (1 for grayscale)
            output_size: Size of the output image (width/height)
            num_conditions: Number of different conditional outputs the model can generate
            tag_dim: Dimension for tag embeddings if using annotations
        """
        super(Generator, self).__init__()
        
        # Store target output size
        self.target_height = output_size[0] if isinstance(output_size, tuple) else output_size
        self.target_width = output_size[1] if isinstance(output_size, tuple) else output_size
        self.num_conditions = num_conditions
        self.has_tag_support = tag_dim > 0
        self.tag_dim = tag_dim
        
        # Determine the number of upsampling blocks needed
        self.base_size = 8  # Starting size for feature maps
        self.num_upsample = 0
        
        # Calculate how many upsampling operations we need to get close to target size
        # Each upsampling doubles the dimensions
        max_dimension = max(self.target_height, self.target_width)
        temp_size = self.base_size
        while temp_size < max_dimension:
            temp_size *= 2
            self.num_upsample += 1
        
        # The actual output size after upsampling will be base_size * 2^num_upsample
        self.generated_size = self.base_size * (2 ** self.num_upsample)
        
        print(f"Creating generator with {self.num_upsample} upsampling blocks")
        print(f"Target size: {self.target_height}x{self.target_width}, Generated size before resizing: {self.generated_size}x{self.generated_size}")
        
        # Embedding layer for condition input
        # Creates a learnable embedding for each condition (image type)
        self.condition_embedding = nn.Embedding(num_conditions, 32)
        
        # Embedding layer for tags if tag support is enabled
        if self.has_tag_support:
            print(f"Enabling tag support with dimension {tag_dim}")
            self.tag_embedding_size = 16
            self.tag_projection = nn.Linear(tag_dim, self.tag_embedding_size)
            
            # Initial dense layer to expand from latent space + condition + tags
            self.fc = nn.Linear(latent_dim + 32 + self.tag_embedding_size, self.base_size * self.base_size * 256)
        else:
            # Initial dense layer to expand from latent space + condition
            self.fc = nn.Linear(latent_dim + 32, self.base_size * self.base_size * 256)
        
        # Create a sequence of upsampling blocks
        layers = []
        in_channels = 256
        
        for i in range(self.num_upsample - 1):
            out_channels = max(16, in_channels // 2)  # Halve the channels but keep minimum 16
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels
        
        # Final upsampling block to output layer
        layers.append(nn.ConvTranspose2d(in_channels, channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Tanh())  # Output in range [-1, 1]
        
        self.conv_layers = nn.Sequential(*layers)
    
    def forward(self, z, condition=None, tag_tensor=None):
        """
        Forward pass through the generator.
        
        Args:
            z: Latent vector input of shape (batch_size, latent_dim)
            condition: Integer tensor specifying which image type to generate
                       (batch_size,), values from 0 to num_conditions-1
            tag_tensor: Optional tensor of tag embeddings (batch_size, tag_dim)
            
        Returns:
            Generated images of shape (batch_size, channels, height, width)
        """
        batch_size = z.size(0)
        
        # If no condition provided, default to first condition (0)
        if condition is None:
            condition = torch.zeros(batch_size, dtype=torch.long, device=z.device)
        
        # Convert condition to embedding vector
        condition_vector = self.condition_embedding(condition)
        
        if self.has_tag_support and tag_tensor is not None:
            # Process tag tensor through embedding layer
            tag_embedding = self.tag_projection(tag_tensor)
            
            # Concatenate latent vector, condition embedding, and tag embedding
            combined_input = torch.cat([z, condition_vector, tag_embedding], dim=1)
        else:
            # Concatenate latent vector with condition embedding
            combined_input = torch.cat([z, condition_vector], dim=1)
        
        # Project and reshape combined input
        x = self.fc(combined_input)
        x = x.view(x.shape[0], 256, self.base_size, self.base_size)
        
        # Apply transposed convolutions
        x = self.conv_layers(x)
        
        # Resize to target dimensions if they don't match what's generated
        if (x.shape[2] != self.target_height) or (x.shape[3] != self.target_width):
            x = F.interpolate(
                x, 
                size=(self.target_height, self.target_width), 
                mode='bilinear', 
                align_corners=False
            )
        
        return x

class ImageGenerator:
    """
    A neural network model for generating 2D wireframe images.
    
    This model uses a conditional generative approach to create images that
    progressively get closer to multiple target images through training.
    The model can be trained to generate different images based on a condition input.
    """
    
    def __init__(self, image_size=(128, 128), latent_dim=100, num_conditions=3, 
                 device=None, mixed_precision=False, tag_dim=0):
        """
        Initialize the image generator model.
        
        Args:
            image_size: Tuple of (height, width) for the output image size
            latent_dim: Dimension of the latent space for the generator
            num_conditions: Number of different conditional outputs the model can generate
            device: PyTorch device to use (cuda or cpu)
            mixed_precision: Whether to use mixed precision training (FP16)
            tag_dim: Dimension for tag embeddings if using annotations (0 to disable)
        """
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_conditions = num_conditions
        self.mixed_precision = mixed_precision
        self.tag_dim = tag_dim
        self.has_annotation_support = tag_dim > 0
        
        # Initialize device (use provided or auto-detect GPU if available)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            # If device is explicitly set to 'cuda', verify availability
            if device == torch.device("cuda") and not torch.cuda.is_available():
                print("WARNING: CUDA requested but not available. Falling back to CPU.")
                self.device = torch.device("cpu")
            else:
                self.device = device
            
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA current device: {torch.cuda.current_device()}")
            
        # Initialize generator model directly with the target dimensions and condition support
        self.generator = Generator(
            latent_dim=latent_dim, 
            output_size=image_size,
            num_conditions=num_conditions,
            tag_dim=tag_dim
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=0.0002,
            betas=(0.5, 0.999)
        )
        
        # Set up mixed precision training if requested and using CUDA
        if mixed_precision and self.device.type == 'cuda':
            # Import autocast for mixed precision
            from torch.cuda.amp import autocast, GradScaler
            self.scaler = GradScaler()
            self.autocast = autocast
            print("Mixed precision training enabled")
        else:
            self.mixed_precision = False
    
    def generate_image(self, latent_vector=None, condition_id=0, tag_tensor=None):
        """
        Generate an image using the generator model with a specific condition.
        
        Args:
            latent_vector: Input vector for the generator. If None, a random vector is used.
            condition_id: Integer specifying which image type to generate (0 to num_conditions-1)
            tag_tensor: Optional tensor of tag embeddings (for annotation-based generation)
            
        Returns:
            Generated image as a numpy array
        """
        self.generator.eval()  # Set to evaluation mode
        
        with torch.no_grad():
            # Create latent vector if not provided
            if latent_vector is None:
                latent_vector = np.random.normal(0, 1, (1, self.latent_dim))
                
            # Convert to PyTorch tensor
            if isinstance(latent_vector, np.ndarray):
                latent_vector = torch.from_numpy(latent_vector).float().to(self.device)
            
            # Convert condition to tensor
            condition = torch.tensor([condition_id], dtype=torch.long, device=self.device)
            
            # Process tag tensor if provided
            if tag_tensor is not None and self.has_annotation_support:
                if isinstance(tag_tensor, np.ndarray):
                    tag_tensor = torch.from_numpy(tag_tensor).float().to(self.device)
            else:
                tag_tensor = None
                
            # Generate the image with the condition and tags
            generated_image = self.generator(latent_vector, condition, tag_tensor)
            
            # Move to CPU and convert to numpy array
            image_array = generated_image.cpu().numpy()
            
            # Reshape to expected format (height, width, channels)
            image_array = image_array[0].transpose(1, 2, 0)
            
            # Rescale from [-1, 1] to [0, 1]
            image_array = (image_array + 1) / 2.0
            
            return image_array