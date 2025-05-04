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
    images from a latent vector input.
    """
    
    def __init__(self, latent_dim=100, channels=1, output_size=256):
        """
        Initialize the generator architecture.
        
        Args:
            latent_dim: Dimension of the latent space input vector
            channels: Number of channels in the output image (1 for grayscale)
            output_size: Size of the output image (width/height)
        """
        super(Generator, self).__init__()
        
        # Store target output size
        self.target_height = output_size[0] if isinstance(output_size, tuple) else output_size
        self.target_width = output_size[1] if isinstance(output_size, tuple) else output_size
        
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
        
        # Initial dense layer to expand from latent space
        self.fc = nn.Linear(latent_dim, self.base_size * self.base_size * 256)
        
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
    
    def forward(self, z):
        """
        Forward pass through the generator.
        
        Args:
            z: Latent vector input of shape (batch_size, latent_dim)
            
        Returns:
            Generated images of shape (batch_size, channels, height, width)
        """
        # Project and reshape latent vector
        x = self.fc(z)
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
    
    This model uses a generative approach to create images that
    progressively get closer to a target image through training.
    """
    
    def __init__(self, image_size=(128, 128), latent_dim=100, device=None, mixed_precision=False):
        """
        Initialize the image generator model.
        
        Args:
            image_size: Tuple of (height, width) for the output image size
            latent_dim: Dimension of the latent space for the generator
            device: PyTorch device to use (cuda or cpu)
            mixed_precision: Whether to use mixed precision training (FP16)
        """
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.mixed_precision = mixed_precision
        
        # Initialize device (use provided or auto-detect GPU if available)
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Use the maximum dimension for square output size to ensure we can fit the entire image
        output_size = max(image_size[0], image_size[1])
        
        # Initialize generator model
        self.generator = Generator(latent_dim=latent_dim, output_size=image_size).to(self.device)
        
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
    
    def generate_image(self, latent_vector=None):
        """
        Generate an image using the generator model.
        
        Args:
            latent_vector: Input vector for the generator. If None, a random vector is used.
            
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
                
            # Generate the image
            generated_image = self.generator(latent_vector)
            
            # Move to CPU and convert to numpy array
            image_array = generated_image.cpu().numpy()
            
            # Reshape to expected format (height, width, channels)
            image_array = image_array[0].transpose(1, 2, 0)
            
            # Rescale from [-1, 1] to [0, 1]
            image_array = (image_array + 1) / 2.0
            
            return image_array
    
    def train_step(self, target_image, latent_vector=None):
        """
        Perform one training step to make the generated image closer to the target.
        
        Args:
            target_image: The target image to aim for
            latent_vector: Input vector for the generator. If None, a random vector is used.
            
        Returns:
            Loss value for this training step
        """
        self.generator.train()  # Set to training mode
        
        # Create latent vector if not provided
        if latent_vector is None:
            latent_vector = np.random.normal(0, 1, (1, self.latent_dim))
        
        # Convert latent vector to tensor
        if isinstance(latent_vector, np.ndarray):
            latent_vector = torch.from_numpy(latent_vector).float().to(self.device)
        
        # Convert target image to tensor if it's a numpy array
        if isinstance(target_image, np.ndarray):
            # Reshape target image to (batch_size, channels, height, width)
            if len(target_image.shape) == 3 and target_image.shape[2] == 1:
                # Already has channel dimension but needs transpose
                target_tensor = torch.from_numpy(target_image).float()
                target_tensor = target_tensor.permute(2, 0, 1).unsqueeze(0)
            elif len(target_image.shape) == 2:
                # Add channel dimension
                target_tensor = torch.from_numpy(target_image).float().unsqueeze(0).unsqueeze(0)
            else:
                # Already in correct format
                target_tensor = torch.from_numpy(target_image).float()
                
            target_tensor = target_tensor.to(self.device)
        else:
            target_tensor = target_image
            
        # Rescale target to [-1, 1] to match generator output
        target_tensor = target_tensor * 2 - 1
        
        # Reset gradients
        self.optimizer.zero_grad()
        
        # Use mixed precision if enabled
        if self.mixed_precision:
            with self.autocast():
                # Generate image
                generated_image = self.generator(latent_vector)
                
                # Calculate loss (mean squared error)
                loss = F.mse_loss(generated_image, target_tensor)
                
            # Backpropagate and update weights with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard precision training
            # Generate image
            generated_image = self.generator(latent_vector)
            
            # Calculate loss (mean squared error)
            loss = F.mse_loss(generated_image, target_tensor)
            
            # Backpropagate and update weights
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
    
    def save_model(self, filepath):
        """Save the generator model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'latent_dim': self.latent_dim,
            'image_size': self.image_size
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load the generator model from disk."""
        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Check for image size and latent dimension changes
        image_size_changed = False
        latent_dim_changed = False
        
        if 'image_size' in checkpoint and checkpoint['image_size'] != self.image_size:
            print(f"Note: Saved model has different image size: {checkpoint['image_size']} vs current: {self.image_size}")
            image_size_changed = True
        
        if 'latent_dim' in checkpoint and checkpoint['latent_dim'] != self.latent_dim:
            print(f"Updating latent dimension from {self.latent_dim} to {checkpoint['latent_dim']}")
            self.latent_dim = checkpoint['latent_dim']
            latent_dim_changed = True
        
        # Recreate the generator if needed due to architecture changes
        if latent_dim_changed or image_size_changed:
            # Use the saved image size if available, otherwise keep current
            model_image_size = checkpoint.get('image_size', self.image_size)
            self.generator = Generator(
                latent_dim=self.latent_dim,
                output_size=model_image_size
            ).to(self.device)
        
        # Load the state dictionaries
        try:
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model loaded successfully from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model state: {e}")
            print("This might be due to architecture changes. Creating a new model with saved parameters.")
            # Create a new generator with the right dimensions but don't load weights
            self.generator = Generator(
                latent_dim=self.latent_dim,
                output_size=self.image_size
            ).to(self.device)
            return False