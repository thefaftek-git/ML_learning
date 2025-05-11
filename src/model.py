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
import pytorch_ssim

class Generator(nn.Module):
    """
    A PyTorch neural network model for generating 2D wireframe images.
    
    This implementation uses transposed convolutional layers to generate
    images from a latent vector input and conditional embedding.
    """
    
    def __init__(self, image_size, latent_dim=100, device=None, mixed_precision=False, channels=1, num_conditions=3, tag_dim=0, **kwargs):
        """
        Initialize the generator architecture.
        
        Args:
            image_size: Tuple (height, width) of the output image
            latent_dim: Dimension of the latent space input vector
            device: PyTorch device to run the model on
            mixed_precision: Boolean indicating if mixed precision is used
            channels: Number of channels in the output image (1 for grayscale)
            num_conditions: Number of different conditional outputs the model can generate
            tag_dim: Dimension for tag embeddings if using annotations
        """
        super(Generator, self).__init__()
        
        self.device = device
        self.mixed_precision = mixed_precision  # Stored, though not explicitly used in this snippet for Generator's own layers

        # Store target output size using image_size
        self.target_height = image_size[0] if isinstance(image_size, tuple) else image_size
        self.target_width = image_size[1] if isinstance(image_size, tuple) else image_size
        self.num_conditions = num_conditions
        self.has_tag_support = tag_dim > 0
        self.tag_dim = tag_dim
        
        # Determine the number of upsampling blocks needed
        self.base_size = 4  # Starting size for feature maps (reduced from 8 to create more layers)
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
        self.condition_embedding = nn.Embedding(num_conditions, 64)  # Increased from 32
        
        # Embedding layer for tags if tag support is enabled
        if self.has_tag_support:
            print(f"Enabling tag support with dimension {tag_dim}")
            self.tag_embedding_size = 32  # Increased from 16
            self.tag_projection = nn.Linear(tag_dim, self.tag_embedding_size)
            
            # Initial dense layer to expand from latent space + condition + tags
            self.fc = nn.Linear(latent_dim + 64 + self.tag_embedding_size, self.base_size * self.base_size * 512)  # Increased from 256
        else:
            # Initial dense layer to expand from latent space + condition
            self.fc = nn.Linear(latent_dim + 64, self.base_size * self.base_size * 512)  # Increased from 256
        
        # Create a sequence of upsampling blocks
        layers = []
        in_channels = 512  # Increased from 256
        
        for i in range(self.num_upsample - 1):
            out_channels = max(32, in_channels // 2)  # Increased minimum from 16 to 32
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))  # Added batch normalization
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels
        
        # Final upsampling block to output layer
        layers.append(nn.ConvTranspose2d(in_channels, channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Tanh())  # Output in range [-1, 1]
        
        self.conv_layers = nn.Sequential(*layers)

        if self.device is not None:
            self.to(self.device)  # Move model to specified device
    
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
        
        # Ensure condition vector has the right batch dimension
        if condition_vector.size(0) != batch_size:
            condition_vector = condition_vector.expand(batch_size, -1)
        
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
        x = x.view(x.shape[0], 512, self.base_size, self.base_size)  # Updated to match increased in_channels
        
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

    def train_step(self, target_image, condition_id, latent_vector):
        """
        Performs a single training step on the generator.

        Args:
            target_image: Numpy array representing the target grayscale image.
                          Expected shapes: (H, W), (H, W, 1). It's processed to (B, 1, H, W).
            condition_id: Integer condition ID for the generator.
            latent_vector: Numpy array for the latent vector, shape (B, latent_dim).

        Returns:
            Loss value for the step.
        """
        self.train() # Ensure model is in training mode

        current_batch_size = latent_vector.shape[0]

        # 1. Process target_image (Numpy to Torch Tensor)
        if isinstance(target_image, torch.Tensor):
            img_np = target_image.detach().cpu().numpy()
        else:
            img_np = np.array(target_image, dtype=np.float32)

        if img_np.ndim == 2:  # (H, W)
            img_np = img_np[:, :, np.newaxis]  # Add channel dim: (H, W, 1)
        
        if img_np.ndim == 3: # Single image (H, W, 1) needs to be batched
            if img_np.shape[2] != 1:
                raise ValueError(f"Expected target_image to be grayscale (1 channel), got {img_np.shape[2]} channels.")
            img_np = np.repeat(img_np[np.newaxis, ...], current_batch_size, axis=0) # (B, H, W, 1)
        elif img_np.ndim == 4: # Already batched (B, H, W, 1)
            if img_np.shape[0] != current_batch_size:
                if img_np.shape[0] == 1: # Single image in batch format, repeat for actual batch size
                    img_np = np.repeat(img_np, current_batch_size, axis=0)
                else:
                    raise ValueError(f"Batch size mismatch: target_image batch {img_np.shape[0]}, latent_vector batch {current_batch_size}")
            if img_np.shape[3] != 1:
                raise ValueError(f"Expected batched target_image to be grayscale (1 channel), got {img_np.shape[3]} channels.")
        else:
            raise ValueError(f"Unsupported target_image ndim: {img_np.ndim}. Expected 2, 3, or 4.")

        if img_np.max() > 1.01: # Normalize if pixels are in [0, 255] range
            img_np = img_np / 255.0
        img_np = np.clip(img_np, 0, 1) # Ensure range [0, 1]

        target = torch.tensor(img_np, dtype=torch.float32, device=self.device) # (B, H, W, 1)
        target = target.permute(0, 3, 1, 2)  # Permute to (B, 1, H, W)
        target = target * 2.0 - 1.0  # Remap to [-1, 1] for Tanh output

        # 2. Process latent_vector (Numpy to Torch Tensor)
        z = torch.tensor(latent_vector, dtype=torch.float32, device=self.device)

        # 3. Process condition_id (Scalar to Torch Tensor for batch)
        condition_input = torch.tensor([condition_id] * current_batch_size, dtype=torch.long, device=self.device)

        # 4. Optimizer (Ad-hoc for benchmark step)
        optimizer = optim.Adam(self.parameters(), lr=0.001) # Consider making lr configurable

        # 5. Forward pass
        optimizer.zero_grad()
        generated_output = self.forward(z, condition_input)

        # 6. Calculate loss
        loss = F.mse_loss(generated_output, target)

        # 7. Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        return loss.item()

    def generate_image(self, latent_vector, condition_id=0):
        """
        Generates an image from a latent vector and a condition ID.

        Args:
            latent_vector: Numpy array or Tensor, shape (batch, latent_dim) or (latent_dim,)
            condition_id: Integer, the condition to use for generation. Defaults to 0.

        Returns:
            Generated image as a 2D Numpy array (H, W) with values in [0, 1] if batch_size is 1,
            otherwise a 3D Numpy array (B, H, W).
        """
        self.eval()  # Set the model to evaluation mode

        # 1. Process latent_vector
        if isinstance(latent_vector, np.ndarray):
            if latent_vector.ndim == 1:  # If shape is (latent_dim,)
                latent_vector = latent_vector[np.newaxis, :]  # Add batch dimension
            z = torch.tensor(latent_vector, dtype=torch.float32, device=self.device)
        else:  # It's already a tensor
            z = latent_vector.to(device=self.device, dtype=torch.float32)
            if z.dim() == 1:
                z = z.unsqueeze(0)  # Add batch dimension
        
        current_batch_size = z.size(0)

        # 2. Process condition_id
        # The forward method expects a condition tensor for the batch
        condition_input = torch.tensor([condition_id] * current_batch_size, dtype=torch.long, device=self.device)

        # 3. Forward pass
        with torch.no_grad():
            # Assuming tag_tensor is not needed for simple generation, self.forward handles tag_tensor=None
            img_tensor = self.forward(z, condition_input, tag_tensor=None)

        # 4. Post-process
        img_np = img_tensor.cpu().numpy()  # Shape (B, C, H, W), C=1 for grayscale

        # Denormalize from Tanh output range [-1, 1] to [0, 1]
        img_np = (img_np + 1.0) / 2.0
        img_np = np.clip(img_np, 0, 1) # Ensure values are strictly in [0,1]

        if current_batch_size == 1:
            # For a single image, return (H, W)
            # Input img_np is (1, 1, H, W) for grayscale
            return img_np[0, 0, :, :]
        else:
            # For a batch of images, return (B, H, W)
            # Input img_np is (B, 1, H, W) for grayscale
            return img_np[:, 0, :, :]

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
            image_size=image_size,
            latent_dim=latent_dim, 
            num_conditions=num_conditions,
            device=self.device,
            mixed_precision=mixed_precision,
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
                
            # Ensure batch dimension is correct (handle both single images and batches)
            if len(latent_vector.shape) == 2:  # Shape is [batch_size, latent_dim]
                pass  # Already has correct shape
            elif len(latent_vector.shape) == 1:  # Shape is [latent_dim]
                latent_vector = latent_vector.unsqueeze(0)  # Add batch dimension
              # Convert condition to tensor and match batch size
            batch_size = latent_vector.size(0)
            condition = torch.full((batch_size,), condition_id, dtype=torch.long, device=self.device)
            
            # Process tag tensor if provided
            if tag_tensor is not None and self.has_annotation_support:/:":":":":":":":":":":":":":":":":":":":":":">
                if isinstance(tag_tensor, np.ndarray):
                    tag_tensor = torch.from_numpy(tag_tensor).float().to(self.device)
                # Make sure tag_tensor has correct batch dimension
                if tag_tensor.size(0) != batch_size:
                    tag_tensor = tag_tensor.expand(batch_size, -1)
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
            
    def train_step(self, target_image, condition_id, latent_vector=None):
        """
        Perform one training step with the given target image and condition.
        
        Args:
            target_image: Target image to learn from (numpy array or tensor)
            condition_id: Integer specifying which image type to learn
            latent_vector: Optional latent vectors for generation (if None, will be generated)
            
        Returns:
            Loss value for this training step
        """
        self.generator.train()  # Set to training mode
        
        batch_size = latent_vector.shape[0] if latent_vector is not None else 1
        
        # Process target image
        if isinstance(target_image, np.ndarray):
            # Ensure target image is the right shape and format
            if target_image.ndim == 3 and target_image.shape[2] == 1:
                # Already (height, width, 1)
                pass
            elif target_image.ndim == 2:
                # Add channel dimension
                target_image = target_image[..., np.newaxis]
                
            # Convert to PyTorch tensor
            target_tensor = torch.from_numpy(target_image).float().to(self.device)
            
            # Reshape to (batch_size, channels, height, width)
            target_tensor = target_tensor.permute(2, 0, 1).unsqueeze(0)
            
            # Expand to match batch size if needed
            if batch_size > 1 and target_tensor.size(0) == 1:
                # Use repeat instead of expand for actual data duplication
                target_tensor = target_tensor.repeat(batch_size, 1, 1, 1)
                
            # Scale from [0, 1] to [-1, 1]
            target_tensor = (target_tensor * 2) - 1
        else:
            # Already a tensor, ensure it's on the right device
            target_tensor = target_image.to(self.device)
        
        # Create or process condition tensor
        if isinstance(condition_id, int):
            # Convert single condition ID to tensor
            condition = torch.full((batch_size,), condition_id, dtype=torch.long, device=self.device)
        else:
            # Already a tensor or array
            if isinstance(condition_id, np.ndarray):
                condition = torch.from_numpy(condition_id).long().to(self.device)
            else:
                condition = condition_id.to(self.device)
        
        # Create latent vector if not provided
        if latent_vector is None:
            latent_vector = np.random.normal(0, 1, (batch_size, self.latent_dim))
        
        # Convert latent vector to tensor if needed
        if isinstance(latent_vector, np.ndarray):
            latent_vector = torch.from_numpy(latent_vector).float().to(self.device)
            
        # Zero the gradients
        self.optimizer.zero_grad()
        
        # Use mixed precision if enabled
        if self.mixed_precision:
            with self.autocast():
                # Forward pass
                generated = self.generator(latent_vector, condition)
                
                # Calculate loss (hybrid MSE + SSIM, with correct scaling)
                mse_loss = F.mse_loss(generated, target_tensor)
                # Rescale to [0, 1] for SSIM
                generated_01 = (generated + 1) / 2
                target_tensor_01 = (target_tensor + 1) / 2
                ssim_loss = 1 - pytorch_ssim.ssim(generated_01, target_tensor_01)
                # Use only SSIM loss for training
                loss = ssim_loss
                
            # Backward pass with scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Forward pass
            generated = self.generator(latent_vector, condition)
            
            # Calculate loss (hybrid MSE + SSIM, with correct scaling)
            mse_loss = F.mse_loss(generated, target_tensor)
            generated_01 = (generated + 1) / 2
            target_tensor_01 = (target_tensor + 1) / 2
            ssim_loss = 1 - pytorch_ssim.ssim(generated_01, target_tensor_01)
            # Use only SSIM loss for training
            loss = ssim_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
        
    def train_with_annotations(self, target_image, condition_id, latent_vector=None, 
                             tag_tensor=None, region_masks=None, region_tag_tensor=None):
        """
        Train the model with additional annotation information.
        
        Args:
            target_image: Target image to learn from (numpy array)
            condition_id: Integer specifying which image type to learn
            latent_vector: Optional latent vectors for generation
            tag_tensor: Optional tensor with tag information (numpy array)
            region_masks: Optional tensor with region masks (numpy array)
            region_tag_tensor: Optional tensor with region-specific tag information
        
        Returns:
            Loss value for this training step
        """
        self.generator.train()  # Set to training mode
        
        batch_size = latent_vector.shape[0] if latent_vector is not None else 1
        
        # Process target image
        if isinstance(target_image, np.ndarray):
            # Ensure target image is the right shape and format
            if target_image.ndim == 3 and target_image.shape[2] == 1:
                # Already (height, width, 1)
                pass
            elif target_image.ndim == 2:
                # Add channel dimension
                target_image = target_image[..., np.newaxis]
                
            # Convert to PyTorch tensor
            target_tensor = torch.from_numpy(target_image).float().to(self.device)
            
            # Reshape to (batch_size, channels, height, width)
            target_tensor = target_tensor.permute(2, 0, 1).unsqueeze(0)
            
            # Expand to match batch size if needed
            if batch_size > 1 and target_tensor.size(0) == 1:
                # Use repeat instead of expand for actual data duplication
                target_tensor = target_tensor.repeat(batch_size, 1, 1, 1)
                
            # Scale from [0, 1] to [-1, 1]
            target_tensor = (target_tensor * 2) - 1
        else:
            # Already a tensor, ensure it's on the right device
            target_tensor = target_image.to(self.device)
        
        # Create or process condition tensor
        if isinstance(condition_id, int):
            # Convert single condition ID to tensor
            condition = torch.full((batch_size,), condition_id, dtype=torch.long, device=self.device)
        else:
            # Already a tensor or array
            if isinstance(condition_id, np.ndarray):
                condition = torch.from_numpy(condition_id).long().to(self.device)
            else:
                condition = condition_id.to(self.device)
        
        # Create latent vector if not provided
        if latent_vector is None:
            latent_vector = np.random.normal(0, 1, (batch_size, self.latent_dim))
        
        # Convert latent vector to tensor if needed
        if isinstance(latent_vector, np.ndarray):
            latent_vector = torch.from_numpy(latent_vector).float().to(self.device)
        
        # Process tag tensor if provided
        if tag_tensor is not None and self.has_annotation_support:
            if isinstance(tag_tensor, np.ndarray):
                tag_tensor = torch.from_numpy(tag_tensor).float().to(self.device)
        else:
            tag_tensor = None
        
        # Zero the gradients
        self.optimizer.zero_grad()
        
        # Use mixed precision if enabled
        if self.mixed_precision:
            with self.autocast():
                # Forward pass
                generated = self.generator(latent_vector, condition, tag_tensor)
                
                # Calculate loss (hybrid MSE + SSIM, with correct scaling)
                mse_loss = F.mse_loss(generated, target_tensor)
                # Rescale to [0, 1] for SSIM
                generated_01 = (generated + 1) / 2
                target_tensor_01 = (target_tensor + 1) / 2
                ssim_loss = 1 - pytorch_ssim.ssim(generated_01, target_tensor_01)
                # Use only SSIM loss for annotation training (mixed-precision branch)
                loss = ssim_loss
                
            # Backward pass with scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Forward pass
            generated = self.generator(latent_vector, condition, tag_tensor)
            
            # Calculate loss (hybrid MSE + SSIM, with correct scaling)
            mse_loss = F.mse_loss(generated, target_tensor)
            generated_01 = (generated + 1) / 2
            target_tensor_01 = (target_tensor + 1) / 2
            ssim_loss = 1 - pytorch_ssim.ssim(generated_01, target_tensor_01)
            # Use only SSIM loss for annotation training (non-mixed-precision branch)
            loss = ssim_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
        
        return loss.item()