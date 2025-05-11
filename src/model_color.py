import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImageGeneratorColor(nn.Module):
    """
    A simple color (RGB) image generator model for demonstration.
    This should mirror the interface of your grayscale ImageGenerator,
    but outputs 3 channels (RGB) instead of 1 (grayscale).
    """
    def __init__(self, image_size, latent_dim, device, mixed_precision=False, **kwargs):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.device = device
        self.mixed_precision = mixed_precision
        # Example: a simple MLP to image (replace with your actual architecture)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, image_size[0] * image_size[1] * 3),
            nn.Tanh()
        )
        self.to(device)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 3, self.image_size[0], self.image_size[1])
        return x
        
    def generate_image(self, latent_vector):
        """
        Generate a color image from a latent vector.
        
        Args:
            latent_vector: numpy array or tensor, shape (batch, latent_dim) or (latent_dim,)
            
        Returns:
            Generated image as numpy array with shape (H, W, 3)
        """
        # Convert latent vector to proper format
        if isinstance(latent_vector, np.ndarray):
            # Add batch dimension if it doesn't exist
            if latent_vector.ndim == 1:  # If shape is (latent_dim,)
                latent_vector = latent_vector[np.newaxis, :]  # Add batch dimension
            z = torch.tensor(latent_vector, dtype=torch.float32, device=self.device)
        else:  # It's already a tensor
            z = latent_vector.to(device=self.device)
            # Add batch dimension if needed
            if z.dim() == 1:
                z = z.unsqueeze(0)  # Add batch dimension
                
        with torch.no_grad():
            img = self.forward(z)
        img = img.cpu().numpy()
        # Map from [-1, 1] to [0, 1]
        img = (img + 1) / 2
        # Return shape (H, W, C) for a single image
        if img.shape[0] == 1:
            return img[0].transpose(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
        else:
            # Multiple images in batch, keep batch dim
            return img.transpose(0, 2, 3, 1)  # Change from (B, C, H, W) to (B, H, W, C)

    def train_step(self, target_image, condition_id, latent_vector):
        # Preprocess target image: ensure float32, shape (B, H, W, 3), values in [0, 1]
        if isinstance(target_image, torch.Tensor):
            img = target_image.detach().cpu().numpy()
        else:
            img = target_image
        # If input is (H, W, 3), add batch dim
        if img.ndim == 3:
            img = img[None, ...]
        # Ensure float32 and values in [0, 1]
        img = img.astype('float32')
        if img.max() > 1.01:
            img = img / 255.0
        # Clamp to [0, 1]
        img = img.clip(0, 1)
        # Convert to torch tensor
        target = torch.tensor(img, dtype=torch.float32, device=self.device)
        # Permute to (B, 3, H, W)
        target = target.permute(0, 3, 1, 2)
        # Remap to [-1, 1]
        target = target * 2 - 1
        # Prepare latent vector
        z = torch.tensor(latent_vector, dtype=torch.float32, device=self.device)
        
        # Make sure batch sizes match - use repeat instead of expand to avoid broadcasting warnings
        batch_size = z.size(0)
        if target.size(0) == 1 and batch_size > 1:
            target = target.repeat(batch_size, 1, 1, 1)
            
        # Forward pass
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        optimizer.zero_grad()
        output = self.forward(z)
        # MSE loss
        loss = F.mse_loss(output, target)
        # Backward pass
        loss.backward()
        optimizer.step()
        return loss.item()

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
