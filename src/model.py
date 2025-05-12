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

# Robust import for pytorch_ssim
try:
    from pytorch_ssim import create_window as pytorch_ssim_create_window, _ssim as pytorch_ssim_internal_ssim
    import pytorch_ssim as pytorch_ssim_module
    PYTORCH_SSIM_WINDOW_SIZE = getattr(pytorch_ssim_module, 'SSIM_WIN_SIZE', 11)
except ImportError:
    print("Warning: pytorch_ssim library not found or structured as expected. SSIM calculation might fail.")
    pytorch_ssim_create_window = lambda a, b: torch.tensor([])
    pytorch_ssim_internal_ssim = lambda a, b, c, d, e, size_average: torch.tensor(0.0)
    PYTORCH_SSIM_WINDOW_SIZE = 11


class Generator(nn.Module):
    """
    A PyTorch neural network model for generating images.
    """
    def __init__(self, image_size, latent_dim=100, device=None, mixed_precision=False, channels=1, num_conditions=3, tag_dim=0, **kwargs):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mixed_precision = mixed_precision
        self.channels = channels

        self.target_height = image_size[0] if isinstance(image_size, tuple) else image_size
        self.target_width = image_size[1] if isinstance(image_size, tuple) else image_size
        self.num_conditions = num_conditions
        self.has_tag_support = tag_dim > 0
        self.tag_dim = tag_dim
        
        self.base_size = 4
        self.num_upsample = 0
        
        max_dimension = max(self.target_height, self.target_width)
        temp_size = self.base_size
        while temp_size < max_dimension:
            temp_size *= 2
            self.num_upsample += 1
        
        self.generated_size = self.base_size * (2 ** self.num_upsample)
        
        self.condition_embedding = nn.Embedding(num_conditions, 64)
        
        fc_input_dim = self.latent_dim + 64 
        if self.has_tag_support:
            self.tag_projection = nn.Linear(tag_dim, 64)
            fc_input_dim += 64

        self.fc = nn.Linear(fc_input_dim, 512 * self.base_size * self.base_size)

        layers = []
        current_channels = 512
        
        for i in range(self.num_upsample - 1):
            out_channels = max(current_channels // 2, 32) 
            layers.append(nn.ConvTranspose2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_channels = out_channels
        
        layers.append(nn.ConvTranspose2d(current_channels, self.channels, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.Tanh())
        
        self.conv_layers = nn.Sequential(*layers)

        if self.device is not None:
            self.to(self.device)

    def forward(self, z, condition=None, tag_tensor=None):
        batch_size = z.size(0)
        
        if condition is None:
            condition = torch.zeros(batch_size, dtype=torch.long, device=z.device)
        
        condition_vector = self.condition_embedding(condition)
        
        combined_input_list = [z, condition_vector]
        if self.has_tag_support and tag_tensor is not None:
            processed_tag_tensor = tag_tensor.to(z.device)
            if processed_tag_tensor.dim() == 1:
                processed_tag_tensor = processed_tag_tensor.unsqueeze(0)
            if processed_tag_tensor.size(0) == 1 and batch_size > 1:
                processed_tag_tensor = processed_tag_tensor.expand(batch_size, -1)
            elif processed_tag_tensor.size(0) != batch_size:
                raise ValueError(f"Batch size mismatch for tag_tensor. Expected {batch_size}, got {processed_tag_tensor.size(0)}")

            tag_embedding = self.tag_projection(processed_tag_tensor)
            combined_input_list.append(tag_embedding)
            
        combined_input = torch.cat(combined_input_list, dim=1)
        
        x = self.fc(combined_input)
        x = x.view(batch_size, 512, self.base_size, self.base_size)
        
        x = self.conv_layers(x)
        
        if (x.shape[2] != self.target_height) or (x.shape[3] != self.target_width):
            x = F.interpolate(
                x, 
                size=(self.target_height, self.target_width), 
                mode='bilinear', 
                align_corners=False
            )
        return x


class ImageGenerator:
    def __init__(self, image_size=(128, 128), latent_dim=100, num_conditions=3, 
                 device=None, mixed_precision=False, tag_dim=0, channels=1):
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_conditions = num_conditions
        self.mixed_precision = mixed_precision
        self.tag_dim = tag_dim
        self.has_annotation_support = tag_dim > 0
        self.channels = channels

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device
            
            if self.device.type == "cuda" and not torch.cuda.is_available():
                print("WARNING: CUDA requested but not available. Falling back to CPU.")
                self.device = torch.device("cpu")
            
        self.generator = Generator(
            image_size=self.image_size,
            latent_dim=self.latent_dim, 
            num_conditions=self.num_conditions,
            device=self.device,
            mixed_precision=self.mixed_precision,
            tag_dim=self.tag_dim,
            channels=self.channels
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=0.0002,
            betas=(0.5, 0.999)
        )
        
        self.autocast = None
        self.scaler = None
        if self.mixed_precision and self.device.type == 'cuda':
            try:
                from torch.cuda.amp import autocast, GradScaler
                self.scaler = GradScaler()
                self.autocast = autocast
            except ImportError:
                print("Warning: torch.cuda.amp not available. Mixed precision disabled.")
                self.mixed_precision = False
        elif self.mixed_precision and self.device.type != 'cuda':
            print("Warning: Mixed precision requested but device is not CUDA. Disabled.")
            self.mixed_precision = False

    def generate_image(self, latent_vector=None, condition_id=0, tag_tensor=None):
        self.generator.eval()
        with torch.no_grad():
            if latent_vector is None:
                latent_vector_np = np.random.normal(0, 1, (1, self.latent_dim))
                z = torch.from_numpy(latent_vector_np).float().to(self.device)
            elif isinstance(latent_vector, np.ndarray):
                z = torch.from_numpy(latent_vector).float().to(self.device)
            elif isinstance(latent_vector, torch.Tensor):
                z = latent_vector.float().to(self.device)
            else:
                raise TypeError(f"Unsupported latent_vector type: {type(latent_vector)}")

            if z.dim() == 1:
                z = z.unsqueeze(0)
            
            batch_size = z.size(0)
            condition = torch.full((batch_size,), condition_id, dtype=torch.long, device=self.device)
            
            processed_tag_tensor = None
            if tag_tensor is not None and self.has_annotation_support:
                if isinstance(tag_tensor, np.ndarray):
                    processed_tag_tensor = torch.from_numpy(tag_tensor).float()
                elif isinstance(tag_tensor, torch.Tensor):
                    processed_tag_tensor = tag_tensor.float()
                else:
                    raise TypeError(f"Unsupported tag_tensor type: {type(tag_tensor)}")
                
                processed_tag_tensor = processed_tag_tensor.to(self.device)
                if processed_tag_tensor.dim() == 1 and self.tag_dim > 0:
                    processed_tag_tensor = processed_tag_tensor.unsqueeze(0)

                if processed_tag_tensor.size(0) == 1 and batch_size > 1:
                    processed_tag_tensor = processed_tag_tensor.expand(batch_size, -1)
                elif processed_tag_tensor.size(0) != batch_size:
                    raise ValueError(f"Batch size mismatch for tag_tensor. Expected {batch_size}, got {processed_tag_tensor.size(0)}")

            generated_image_tensor = self.generator(z, condition, processed_tag_tensor) 
            image_array = generated_image_tensor.cpu().numpy()
            
            if batch_size == 1:
                image_array = image_array[0] 
                image_array = image_array.transpose(1, 2, 0)
                if self.channels == 1:
                    image_array = image_array.squeeze(axis=-1)
            else:
                image_array = image_array.transpose(0, 2, 3, 1)
                if self.channels == 1:
                    image_array = image_array.squeeze(axis=-1)

            image_array = (image_array + 1) / 2.0
            image_array = np.clip(image_array, 0, 1)
            return image_array
            
    def train_step(self, target_image, condition_id, latent_vector=None):
        self.generator.train()
        
        current_batch_size = 1
        if latent_vector is not None:
            if isinstance(latent_vector, np.ndarray): current_batch_size = latent_vector.shape[0]
            elif isinstance(latent_vector, torch.Tensor): current_batch_size = latent_vector.size(0)
        
        if isinstance(target_image, np.ndarray):
            img_np = target_image.astype(np.float32)
            if img_np.ndim == 2:
                img_np = img_np[:, :, np.newaxis]
            if img_np.shape[-1] != self.channels:
                if self.channels == 1 and img_np.ndim == 3 and img_np.shape[-1] != 1:
                    print(f"Warning: Target image has {img_np.shape[-1]} channels, but model expects {self.channels}. Converting to grayscale.")
                    img_np = np.mean(img_np, axis=2, keepdims=True)
                elif self.channels == 3 and img_np.ndim == 3 and img_np.shape[-1] == 1:
                    print(f"Warning: Target image has 1 channel, but model expects {self.channels}. Repeating channel.")
                    img_np = np.repeat(img_np, 3, axis=2)

            if img_np.ndim == 3:
                img_np_batch = np.repeat(img_np[np.newaxis, ...], current_batch_size, axis=0)
            elif img_np.ndim == 4 and img_np.shape[0] == 1 and current_batch_size > 1:
                img_np_batch = np.repeat(img_np, current_batch_size, axis=0)
            elif img_np.ndim == 4 and img_np.shape[0] == current_batch_size:
                img_np_batch = img_np
            else:
                raise ValueError(f"Unsupported target_image numpy shape: {img_np.shape}")

            target_tensor = torch.from_numpy(img_np_batch).permute(0, 3, 1, 2).to(self.device)
            
            if target_tensor.max() > 1.001: target_tensor = target_tensor / 255.0
            target_tensor = torch.clamp(target_tensor, 0.0, 1.0)
            target_tensor = (target_tensor * 2.0) - 1.0
        
        elif isinstance(target_image, torch.Tensor):
            target_tensor = target_image.to(self.device)
            if target_tensor.dim() == 2:
                target_tensor = target_tensor.unsqueeze(0).unsqueeze(0) 
            elif target_tensor.dim() == 3:
                if target_tensor.size(0) == self.channels: target_tensor = target_tensor.unsqueeze(0) 
                elif target_tensor.size(2) == self.channels: target_tensor = target_tensor.permute(2,0,1).unsqueeze(0)
                else: raise ValueError(f"Unsupported 3D target_tensor shape: {target_tensor.shape} for {self.channels} channels")
            
            if target_tensor.size(0) == 1 and current_batch_size > 1:
                target_tensor = target_tensor.repeat(current_batch_size, 1, 1, 1)
            
            if not (target_tensor.min() >= -1.001 and target_tensor.max() <= 1.001):
                if target_tensor.min() >= -0.001 and target_tensor.max() <= 1.001:
                    target_tensor = (target_tensor * 2.0) - 1.0
                elif target_tensor.min() >= -0.001 and target_tensor.max() > 1.001:
                    target_tensor = target_tensor / 255.0
                    target_tensor = (target_tensor * 2.0) - 1.0
                else:
                    print(f"Warning: Target tensor values are outside expected ranges ([0,1] or [-1,1]): min {target_tensor.min()}, max {target_tensor.max()}")

        else: raise TypeError(f"Unsupported target_image type: {type(target_image)}")

        if isinstance(condition_id, int):
            condition = torch.full((current_batch_size,), condition_id, dtype=torch.long, device=self.device)
        elif isinstance(condition_id, (np.ndarray, torch.Tensor)):
            condition = torch.as_tensor(condition_id, dtype=torch.long, device=self.device)
            if condition.dim() == 0: condition = condition.repeat(current_batch_size)
            elif condition.size(0) == 1 and current_batch_size > 1: condition = condition.repeat(current_batch_size)
        else: raise TypeError(f"Unsupported condition_id type: {type(condition_id)}")

        if latent_vector is None:
            latent_vector_np = np.random.normal(0, 1, (current_batch_size, self.latent_dim))
            z = torch.from_numpy(latent_vector_np).float().to(self.device)
        elif isinstance(latent_vector, np.ndarray): z = torch.from_numpy(latent_vector).float().to(self.device)
        elif isinstance(latent_vector, torch.Tensor): z = latent_vector.float().to(self.device)
        else: raise TypeError(f"Unsupported latent_vector type: {type(latent_vector)}")

        self.optimizer.zero_grad(set_to_none=True)
        
        loss_val = 0.0
        if self.mixed_precision and self.autocast is not None:
            with self.autocast():
                generated = self.generator(z, condition)
                generated_01 = (generated + 1) / 2.0
                target_tensor_01 = (target_tensor + 1) / 2.0
                window = pytorch_ssim_create_window(PYTORCH_SSIM_WINDOW_SIZE, self.channels).to(self.device)
                ssim_val = pytorch_ssim_internal_ssim(generated_01, target_tensor_01, window, PYTORCH_SSIM_WINDOW_SIZE, self.channels, size_average=True)
                current_loss = 1.0 - ssim_val
            if self.scaler: self.scaler.scale(current_loss).backward()
            else: current_loss.backward()
            if self.scaler: self.scaler.step(self.optimizer)
            if self.scaler: self.scaler.update()
        else:
            generated = self.generator(z, condition)
            generated_01 = (generated + 1) / 2.0
            target_tensor_01 = (target_tensor + 1) / 2.0
            window = pytorch_ssim_create_window(PYTORCH_SSIM_WINDOW_SIZE, self.channels).to(self.device)
            ssim_val = pytorch_ssim_internal_ssim(generated_01, target_tensor_01, window, PYTORCH_SSIM_WINDOW_SIZE, self.channels, size_average=True)
            current_loss = 1.0 - ssim_val
            current_loss.backward()
            self.optimizer.step()
        loss_val = current_loss.item()
        return loss_val
        
    def train_with_annotations(self, target_image, condition_id, latent_vector=None, 
                             tag_tensor=None, region_masks=None, region_tag_tensor=None):
        self.generator.train()
        current_batch_size = 1
        if latent_vector is not None:
            if isinstance(latent_vector, np.ndarray): current_batch_size = latent_vector.shape[0]
            elif isinstance(latent_vector, torch.Tensor): current_batch_size = latent_vector.size(0)
        elif tag_tensor is not None:
            if isinstance(tag_tensor, np.ndarray): current_batch_size = tag_tensor.shape[0]
            elif isinstance(tag_tensor, torch.Tensor): current_batch_size = tag_tensor.size(0)

        if isinstance(target_image, np.ndarray):
            img_np = target_image.astype(np.float32)
            if img_np.ndim == 2: img_np = img_np[:,:,np.newaxis]
            if img_np.shape[-1] != self.channels:
                if self.channels==1: img_np = np.mean(img_np, axis=2, keepdims=True)
                else: img_np = np.repeat(img_np, self.channels, axis=2) if img_np.shape[-1]==1 else img_np[:,:,:self.channels]

            img_np_batch = np.repeat(img_np[np.newaxis, ...], current_batch_size, axis=0)
            target_tensor_p = torch.from_numpy(img_np_batch).permute(0, 3, 1, 2).to(self.device)
            if target_tensor_p.max() > 1.001: target_tensor_p = target_tensor_p / 255.0
            target_tensor_p = torch.clamp(target_tensor_p, 0.0, 1.0)
            target_tensor_p = (target_tensor_p * 2.0) - 1.0
        elif isinstance(target_image, torch.Tensor):
            target_tensor_p = target_image.to(self.device)
        else:
            raise TypeError("Invalid target_image type")

        z_p = torch.randn(current_batch_size, self.latent_dim, device=self.device)
        if latent_vector is not None:
             z_p = torch.as_tensor(latent_vector, dtype=torch.float, device=self.device)
        if z_p.dim()==1: z_p = z_p.unsqueeze(0)
        if z_p.size(0)==1 and current_batch_size > 1: z_p = z_p.repeat(current_batch_size,1)

        condition_p = torch.as_tensor([condition_id] * current_batch_size, dtype=torch.long, device=self.device)

        processed_tag_tensor_p = None
        if tag_tensor is not None and self.has_annotation_support:
            processed_tag_tensor_p = torch.as_tensor(tag_tensor, dtype=torch.float, device=self.device)
            if processed_tag_tensor_p.dim() == 1: processed_tag_tensor_p = processed_tag_tensor_p.unsqueeze(0)
            if processed_tag_tensor_p.size(0) == 1 and current_batch_size > 1:
                processed_tag_tensor_p = processed_tag_tensor_p.repeat(current_batch_size, 1)
        
        self.optimizer.zero_grad(set_to_none=True)
        loss_val = 0.0
        if self.mixed_precision and self.autocast is not None:
            with self.autocast():
                generated = self.generator(z_p, condition_p, processed_tag_tensor_p)
                generated_01 = (generated + 1) / 2.0
                target_tensor_01 = (target_tensor_p + 1) / 2.0
                window = pytorch_ssim_create_window(PYTORCH_SSIM_WINDOW_SIZE, self.channels).to(self.device)
                ssim_val = pytorch_ssim_internal_ssim(generated_01, target_tensor_01, window, PYTORCH_SSIM_WINDOW_SIZE, self.channels, size_average=True)
                current_loss = 1.0 - ssim_val
            if self.scaler: self.scaler.scale(current_loss).backward()
            else: current_loss.backward()
            if self.scaler: self.scaler.step(self.optimizer)
            if self.scaler: self.scaler.update()
        else:
            generated = self.generator(z_p, condition_p, processed_tag_tensor_p)
            generated_01 = (generated + 1) / 2.0
            target_tensor_01 = (target_tensor_p + 1) / 2.0
            window = pytorch_ssim_create_window(PYTORCH_SSIM_WINDOW_SIZE, self.channels).to(self.device)
            ssim_val = pytorch_ssim_internal_ssim(generated_01, target_tensor_01, window, PYTORCH_SSIM_WINDOW_SIZE, self.channels, size_average=True)
            current_loss = 1.0 - ssim_val
            current_loss.backward()
            self.optimizer.step()
        loss_val = current_loss.item()
        return loss_val

    def save_model(self, file_path):
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        torch.save(self.generator.state_dict(), file_path)

    def load_model(self, file_path):
        if os.path.exists(file_path):
            try:
                state_dict = torch.load(file_path, map_location=self.device)
                self.generator.load_state_dict(state_dict)
                self.generator.to(self.device) 
                self.generator.eval() 
            except Exception as e:
                print(f"Error loading model from {file_path}: {e}. Starting with a new model.")
        else:
            print(f"Warning: Model file not found at {file_path}. Starting with a new model.")

    def get_architecture_info(self):
        return {
            "image_size": self.image_size,
            "latent_dim": self.latent_dim,
            "num_conditions": self.num_conditions,
            "device": str(self.device),
            "mixed_precision": self.mixed_precision,
            "tag_dim": self.tag_dim,
            "channels": self.channels,
            "generator_base_size": self.generator.base_size if hasattr(self.generator, 'base_size') else 'N/A',
            "generator_num_upsample": self.generator.num_upsample if hasattr(self.generator, 'num_upsample') else 'N/A',
            "generator_generated_size": self.generator.generated_size if hasattr(self.generator, 'generated_size') else 'N/A',
        }