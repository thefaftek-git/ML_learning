"""
Image normalization utilities to ensure consistent brightness and contrast.

This module provides functions to analyze and match the brightness and contrast
between reference and generated images.
"""
import numpy as np
from skimage import exposure

def analyze_image_stats(image):
    """
    Analyze an image to extract brightness and contrast statistics.
    
    Args:
        image: Numpy array of image data with shape (height, width, channels)
        
    Returns:
        Dictionary with mean, std, min, max values for each channel
    """
    stats = {}
    
    # Handle grayscale (height, width, 1) and color (height, width, 3) images
    channels = image.shape[2] if len(image.shape) == 3 else 1
    
    if channels == 1:
        # For grayscale images
        img_data = image.reshape(-1)
        stats['mean'] = np.mean(img_data)
        stats['std'] = np.std(img_data)
        stats['min'] = np.min(img_data)
        stats['max'] = np.max(img_data)
    else:
        # For color images, compute per channel
        stats['mean'] = [np.mean(image[:,:,c]) for c in range(channels)]
        stats['std'] = [np.std(image[:,:,c]) for c in range(channels)]
        stats['min'] = [np.min(image[:,:,c]) for c in range(channels)]
        stats['max'] = [np.max(image[:,:,c]) for c in range(channels)]
    
    return stats

def match_image_brightness_contrast(generated_image, reference_image):
    """
    Match the brightness and contrast of a generated image to a reference image.
    
    This uses histogram matching to ensure the generated image has similar
    brightness and contrast characteristics as the reference.
    
    Args:
        generated_image: Numpy array of generated image (height, width, channels)
        reference_image: Numpy array of reference image (height, width, channels)
        
    Returns:
        Matched image with brightness/contrast similar to reference
    """
    # Handle grayscale and color images
    if generated_image.shape[2] == 1 and reference_image.shape[2] == 1:
        # Grayscale images
        matched = exposure.match_histograms(
            generated_image[:,:,0], 
            reference_image[:,:,0],
            channel_axis=None
        )
        # Restore channel dimension
        return matched[:,:,np.newaxis]
    else:
        # RGB images - match each channel separately
        return exposure.match_histograms(
            generated_image, 
            reference_image,
            channel_axis=2
        )

def adjust_image_range(image, target_min=0.0, target_max=1.0):
    """
    Adjust the range of an image to match a target range.
    
    Args:
        image: Numpy array of image data
        target_min: Target minimum value
        target_max: Target maximum value
        
    Returns:
        Rescaled image
    """
    # Find current min/max values
    current_min = np.min(image)
    current_max = np.max(image)
    
    # Avoid division by zero
    if current_min == current_max:
        return np.full_like(image, target_min)
    
    # Rescale to target range
    rescaled = (image - current_min) / (current_max - current_min)
    rescaled = rescaled * (target_max - target_min) + target_min
    
    return rescaled
