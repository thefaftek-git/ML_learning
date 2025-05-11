"""
Reference Image Manager for Image Generation

This module handles loading and managing reference images for histogram matching
during image generation.
"""
import os
import numpy as np
from utils import load_image

class ReferenceManager:
    """Manager for loading and caching reference images used in histogram matching."""
    
    def __init__(self, mapping=None, data_dir="data"):
        """
        Initialize the reference manager.
        
        Args:
            mapping: Dictionary mapping condition IDs to image info (from condition_mapping.json)
            data_dir: Base directory where reference images are stored
        """
        self.mapping = mapping
        self.data_dir = data_dir
        self.reference_images = {}  # Cache for loaded reference images
        
    def get_reference_image(self, condition_id=0, size=(128, 128)):
        """
        Get a reference image for the specified condition.
        
        Args:
            condition_id: The condition ID to get a reference image for
            size: The size to resize the image to (width, height)
            
        Returns:
            A reference image as numpy array, or None if not found
        """
        # Check if we've already loaded this reference image
        if condition_id in self.reference_images:
            return self.reference_images[condition_id]
            
        # If we have mapping information, try to load the referenced image
        if self.mapping and condition_id in self.mapping:
            filename = self.mapping[condition_id]['filename']
            
            # Try different possible locations for the reference image
            potential_paths = [
                os.path.join(self.data_dir, "reference", filename),
                os.path.join(self.data_dir, "input", filename),
                os.path.join(self.data_dir, filename),
            ]
            
            for path in potential_paths:
                if os.path.exists(path):
                    try:
                        # Load and cache the reference image
                        ref_image = load_image(path, size=size, grayscale=True)
                        self.reference_images[condition_id] = ref_image
                        return ref_image
                    except Exception as e:
                        print(f"Error loading reference image {path}: {e}")
                        continue
        
        # If no specific reference image was found, try to load a default one
        try:
            # Try to find any reference image in the data/reference directory
            ref_dir = os.path.join(self.data_dir, "reference")
            if os.path.exists(ref_dir):
                for filename in os.listdir(ref_dir):
                    if filename.endswith(('.png', '.jpg', '.jpeg')):
                        path = os.path.join(ref_dir, filename)
                        try:
                            ref_image = load_image(path, size=size, grayscale=True)
                            self.reference_images[condition_id] = ref_image
                            print(f"Using {filename} as default reference image")
                            return ref_image
                        except Exception as e:
                            print(f"Error loading default reference image: {e}")
                            continue
        except Exception as e:
            print(f"Error searching for default reference image: {e}")
            
        # If we get here, we couldn't find any reference image
        print("Warning: No reference image found for histogram matching. Generated images may have inconsistent brightness/contrast.")
        return None
