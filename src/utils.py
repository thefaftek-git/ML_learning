"""
Utilities for Image Processing and SVG Handling

This module provides utility functions for loading, processing, and saving images,
particularly for SVG format handling required by the image generator.
"""

import os
import numpy as np
import svgwrite
from PIL import Image
import io
import matplotlib.pyplot as plt
import json
from skimage.draw import polygon, rectangle
import cv2

# Try to import cairosvg but provide a fallback if not available
CAIRO_AVAILABLE = False
try:
    import cairosvg
    CAIRO_AVAILABLE = True
except (ImportError, OSError):
    print("Warning: CairoSVG not available. SVG import functionality will be limited.")
    # Continue without cairosvg

# Annotation handling functions
def find_annotation_file(image_path):
    """
    Find an annotation file associated with an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Path to the annotation file if it exists, None otherwise
    """
    base_path = os.path.splitext(image_path)[0]
    anno_path = f"{base_path}.anno.json"
    
    if os.path.exists(anno_path):
        print(f"Found annotation file: {anno_path}")
        return anno_path
    return None

def load_annotations(annotation_path):
    """
    Load annotations from a JSON file.
    
    Args:
        annotation_path: Path to the annotation JSON file
        
    Returns:
        Dictionary containing the parsed annotations
    """
    try:
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        return annotations
    except Exception as e:
        print(f"Error loading annotation file: {e}")
        return None

def create_tag_mapping(annotations_list):
    """
    Create a mapping from tags to numerical indices.
    
    Args:
        annotations_list: List of annotation dictionaries
        
    Returns:
        Dictionary mapping tag names to indices
    """
    # Collect all unique tags from all annotations
    all_tags = set()
    
    for annotations in annotations_list:
        if not annotations:
            continue
            
        # Global image tags
        if 'tags' in annotations:
            for tag in annotations['tags']:
                all_tags.add(tag)
                
        # Region-specific tags
        if 'regions' in annotations:
            for region in annotations['regions']:
                if 'tags' in region:
                    for tag in region['tags']:
                        all_tags.add(tag)
    
    # Create a mapping from tags to indices
    tag_mapping = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
    
    print(f"Created tag mapping with {len(tag_mapping)} unique tags")
    return tag_mapping

def regions_to_mask(regions, image_shape):
    """
    Convert annotation regions to a binary mask.
    
    Args:
        regions: List of region dictionaries from the annotation file
        image_shape: Tuple of (height, width) for the target mask
        
    Returns:
        Binary mask as a numpy array with shape (height, width, 1)
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.float32)
    
    for region in regions:
        region_type = region.get('type', 'unknown')
        
        if region_type == 'rectangle':
            # Rectangle: [x, y, width, height]
            x, y = int(region['x']), int(region['y'])
            w, h = int(region['width']), int(region['height'])
            rr, cc = rectangle((y, x), (y + h - 1, x + w - 1))
            # Make sure coordinates are within bounds
            valid_idx = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            mask[rr[valid_idx], cc[valid_idx]] = 1
            
        elif region_type == 'polygon':
            # Polygon: list of [x, y] coordinates
            points = np.array(region['points'])
            # Separate x and y coordinates
            row_coords = points[:, 1].astype(int)
            col_coords = points[:, 0].astype(int)
            # Draw polygon on mask
            rr, cc = polygon(row_coords, col_coords, mask.shape)
            mask[rr, cc] = 1
            
        elif region_type == 'brush' or region_type == 'freeform':
            # Brush strokes or freeform paths
            # These are typically stored as paths with control points
            if 'path' in region:
                # For simplicity, we'll draw lines between consecutive points
                points = np.array(region['path'])
                for i in range(len(points) - 1):
                    pt1 = (int(points[i][0]), int(points[i][1]))
                    pt2 = (int(points[i+1][0]), int(points[i+1][1]))
                    cv2.line(mask, pt1, pt2, 1, thickness=region.get('thickness', 1))
    
    # Reshape to (height, width, 1)
    return mask.reshape(height, width, 1)

def create_tag_tensor(annotations, tag_mapping, total_tags):
    """
    Create a tensor representing the tags present in the annotations.
    
    Args:
        annotations: Annotations dictionary
        tag_mapping: Dictionary mapping tag names to indices
        total_tags: Total number of unique tags
        
    Returns:
        One-hot encoded tensor representing the tags
    """
    tag_tensor = np.zeros(total_tags, dtype=np.float32)
    
    if not annotations or 'tags' not in annotations:
        return tag_tensor
    
    for tag in annotations['tags']:
        if tag in tag_mapping:
            tag_tensor[tag_mapping[tag]] = 1.0
    
    return tag_tensor

def create_region_tag_tensor(annotations, tag_mapping, total_tags, image_shape):
    """
    Create a tensor representing tags for specific regions in the image.
    
    Args:
        annotations: Annotations dictionary
        tag_mapping: Dictionary mapping tag names to indices
        total_tags: Total number of unique tags
        image_shape: Tuple of (height, width) for the image
        
    Returns:
        Tensor of shape (height, width, total_tags) with tag information per pixel
    """
    height, width = image_shape
    region_tag_tensor = np.zeros((height, width, total_tags), dtype=np.float32)
    
    if not annotations or 'regions' not in annotations:
        return region_tag_tensor
    
    for region in annotations['regions']:
        if 'tags' not in region:
            continue
            
        # Create a mask for this region
        region_mask = regions_to_mask([region], image_shape)
        
        # Set tag values for this region
        for tag in region['tags']:
            if tag in tag_mapping:
                tag_idx = tag_mapping[tag]
                region_tag_tensor[:, :, tag_idx] = np.maximum(
                    region_tag_tensor[:, :, tag_idx],
                    region_mask[:, :, 0]
                )
    
    return region_tag_tensor

def augment_with_annotations(image, annotations, tag_mapping=None):
    """
    Apply data augmentation to an image and its annotations together.
    
    Args:
        image: Image as a numpy array with shape (height, width, channels)
        annotations: Annotations dictionary
        tag_mapping: Optional dictionary mapping tag names to indices
        
    Returns:
        Tuple of (augmented_image, updated_annotations)
    """
    # Simple augmentations that preserve annotations
    height, width = image.shape[:2]
    
    # If no annotations or no regions, just return the original
    if not annotations or 'regions' not in annotations:
        return image, annotations
        
    # Choose an augmentation randomly
    augmentation_type = np.random.choice(['none', 'flip_h', 'flip_v', 'rotate_90'])
    
    if augmentation_type == 'none':
        return image, annotations
    
    # Create a deep copy of annotations to modify
    new_annotations = json.loads(json.dumps(annotations))
    
    # Apply the selected augmentation
    if augmentation_type == 'flip_h':
        # Flip image horizontally
        augmented_image = np.fliplr(image).copy()
        
        # Update region coordinates
        for region in new_annotations['regions']:
            region_type = region.get('type', 'unknown')
            
            if region_type == 'rectangle':
                region['x'] = width - region['x'] - region['width']
                
            elif region_type == 'polygon':
                for point in region['points']:
                    point[0] = width - point[0]
                    
            elif region_type == 'brush' or region_type == 'freeform':
                for point in region['path']:
                    point[0] = width - point[0]
                    
    elif augmentation_type == 'flip_v':
        # Flip image vertically
        augmented_image = np.flipud(image).copy()
        
        # Update region coordinates
        for region in new_annotations['regions']:
            region_type = region.get('type', 'unknown')
            
            if region_type == 'rectangle':
                region['y'] = height - region['y'] - region['height']
                
            elif region_type == 'polygon':
                for point in region['points']:
                    point[1] = height - point[1]
                    
            elif region_type == 'brush' or region_type == 'freeform':
                for point in region['path']:
                    point[1] = height - point[1]
                    
    elif augmentation_type == 'rotate_90':
        # Rotate image 90 degrees
        augmented_image = np.rot90(image).copy()
        new_height, new_width = augmented_image.shape[:2]
        
        # Update region coordinates
        for region in new_annotations['regions']:
            region_type = region.get('type', 'unknown')
            
            if region_type == 'rectangle':
                old_x, old_y = region['x'], region['y']
                old_w, old_h = region['width'], region['height']
                
                # In a 90-degree rotation, we swap x,y and width,height
                region['x'] = height - old_y - old_h
                region['y'] = old_x
                region['width'], region['height'] = old_h, old_w
                
            elif region_type == 'polygon':
                for point in region['points']:
                    old_x, old_y = point
                    point[0] = height - old_y
                    point[1] = old_x
                    
            elif region_type == 'brush' or region_type == 'freeform':
                for point in region['path']:
                    old_x, old_y = point
                    point[0] = height - old_y
                    point[1] = old_x
    
    # Update image dimensions in annotations
    new_annotations['width'] = augmented_image.shape[1]
    new_annotations['height'] = augmented_image.shape[0]
    
    return augmented_image, new_annotations

def visualize_annotations(image, annotations, output_path=None, show=False):
    """
    Create a visualization of the image with annotations overlaid.
    
    Args:
        image: Image as a numpy array with shape (height, width, channels)
        annotations: Annotations dictionary
        output_path: Path to save the visualization (optional)
        show: Whether to display the visualization (default: False)
        
    Returns:
        Visualization image as numpy array
    """
    if image.ndim == 2:
        # Convert grayscale to RGB
        viz_image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 1:
        # Convert single-channel to RGB
        viz_image = np.concatenate([image] * 3, axis=-1)
    else:
        # Already RGB
        viz_image = image.copy()
    
    # Make sure image is in proper range for visualization
    if viz_image.max() <= 1.0:
        viz_image = viz_image * 255
    viz_image = viz_image.astype(np.uint8)
    
    # Draw annotations
    if annotations and 'regions' in annotations:
        for region in annotations['regions']:
            region_type = region.get('type', 'unknown')
            
            # Choose a random color for this region
            color = np.random.randint(0, 255, 3).tolist()
            
            if region_type == 'rectangle':
                x, y = int(region['x']), int(region['y'])
                w, h = int(region['width']), int(region['height'])
                cv2.rectangle(viz_image, (x, y), (x+w, y+h), color, 2)
                
                # Draw tags if present
                if 'tags' in region:
                    tag_text = ', '.join(region['tags'])
                    cv2.putText(viz_image, tag_text, (x, y-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
            elif region_type == 'polygon':
                points = np.array(region['points'], dtype=np.int32)
                cv2.polylines(viz_image, [points], True, color, 2)
                
                # Draw tags if present
                if 'tags' in region and len(points) > 0:
                    # Place tag text near the first point
                    x, y = points[0]
                    tag_text = ', '.join(region['tags'])
                    cv2.putText(viz_image, tag_text, (x, y-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
            elif region_type == 'brush' or region_type == 'freeform':
                if 'path' in region:
                    points = np.array(region['path'], dtype=np.int32)
                    thickness = region.get('thickness', 1)
                    cv2.polylines(viz_image, [points], False, color, thickness)
    
    # Draw global tags if present
    if annotations and 'tags' in annotations:
        global_tags = ', '.join(annotations['tags'])
        cv2.putText(viz_image, f"Global: {global_tags}", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save visualization if path provided
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))
        print(f"Saved annotation visualization to {output_path}")
    
    # Show visualization if requested
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(viz_image)
        plt.axis('off')
        plt.title("Image with Annotations")
        plt.show()
    
    return viz_image

def load_image(image_path, size=(128, 128)):
    """
    Load an image (JPG, PNG, or SVG) and convert it to a numpy array.
    
    Args:
        image_path: Path to the image file
        size: Tuple (height, width) for the desired image size
        
    Returns:
        Numpy array with shape (height, width, 1) containing the image data
    """
    # Check if the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")
    
    # Handle based on file extension
    file_ext = os.path.splitext(image_path)[1].lower()
    
    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        # For raster image formats (JPG, PNG, etc.)
        return load_raster_image(image_path, size)
    elif file_ext == '.svg':
        # For SVG images
        if CAIRO_AVAILABLE:
            return load_svg_image(image_path, size)
        else:
            print(f"Warning: Cannot load SVG file {image_path} without CairoSVG.")
            # Use simplified SVG loader as a fallback
            return load_svg_simplified(image_path, size)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def preprocess_reference_image(reference_path, output_dir, size=None, show_preview=True, preserve_dimensions=True, load_annotations=True):
    """
    Preprocess the reference image for training.
    
    This function:
    1. Loads the reference image
    2. Converts it to grayscale if it's not already
    3. Checks for and loads associated annotation files
    4. Saves a visualization of the processed image (with annotations if available)
    
    Args:
        reference_path: Path to the reference image
        output_dir: Directory to save processed outputs
        size: Desired size for the processed image as (width, height) tuple or single int for square.
             If None, uses original dimensions.
        show_preview: Whether to save a preview visualization
        preserve_dimensions: Whether to preserve the original aspect ratio
        load_annotations: Whether to look for annotation files
        
    Returns:
        Tuple of (processed_image, original_dimensions, annotations) where:
            - processed_image is a numpy array with shape (height, width, 1)
            - original_dimensions is a tuple (height, width)
            - annotations is a dictionary with annotation data (or None if no annotations found)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect original image dimensions
    original_dimensions = get_image_dimensions(reference_path)
    print(f"Original reference image dimensions: {original_dimensions[0]}x{original_dimensions[1]}")
    
    # If size is not provided, use original dimensions
    if size is None:
        size = original_dimensions
        print(f"Using original dimensions for training: {size[0]}x{size[1]}")
    elif isinstance(size, int):
        # Single integer means a square image
        size = (size, size)
    elif isinstance(size, tuple) and len(size) == 2:
        # If either dimension is None, use original dimension
        width, height = size
        if width is None:
            width = original_dimensions[1]
        if height is None:
            height = original_dimensions[0]
        size = (height, width)  # Note: size is (height, width) for consistency with numpy arrays
        
    print(f"Processing reference image to dimensions: {size[0]}x{size[1]}")
    
    # Check if it's an SVG file
    file_ext = os.path.splitext(reference_path)[1].lower()
    if file_ext == '.svg':
        # Try our simplified SVG loader first
        print(f"Processing SVG file: {reference_path}")
        processed_image = load_svg_simplified(reference_path, size)
    else:
        # Use standard image loading for other formats
        processed_image = load_image(reference_path, size)
    
    # Look for annotation file if requested
    annotations = None
    if load_annotations:
        annotation_path = find_annotation_file(reference_path)
        if annotation_path:
            annotations = load_annotations(annotation_path)
            print(f"Loaded annotations for {os.path.basename(reference_path)}")
            
            # Visualize annotations if preview is requested
            if show_preview and annotations:
                viz_path = os.path.join(output_dir, 'reference_with_annotations.png')
                visualize_annotations(processed_image, annotations, output_path=viz_path)
    
    if show_preview:
        try:
            # Save the processed reference image for visualization
            preview_path = os.path.join(output_dir, 'reference_processed.png')
            fig = plt.figure(figsize=(6, 6))
            plt.imshow(processed_image[:, :, 0], cmap='gray')
            plt.title("Processed Reference Image")
            plt.axis('off')
            plt.savefig(preview_path)
            plt.close(fig)  # Explicitly close the figure to prevent memory leaks
            print(f"Saved processed reference preview to {preview_path}")
        except Exception as e:
            print(f"Error saving preview: {e}")
            plt.close('all')  # Ensure all figures are closed in case of error
    
    # Save as SVG for comparison with generated outputs
    svg_path = os.path.join(output_dir, 'reference_processed.svg')
    save_as_svg(processed_image, svg_path)
    print(f"Saved processed reference as SVG to {svg_path}")
    
    return processed_image, original_dimensions, annotations