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

# Try to import cairosvg but provide a fallback if not available
CAIRO_AVAILABLE = False
try:
    import cairosvg
    CAIRO_AVAILABLE = True
except (ImportError, OSError):
    print("Warning: CairoSVG not available. SVG import functionality will be limited.")
    # Continue without cairosvg

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

def create_blank_image(size=(128, 128)):
    """
    Create a blank white image as a fallback when SVG loading isn't available.
    
    Args:
        size: Tuple (height, width) for the desired image size
        
    Returns:
        Numpy array with shape (height, width, 1) containing a white image
    """
    # Create a white image
    image_array = np.ones((size[0], size[1], 1), dtype=np.float32)
    return image_array

def load_raster_image(image_path, size=(128, 128)):
    """
    Load a raster image (JPG, PNG) and convert it to a numpy array.
    
    Args:
        image_path: Path to the raster image file
        size: Tuple (height, width) for the desired image size
        
    Returns:
        Numpy array with shape (height, width, 1) containing the image data
    """
    # Open the image with PIL and convert to grayscale
    image = Image.open(image_path).convert('L')
    
    # Resize the image to the desired dimensions
    image = image.resize(size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image).astype(np.float32) / 255.0
    
    # Reshape to (height, width, 1)
    image_array = image_array.reshape(size[0], size[1], 1)
    
    return image_array

def load_svg_image(svg_path, size=(128, 128), invert_colors=True):
    """
    Load an SVG image and convert it to a numpy array.
    
    Args:
        svg_path: Path to the SVG file
        size: Tuple (height, width) for the desired image size
        invert_colors: Whether to invert colors for dark SVGs with filled paths
        
    Returns:
        Numpy array with shape (height, width, 1) containing the image data
    """
    # This function requires cairosvg
    if not CAIRO_AVAILABLE:
        print(f"Warning: CairoSVG not available, cannot load SVG file {svg_path}")
        return create_blank_image(size)
    
    # Check if the file exists
    if not os.path.exists(svg_path):
        raise FileNotFoundError(f"SVG file not found at {svg_path}")
    
    try:
        # Read SVG file to check if it contains dark fills
        with open(svg_path, 'r') as file:
            svg_content = file.read()
            
        # Convert SVG to PNG using cairosvg
        png_data = cairosvg.svg2png(url=svg_path, output_width=size[1], output_height=size[0])
        
        # Load the PNG data into a PIL Image
        image = Image.open(io.BytesIO(png_data)).convert('L')  # Convert to grayscale
        
        # Check if SVG contains dark fills by looking at content and image statistics
        if invert_colors and ('fill="#' in svg_content.lower() or 'fill="rgb' in svg_content.lower()):
            # Calculate average brightness
            avg_brightness = np.mean(np.array(image))
            
            # If the image is predominantly dark (filled shapes), invert it to get white shapes on black background
            if avg_brightness < 128:
                print(f"Detected dark fill in SVG {svg_path}, inverting colors for wireframe extraction")
                image = Image.fromarray(255 - np.array(image))
        
        # Convert to numpy array and normalize to [0, 1]
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Reshape to (height, width, 1)
        image_array = image_array.reshape(size[0], size[1], 1)
        
        return image_array
    except Exception as e:
        print(f"Error loading SVG image: {e}")
        return create_blank_image(size)

def load_svg_simplified(svg_path, size=(256, 256)):
    """
    A simplified SVG loader that extracts path data and renders to a numpy array
    without requiring CairoSVG.
    
    Args:
        svg_path: Path to the SVG file
        size: Tuple (height, width) for the desired image size
        
    Returns:
        Numpy array with shape (height, width, 1) containing the image data
    """
    import re
    import math
    from xml.dom import minidom
    
    try:
        print(f"Loading SVG file with simplified loader: {svg_path}")
        # Parse the SVG file
        doc = minidom.parse(svg_path)
        
        # Create a blank canvas
        height, width = size
        canvas = np.ones((height, width), dtype=np.float32)
        
        # Get SVG dimensions
        svg_elem = doc.getElementsByTagName('svg')[0]
        svg_width = float(svg_elem.getAttribute('width') or 1000)
        svg_height = float(svg_elem.getAttribute('height') or 1000)
        print(f"SVG dimensions: {svg_width}x{svg_height}")
        
        # Scaling factors to fit the SVG into our canvas
        scale_x = width / svg_width
        scale_y = height / svg_height
        scale = min(scale_x, scale_y)
        print(f"Using scale factor: {scale} (from {scale_x}, {scale_y})")
        
        # Process all path elements in the SVG
        paths = doc.getElementsByTagName('path')
        print(f"Found {len(paths)} path elements in SVG")
        
        # If we didn't find any paths, return a blank image
        if not paths:
            print(f"No path elements found in SVG: {svg_path}")
            return create_blank_image(size)
        
        # Process each path element
        for path in paths:
            path_str = path.getAttribute('d')
            
            # Extract fill color if present
            fill_color = "#141d27"  # Default fill matching the one in your SVG
            style_attr = path.getAttribute('style')
            if 'fill:' in style_attr:
                fill_match = re.search(r'fill:(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3}|rgb\([^)]+\)|[a-zA-Z]+)', style_attr)
                if fill_match:
                    fill_color = fill_match.group(1)
            elif path.hasAttribute('fill'):
                fill_color = path.getAttribute('fill')
            
            # Determine if we should invert based on fill color (dark fill means we should)
            invert = fill_color.lower() in ['#000000', '#000', 'black', '#141d27']
            fill_value = 0.0 if invert else 1.0
            
            # Extract coordinates from the path
            coords = re.findall(r'([0-9.-]+)[, ]([0-9.-]+)', path_str)
            
            # Draw path points onto canvas
            for i in range(len(coords) - 1):
                x1, y1 = float(coords[i][0]) * scale, float(coords[i][1]) * scale
                x2, y2 = float(coords[i+1][0]) * scale, float(coords[i+1][1]) * scale
                
                # Simple line drawing algorithm
                line_points = bresenham_line(int(x1), int(y1), int(x2), int(y2))
                for x, y in line_points:
                    if 0 <= x < width and 0 <= y < height:
                        # Draw the line
                        canvas[y, x] = fill_value
        
        # Clean up
        doc.unlink()
        
        # Invert the image if we have a dark background
        if np.mean(canvas) < 0.5:
            print(f"Detected dark background in SVG {svg_path}, inverting colors")
            canvas = 1.0 - canvas
            
        # Reshape to match expected format (height, width, 1)
        return canvas.reshape(height, width, 1)
    except Exception as e:
        print(f"Error loading SVG with simplified loader: {e}")
        import traceback
        traceback.print_exc()
        return create_blank_image(size)

def bresenham_line(x0, y0, x1, y1):
    """
    Bresenham's line algorithm to get all points on a line.
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            if x0 == x1:
                break
            err -= dy
            x0 += sx
        if e2 < dx:
            if y0 == y1:
                break
            err += dx
            y0 += sy
    
    return points

def save_as_svg(image_array, output_path, threshold=0.5):
    """
    Convert a numpy array to an SVG wireframe image.
    
    Args:
        image_array: numpy array with shape (height, width, 1) containing the image data
        output_path: Path where to save the SVG file
        threshold: Threshold for binarizing the image (default: 0.5)
    """
    # Get image dimensions
    height, width = image_array.shape[0], image_array.shape[1]
    
    # Create SVG drawing
    dwg = svgwrite.Drawing(output_path, profile='tiny', size=(f"{width}px", f"{height}px"))
    
    # Binarize the image using the threshold
    binary_image = (image_array[:, :, 0] > threshold).astype(np.uint8)
    
    # Find edges using a simple approach
    edges = np.zeros_like(binary_image)
    for i in range(1, height-1):
        for j in range(1, width-1):
            # If the pixel value differs from any of its neighbors, it's an edge
            if binary_image[i, j] != binary_image[i-1, j] or \
               binary_image[i, j] != binary_image[i+1, j] or \
               binary_image[i, j] != binary_image[i, j-1] or \
               binary_image[i, j] != binary_image[i, j+1]:
                edges[i, j] = 1
    
    # Convert edges to line segments
    for i in range(height):
        for j in range(width-1):
            if edges[i, j] == 1 and edges[i, j+1] == 1:
                dwg.add(dwg.line((j, i), (j+1, i), stroke='black', stroke_width=1))
    
    for j in range(width):
        for i in range(height-1):
            if edges[i, j] == 1 and edges[i+1, j] == 1:
                dwg.add(dwg.line((j, i), (j, i+1), stroke='black', stroke_width=1))
    
    # Save the SVG file
    dwg.save()

def create_placeholder_svg(output_path, size=(128, 128), shape_type="circle"):
    """
    Create a simple placeholder SVG file with a basic shape.
    
    Args:
        output_path: Path where to save the SVG file
        size: Tuple (height, width) for the desired image size
        shape_type: Type of shape to create (circle, square, triangle)
    """
    width, height = size
    dwg = svgwrite.Drawing(output_path, profile='tiny', size=(f"{width}px", f"{height}px"))
    
    # Set background
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='white'))
    
    if shape_type == "circle":
        # Draw a circle in the center
        radius = min(width, height) // 4
        dwg.add(dwg.circle(center=(width//2, height//2), r=radius, stroke='black',
                          stroke_width=2, fill='none'))
    
    elif shape_type == "square":
        # Draw a square in the center
        side = min(width, height) // 3
        x = (width - side) // 2
        y = (height - side) // 2
        dwg.add(dwg.rect(insert=(x, y), size=(side, side), stroke='black',
                        stroke_width=2, fill='none'))
    
    elif shape_type == "triangle":
        # Draw a triangle in the center
        side = min(width, height) // 2
        x_center = width // 2
        y_center = height // 2
        dwg.add(dwg.polygon(points=[
            (x_center, y_center - side//2),
            (x_center - side//2, y_center + side//2),
            (x_center + side//2, y_center + side//2)
        ], stroke='black', stroke_width=2, fill='none'))
    
    # Save the SVG file
    dwg.save()
    print(f"Created placeholder SVG at {output_path}")

def preprocess_reference_image(reference_path, output_dir, size=None, show_preview=True, preserve_dimensions=True):
    """
    Preprocess the reference image for training.
    
    This function:
    1. Loads the reference image
    2. Converts it to grayscale if it's not already
    3. Saves a visualization of the processed image
    
    Args:
        reference_path: Path to the reference image
        output_dir: Directory to save processed outputs
        size: Desired size for the processed image as (width, height) tuple or single int for square.
             If None, uses original dimensions.
        show_preview: Whether to save a preview visualization
        preserve_dimensions: Whether to preserve the original aspect ratio
        
    Returns:
        Tuple of (processed_image, original_dimensions) where:
            - processed_image is a numpy array with shape (height, width, 1)
            - original_dimensions is a tuple (height, width)
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
    
    return processed_image, original_dimensions

def get_image_dimensions(image_path):
    """
    Get the original dimensions of an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple (height, width) of the original image dimensions
    """
    # Check if the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")
    
    # Handle based on file extension
    file_ext = os.path.splitext(image_path)[1].lower()
    
    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        # For raster image formats (JPG, PNG, etc.)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                return (height, width)
        except Exception as e:
            print(f"Error getting image dimensions: {e}")
            return (256, 256)  # Default fallback size
    elif file_ext == '.svg':
        # For SVG images
        try:
            from xml.dom import minidom
            doc = minidom.parse(image_path)
            svg_elem = doc.getElementsByTagName('svg')[0]
            
            # Try to get width and height from SVG attributes
            width = svg_elem.getAttribute('width')
            height = svg_elem.getAttribute('height')
            
            # Parse dimensions, removing any units like 'px'
            if width and height:
                width = float(''.join(c for c in width if c.isdigit() or c == '.'))
                height = float(''.join(c for c in height if c.isdigit() or c == '.'))
                doc.unlink()
                return (int(height), int(width))
            
            # If no explicit dimensions, try to find a viewBox
            viewbox = svg_elem.getAttribute('viewBox')
            if viewbox:
                parts = viewbox.split()
                if len(parts) == 4:
                    width = float(parts[2])
                    height = float(parts[3])
                    doc.unlink()
                    return (int(height), int(width))
            
            doc.unlink()
            return (256, 256)  # Default size for SVGs without dimensions
        except Exception as e:
            print(f"Error getting SVG dimensions: {e}")
            return (256, 256)  # Default fallback size
    else:
        # Unsupported format
        print(f"Unsupported format for dimension detection: {file_ext}")
        return (256, 256)  # Default fallback size