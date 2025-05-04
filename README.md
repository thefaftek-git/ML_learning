# 2D Image Generator ML Project

This is a basic machine learning project that generates 2D wireframe images. The model starts from scratch without training data and learns to produce a specific SVG image.

## Project Structure

- `src/`: Source code for the image generation model
- `models/`: Saved model checkpoints 
- `data/`: Data including generated images and target SVG

## Setup

1. Install Python 3.8 or higher
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Single Image Training
To train on a single reference image:
```
python src/train.py --reference reference.png
```

### Multi-Image Training
To train on a folder containing multiple reference images:
```
python src/train.py --reference-dir data/training_images
```

The model can use either sequential or random selection when training on multiple images:
```
# Randomly select from the training images each epoch (default)
python src/train.py --reference-dir data/training_images --image-selection random

# Cycle through images sequentially
python src/train.py --reference-dir data/training_images --image-selection sequential
```

When training on multiple images, all images will be automatically resized to a common size to ensure consistent training.

### Generate New Images
Once trained, you can generate new images with:
```
python src/generate.py
```

## Advanced Options

The model now automatically detects and uses the resolution of your reference image:

```
# Use the reference image's native resolution (default behavior):
python src/train.py

# Use a custom resolution (if you prefer):
python src/train.py --image-size 512
```

Other available options:

```
# Use GPU acceleration (if available):
python src/train.py --use-gpu

# Optimize memory usage:
python src/train.py --optimize-memory

# Run more training epochs:
python src/train.py --epochs 5000

# Enable parallel training for speed improvements:
python src/train.py --parallel

# Specify number of parallel workers (default: auto-detect):
python src/train.py --parallel --parallel-workers 4

# Set iterations per parallel batch:
python src/train.py --parallel --parallel-iterations 100

# Maintain aspect ratio when specifying custom dimensions:
python src/train.py --width 512 --height 256 --preserve-aspect
```

Run `python src/train.py --help` for the complete list of options.

## Feature Highlights

- **Training with Multiple Images**: Train on a folder of reference images to produce a more versatile model
- **Image Selection Strategies**: Choose between random or sequential image selection during training
- **Automatic Resolution Detection**: Detects and uses the native resolution of your reference image by default
- **Flexible Dimensions**: Handles arbitrary image dimensions and aspect ratios
- **Aspect Ratio Preservation**: Option to maintain the original aspect ratio when specifying custom dimensions
- **GPU Acceleration**: Supports hardware acceleration for faster training
- **Progress Visualization**: Saves progress images during training to track improvement
- **Parallel Training**: Trains multiple models simultaneously and selects the best performers for continued training

## Aspect Ratio Handling

When specifying custom image dimensions with `--width` and `--height`, you have two options:

1. **Free dimensions**: By default, you can specify any dimensions, but if they don't match the original aspect ratio, you'll see a warning about potential distortion.

2. **Preserve aspect ratio**: Using the `--preserve-aspect` flag ensures the output maintains the same aspect ratio as the reference image. When this flag is set:
   - If both width and height are specified, the larger dimension is prioritized and the other is adjusted accordingly
   - If only width is specified, height is calculated automatically
   - If only height is specified, width is calculated automatically

Example for a reference image with a 2:1 aspect ratio:
```
# Automatically adjusts height to 256 to maintain 2:1 ratio:
python src/train.py --width 512 --preserve-aspect

# Keeps width at 512, regardless of any height value provided:
python src/train.py --width 512 --height 300 --preserve-aspect
```

## Multi-Image Training Details

When training with multiple reference images (`--reference-dir`), the system:

1. Loads all supported image files from the specified directory (PNG, JPG, SVG, etc.)
2. Automatically resizes all images to a common size based on your specifications or the first image
3. Creates a reference grid visualization showing all training images
4. Alternates between images during training based on your selection strategy
5. Creates comparison visualizations showing how the model performs against all reference images

This is particularly useful for:
- Training on variations of a similar object
- Creating models that can generalize across multiple related examples
- Finding a compromise solution between different target images

## Parallel Training

The parallel training feature runs multiple model instances simultaneously and keeps only the best 10% (rounded down) of results from each parallel batch. This approach can significantly speed up training by exploring multiple solutions in parallel.

**Note:** Parallel training may require more total iterations compared to sequential training to achieve equivalent results, as it explores different initialization paths. However, this is often outweighed by the speed improvements from parallelization.

When using parallel training, consider:
- Increasing the total number of epochs (`--epochs`) to ensure sufficient convergence
- Adjusting the number of parallel workers based on your CPU capabilities
- Setting a smaller `--visualization-interval` to track progress more frequently

## Educational Purpose

This project is designed for educational purposes to demonstrate how machine learning models can be trained to generate specific image output without pre-existing training data.