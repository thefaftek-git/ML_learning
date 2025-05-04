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

1. Add your target SVG image to the `data/` directory
2. Run the training script:
   ```
   python src/train.py
   ```
3. Generate new images:
   ```
   python src/generate.py
   ```

### Advanced Options

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
```

Run `python src/train.py --help` for the complete list of options.

## Feature Highlights

- **Automatic Resolution Detection**: Detects and uses the native resolution of your reference image by default
- **Flexible Dimensions**: Handles arbitrary image dimensions and aspect ratios
- **GPU Acceleration**: Supports hardware acceleration for faster training
- **Progress Visualization**: Saves progress images during training to track improvement
- **Parallel Training**: Trains multiple models simultaneously and selects the best performers for continued training

## Parallel Training

The parallel training feature runs multiple model instances simultaneously and keeps only the best 10% (rounded down) of results from each parallel batch. This approach can significantly speed up training by exploring multiple solutions in parallel.

**Note:** Parallel training may require more total iterations compared to sequential training to achieve equivalent results, as it explores different initialization paths. However, this is often outweighed by the speed improvements from parallelization.

When using parallel training, consider:
- Increasing the total number of epochs (`--epochs`) to ensure sufficient convergence
- Adjusting the number of parallel workers based on your CPU capabilities
- Setting a smaller `--visualization-interval` to track progress more frequently

## Educational Purpose

This project is designed for educational purposes to demonstrate how machine learning models can be trained to generate specific image output without pre-existing training data.