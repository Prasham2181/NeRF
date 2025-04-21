# Neural Radiance Fields (NeRF) Implementation


## Project Overview
This project implements Neural Radiance Fields (NeRF), a neural network-based method for synthesizing novel views of complex 3D scenes. NeRF works by optimizing a continuous volumetric scene function from a set of input images, allowing for high-quality view synthesis.

## Features
- Neural network that maps 5D coordinates (3D position + 2D viewing direction) to volume density and RGB color
- Hierarchical sampling strategy for more efficient rendering
- Ray sampling and volumetric rendering implementation
- Training pipeline with customizable hyperparameters
- Validation and testing modules with metrics calculation (MSE, PSNR, SSIM)
- Novel view synthesis capabilities
- Support for rendering spiral camera paths for video creation
- 
## Team Members
- Prasham Soni
- Sarthak Mehta
## Output for the Project

![NeRF Output](https://raw.githubusercontent.com/Prasham2181/NeRF/main/NeRF.gif)

## Dependencies
- PyTorch
- NumPy
- Matplotlib
- OpenCV
- scikit-image
- tqdm

## Project Structure
```
Code
├── train.py            # Training script for the NeRF model
├── Wrapper.py          # Validation and testing utilities
├── Network.py          # Neural network architecture
├── Dataset.py          # Dataset loader for NeRF
├── ray_sampling.py     # Ray generation and sampling functions
├── Render.py           # Volumetric rendering implementation
├── checkpoints/        # Directory for model checkpoints
└──NeRF.gif             # Gif Output for Legoset
REPORT  
```

## How to Run

### Training
To train the NeRF model from scratch:

```bash
python train.py
```

The training script uses the following default hyperparameters:
- Number of epochs: 30000
- Batch size: 1 (image)
- Number of rays sampled per image: 1024
- Learning rate: 5e-4
- Near bound: 2.0
- Far bound: 6.0
- Samples per ray: 192

### Validation and Testing
To validate and test the trained model:

```bash
python Wrapper.py --checkpoint /path/to/checkpoint --data_path /path/to/dataset
```

Additional options:
```bash
python Wrapper.py --help
```

### Rendering Novel Views
To render a novel view from a specific camera position:

```bash
python Wrapper.py --checkpoint /path/to/checkpoint --data_path /path/to/dataset --novel_view --theta 45 --phi -30 --radius 4
```

### Creating Spiral Path Rendering for Video
To create frames for a spiral path video:

```bash
python Wrapper.py --checkpoint /path/to/checkpoint --data_path /path/to/dataset --create_spiral --n_frames 60
```

## Model Architecture
The NeRF model consists of:
- Position encoding for 3D coordinates and viewing directions
- MLP network that maps position and direction to density and color
- Hierarchical sampling with coarse and fine networks for better rendering quality

## Results
The model produces high-quality novel view syntheses of 3D scenes from a set of input images. Performance is measured using:
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)

Example renderings and metrics can be found in the `logs/media/` directory after running the testing script.

## Dataset
The code is configured to work with the standard NeRF synthetic datasets. The default path is set for the "lego" dataset. To use a different dataset, specify the path using the `--data_path` argument.

## Implementation Details
- The implementation follows the original NeRF paper with position encoding and hierarchical volume sampling
- Ray origins and directions are calculated from camera poses and focal lengths
- Volume rendering is performed by integrating density and color along each ray
- Training uses MSE loss between rendered and ground truth pixels
- Hierarchical sampling is used with both coarse and fine sampling to allocate more samples to regions with higher expected density

## Future Work
- Implement NeRF in the Wild for real-world scenes
- Add support for larger scenes with unbounded rendering
- Integrate Instant-NGP for faster training and rendering
- Add support for dynamic scenes
