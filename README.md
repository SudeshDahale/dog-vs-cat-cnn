# Dog vs Cat CNN

A Python monolithic implementation of a convolutional neural network for binary image classification (dogs vs cats).

## Overview

This repository provides a self‑contained Python project that defines, trains, and evaluates a convolutional neural network (CNN) to distinguish between dog and cat images. The core logic lives in the `src` package, with a single module `convolutional_neural_network.py` that includes the model architecture, training loop, and evaluation utilities. An accompanying Jupyter notebook demonstrates usage and visualizes results.

The project is structured as a monolith: all code, dependencies, and configuration are packaged together, making it easy to clone, install, and run end‑to‑end without additional services.

## Features

- CNN model definition using TensorFlow/Keras for binary classification
- Training routine with configurable epochs, batch size, and learning rate
- Evaluation utilities reporting accuracy, loss curves, and confusion matrix
- Utility functions for data loading, preprocessing, and augmentation
- Jupyter notebook (`src/notebooks/convolutional_neural_network.ipynb`) that walks through data exploration, model training, and result visualization

## Quick Start

```bash
```bash
# Clone the repository
git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
cd dog-vs-cat-cnn

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required Python packages
pip install -r requirements.txt

# Run the training script (modify arguments as needed)
python src/convolutional_neural_network.py --data_dir path/to/dataset --epochs 20 --batch_size 32

# Or launch the notebook to interactively explore the model
jupyter notebook src/notebooks/convolutional_neural_network.ipynb
```
```

## Architecture

The project follows a monolithic architecture: all functionality resides in a single Python package under `src`. The primary module `convolutional_neural_network.py` encapsulates model definition, training, and evaluation logic. No external services or micro‑service boundaries are required; the code runs as a stand‑alone script or within a notebook, directly accessing the local image dataset.

---
*This file is kept in sync by [AutoScribe](https://github.com) — edits here may be overwritten on the next sync.*
