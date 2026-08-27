# Dog vs Cat CNN

A Python monolithic ML pipeline that trains a convolutional neural network to classify images as dog or cat.

## Overview

This repository provides a complete end‑to‑end solution for binary image classification of dogs vs. cats using a custom convolutional neural network. The core training logic lives in `src/convolutional_neural_network.py`, while `src/notebooks/convolutional_neural_network.ipynb` offers an interactive environment for data exploration, model training runs, and visual result analysis. The project is structured as a single monolithic codebase that follows a clear ML pipeline: data loading → preprocessing → model definition → training → evaluation → model persistence.

## Features

- Modular CNN definition with configurable layers and hyper‑parameters
- Integrated data preprocessing pipeline for image resizing and normalization
- Training loop with real‑time loss tracking and early stopping support
- Model checkpointing to `models/` directory for later inference
- Evaluation utilities that compute accuracy, precision, recall, and F1 score
- Jupyter notebook that reproduces the full experiment workflow with visualizations of training curves and sample predictions

## Quick Start

```bash
# Clone the repository
git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
cd dog-vs-cat-cnn

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the training script (adjust paths to your dataset if needed)
python src/convolutional_neural_network.py --data_dir path/to/dataset --epochs 20 --batch_size 32

# Launch the exploratory notebook
jupyter notebook src/notebooks/convolutional_neural_network.ipynb
```

## Architecture

The project follows a monolithic architecture where all Python code resides under the `src/` package. `convolutional_neural_network.py` implements the full ML pipeline—data handling, model architecture, training, and evaluation—while the notebook under `src/notebooks/` imports the same modules to ensure reproducibility. This design keeps the pipeline self‑contained, making it easy to run end‑to‑end from a single script or interactively in Jupyter.

---
*This file is kept in sync by [AutoScribe](https://github.com) — edits here may be overwritten on the next sync.*
