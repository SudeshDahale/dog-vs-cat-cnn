# Dog vs Cat CNN

A simple CNN for binary classification of dog and cat images.

## Overview

This monolithic Python project demonstrates how to build, train, and evaluate a convolutional neural network that distinguishes between dog and cat pictures. The workflow is organized under `src/` with separate modules for data preparation, model definition, and training/evaluation, while an interactive Jupyter notebook ties everything together for rapid experimentation and visualisation.

## Features

- Data preparation pipeline that loads images, resizes to a uniform shape, normalises pixel values, and optionally applies augmentation (flip, rotation, zoom).
- A clean `ConvolutionalNeuralNetwork` class in `src/convolutional_neural_network.py` that stacks convolution, ReLU, max‑pooling, and fully‑connected layers for binary classification.
- Training loop with cross‑entropy loss, Adam optimizer, real‑time accuracy reporting, early‑stopping, and model checkpointing of the best validation score.
- Comprehensive evaluation utilities that compute final loss/accuracy and generate confusion matrices and ROC curves.
- Interactive Jupyter notebook (`src/notebooks/convolutional_neural_network.ipynb`) that walks through data loading, model instantiation, training, and visualisation of predictions and metrics.

## Quick Start

```bash
```bash
# Clone the repository
git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
cd dog-vs-cat-cnn

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # on Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the training script (example)
python - <<EOF
from src.convolutional_neural_network import ConvolutionalNeuralNetwork, train_model, evaluate_model
# Adjust paths and hyper‑parameters as needed
train_model(data_dir='data/train', val_dir='data/val', epochs=20, batch_size=32)
EOF

# Or launch the interactive notebook for experimentation
jupyter notebook src/notebooks/convolutional_neural_network.ipynb
```
```

## Architecture

The codebase follows a monolithic architecture: all functionality lives in the `src/` package, with clear logical separation into modules (data preparation, model definition, training/evaluation). The notebook imports these modules, orchestrating the end‑to‑end pipeline without any micro‑service boundaries.

---
*This file is kept in sync by [AutoScribe](https://github.com) — edits here may be overwritten on the next sync.*
