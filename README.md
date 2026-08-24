# Dog vs Cat CNN

A Python CNN that classifies images as dogs or cats.

## Overview

This repository implements a convolutional neural network (CNN) for binary image classification of dogs and cats. The project is organized as a single monolithic codebase that follows a classic batch‑processing ML pipeline: raw images are loaded and pre‑processed, the CNN architecture is defined, and a training engine runs the training loop, evaluates performance, and persists the best model. An accompanying Jupyter notebook provides interactive exploration of the data and model behaviour.

## Features

- Model Definition: `src/convolutional_neural_network.py` contains a Keras/TensorFlow based CNN architecture with configurable layers and hyper‑parameters.
- Data Preparation: Functions to load image datasets, apply resizing, normalization, and optional data augmentation (random flips, rotations, zoom).
- Training Engine: End‑to‑end training loop with batch processing, validation, early stopping, model checkpointing, and final evaluation metrics.
- Model Persistence: Trained weights are saved to `models/` and can be re‑loaded for inference.
- Exploratory Notebook: `src/notebooks/convolutional_neural_network.ipynb` demonstrates data visualization, model training progress, and prediction examples.

## Quick Start

```bash
```bash
# Clone the repository
git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
cd dog-vs-cat-cnn

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the training script (default uses the 'data/' folder)
python -m src.convolutional_neural_network

# Launch the exploratory notebook
jupyter notebook src/notebooks/convolutional_neural_network.ipynb
```
```

## Architecture

Monolithic Python package that implements a batch‑processing ML pipeline: (1) Data Preparation module reads and augments image files, (2) Model Definition module builds a Keras CNN, (3) Training Engine orchestrates epoch‑wise training, validation, checkpointing, and model saving, all executed from a single entry‑point script. The Jupyter notebook sits on top for interactive analysis.

---
*This file is kept in sync by [AutoScribe](https://github.com) — edits here may be overwritten on the next sync.*
