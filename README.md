# Dog vs Cat CNN

A simple convolutional neural network for classifying dog and cat images.

## Overview

This repository provides a lightweight, end‑to‑end implementation of a convolutional neural network (CNN) that distinguishes between dog and cat images. The codebase follows a monolithic ML‑pipeline structure: data loading and preprocessing utilities, model definition and training logic, and an interactive Jupyter notebook for experimentation and visualising training metrics. All components live under the `src/` package, making it easy to run the full training loop or explore the model step‑by‑step.

## Features

- Python implementation of a CNN architecture tailored for binary image classification.
- Data loading module that reads image files, resizes them, and applies standard augmentations (normalisation, random flip, rotation).
- Training script (`src/convolutional_neural_network.py`) that manages the training loop, validation, checkpointing, and metric logging.
- Jupyter notebook (`src/notebooks/convolutional_neural_network.ipynb`) for interactive exploration, visualising loss/accuracy curves, and quick inference on sample images.
- Requirements file pinning all necessary libraries (TensorFlow/Keras, NumPy, pandas, matplotlib, etc.).

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

# Run the training script (default dataset path expects `data/` folder with `train/` and `val/` subfolders)
python -m src.convolutional_neural_network

# Launch the experiment notebook
jupyter notebook src/notebooks/convolutional_neural_network.ipynb
```
```

## Architecture

The project is organised as a single monolithic Python package (`src/`).
- **Data Loading & Preprocessing** – Functions read image files from a directory tree, resize to a fixed shape, normalise pixel values, and optionally apply augmentations. This module supplies `tf.data.Dataset` objects used by the trainer.
- **Model Definition & Training** – The CNN architecture (convolutional layers, pooling, dropout, and dense head) is defined using TensorFlow/Keras. A training loop orchestrates epoch iteration, loss computation, back‑propagation, validation, and model checkpoint saving.
- **Experiment Notebook** – A Jupyter notebook imports the same modules, reproduces the training pipeline, and adds visualisation cells for loss/accuracy plots, confusion matrices, and sample predictions.
All three layers share the same codebase, ensuring reproducibility between script‑based runs and notebook experiments.

---
*This file is kept in sync by [AutoScribe](https://github.com) — edits here may be overwritten on the next sync.*
