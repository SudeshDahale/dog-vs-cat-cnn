# Dog vs Cat CNN

A Python monolithic CNN pipeline for classifying dog and cat images.

## Overview

This repository implements a complete end‑to‑end workflow for training a convolutional neural network (CNN) to distinguish between dog and cat images. It follows a monolithic ML‑workflow architecture: data preprocessing, model definition/training, and evaluation are each encapsulated in dedicated modules under `src/`, with an accompanying Jupyter notebook for interactive exploration and visualization.

## Features

- Loads and preprocesses raw image datasets for training and validation (data‑prep module).
- Defines a configurable CNN architecture using Keras/TensorFlow (cnn‑model module).
- Trains the model with support for early stopping and checkpointing, saving the best weights.
- Evaluates model performance on a hold‑out test set, producing accuracy, loss curves, and confusion matrices (evaluation module).
- Generates visualizations of training history and sample predictions.
- Provides an interactive Jupyter notebook (`src/notebooks/convolutional_neural_network.ipynb`) for step‑by‑step experimentation.

## Quick Start

```bash
```bash
# Clone the repository
git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
cd dog-vs-cat-cnn

# Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install --upgrade pip
pip install -r requirements.txt

# Run the end‑to‑end pipeline (data prep, training, evaluation)
python src/convolutional_neural_network.py

# Or launch the notebook for interactive work
jupyter notebook src/notebooks/convolutional_neural_network.ipynb
```
```

## Architecture

The project follows a monolithic ML‑workflow architecture. The `src/` package contains three primary modules: `data-prep` for loading and preprocessing images, `cnn-model` for defining and training the convolutional network, and `evaluation` for assessing performance and generating visual reports. All modules share a common configuration and are orchestrated by the main script `convolutional_neural_network.py`. The Jupyter notebook mirrors this workflow, providing a step‑wise, visual interface for developers and researchers.

---
*This file is kept in sync by [AutoScribe](https://github.com) — edits here may be overwritten on the next sync.*
