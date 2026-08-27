# Dog vs Cat CNN

A simple Convolutional Neural Network for binary classification of dog and cat images.

## Overview

This repository implements a complete end‑to‑end machine‑learning pipeline for distinguishing dog and cat images. The project is organized as a monolithic Python package under `src/` that handles data loading, preprocessing, model definition, training, and evaluation. All steps are reproducible via a single script or interactive Jupyter notebook, making it easy to experiment with hyper‑parameters or extend the model.

The core components are:
* **Data Loading & Preprocessing** – Reads image files, resizes them to a uniform shape, normalizes pixel values, and applies optional data augmentation.
* **Model Definition** – Constructs a Keras `Sequential` CNN model with convolutional, pooling, and dense layers tailored for binary classification.
* **Training & Evaluation** – Trains the network, logs loss/accuracy, and evaluates the final model on a held‑out test set.

The repository also includes a Jupyter notebook (`src/notebooks/convolutional_neural_network.ipynb`) that walks through each stage interactively.

## Features

- Loads image datasets from a directory structure (e.g., `data/train/dog`, `data/train/cat`).
- Applies resizing, pixel‑value scaling, and optional augmentation (random flip, rotation) to improve generalization.
- Defines a Keras Sequential CNN with configurable number of convolutional blocks and dropout layers.
- Tracks training progress with loss and accuracy metrics and saves the best model checkpoint.
- Evaluates the trained model on a separate test split and reports classification accuracy.
- Provides a ready‑to‑run script (`src/convolutional_neural_network.py`) and an accompanying notebook for interactive exploration.

## Quick Start

```bash
```bash
# Clone the repository
git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
cd dog-vs-cat-cnn

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt

# Run the training script (default expects data under ./data)
python src/convolutional_neural_network.py

# Or launch the notebook for step‑by‑step execution
jupyter notebook src/notebooks/convolutional_neural_network.ipynb
```
```

## Architecture

The project follows a monolithic ML‑pipeline architecture where all stages—data ingestion, preprocessing, model building, training, and evaluation—reside in the `src/` package. The pipeline is orchestrated by `convolutional_neural_network.py`, which sequentially calls utility functions for each stage. The notebook mirrors this flow, providing a visual and interactive representation of the same pipeline.

---
*This file is kept in sync by [AutoScribe](https://github.com) — edits here may be overwritten on the next sync.*
