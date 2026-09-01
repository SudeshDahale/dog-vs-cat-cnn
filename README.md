# Dog vs Cat CNN

A simple CNN prototype for dog vs cat image classification.

## Overview

This repository provides a monolithic, prototype‑style implementation of a convolutional neural network (CNN) that distinguishes between dogs and cats. The codebase is organized under a single `src` package and includes data preparation, model definition, training, evaluation, and inference utilities. A Jupyter notebook demonstrates end‑to‑end usage, while `requirements.txt` lists the necessary Python dependencies.

## Features

- Defines a CNN architecture in `src/convolutional_neural_network.py` using TensorFlow/Keras.
- Data loading, preprocessing, and augmentation pipelines for the dog vs cat dataset.
- Training script with configurable hyper‑parameters and checkpointing.
- Evaluation utilities to compute accuracy, loss, and confusion matrix on a validation set.
- Inference helper to classify new images from the command line or notebook.
- Jupyter notebook (`src/notebooks/convolutional_neural_network.ipynb`) that walks through the full workflow from data preparation to model inference.

## Quick Start

```bash
```bash
# Clone the repository
git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
cd dog-vs-cat-cnn

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run training (adjust arguments as needed)
python -m src.convolutional_neural_network train --epochs 20 --batch-size 32

# Evaluate the trained model
python -m src.convolutional_neural_network evaluate --checkpoint path/to/checkpoint.h5

# Perform inference on a new image
python -m src.convolutional_neural_network predict --image examples/dog1.jpg

# Optional: explore the notebook
jupyter notebook src/notebooks/convolutional_neural_network.ipynb
```
```

## Architecture

The project follows a monolithic, prototype‑style architecture: all core components—data preparation, model definition, training, evaluation, and inference—reside within the `src` package. The main entry point (`src/convolutional_neural_network.py`) uses sub‑functions to orchestrate each stage, making the workflow linear and easy to follow for experimentation.

---
*This file is kept in sync by [AutoScribe](https://github.com) — edits here may be overwritten on the next sync.*
