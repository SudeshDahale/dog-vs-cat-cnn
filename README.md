# Dog vs Cat CNN

A Python CNN that classifies images as dogs or cats.

## Overview

This repository provides a complete, monolithic implementation of a convolutional neural network (CNN) for binary image classification of dogs and cats. The core training logic lives in `src/convolutional_neural_network.py`, while an accompanying Jupyter notebook (`src/notebooks/convolutional_neural_network.ipynb`) offers step‑by‑step exploration, visualization, and interactive experimentation.

The project is designed for batch‑style machine‑learning workflows: you can train the model on a dataset, evaluate its performance, and export the trained weights for downstream inference.


## Features

- Full CNN architecture (convolution, pooling, fully‑connected layers) implemented in pure Python with TensorFlow/Keras.
- Training loop with configurable hyper‑parameters (learning rate, epochs, batch size).
- Data preprocessing utilities for image resizing, normalization, and train/validation split.
- Model checkpointing and early‑stopping callbacks to prevent over‑fitting.
- Jupyter notebook that visualizes training curves, sample predictions, and model architecture.
- Requirements file pinning all necessary libraries for reproducible environments.

## Quick Start

```bash
```bash
# Clone the repository
git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
cd dog-vs-cat-cnn

# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the training script (provide your dataset path via the --data_dir argument)
python src/convolutional_neural_network.py --data_dir /path/to/dog_cat_dataset

# Launch the exploratory notebook
jupyter notebook src/notebooks/convolutional_neural_network.ipynb
```
```

## Architecture

The project follows a monolithic, batch‑oriented ML architecture. All model definition, data preprocessing, training, and evaluation code reside in a single Python module (`src/convolutional_neural_network.py`). The notebook serves as a thin, interactive wrapper that imports the same module, enabling reproducible experiments without duplicating logic. This design keeps the codebase simple, promotes reuse of the core CNN implementation, and aligns with typical batch training pipelines where a one‑off training run produces a saved model for later inference.

---
*This file is kept in sync by [AutoScribe](https://github.com) — edits here may be overwritten on the next sync.*
