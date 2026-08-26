# Dog vs Cat CNN

A simple Python CNN to classify dog and cat images.

## Overview

This repository implements a convolutional neural network (CNN) for binary image classification of dogs vs. cats. The project is organized as a monolithic ML pipeline in Python, with distinct modules for data preprocessing, model definition, training, and evaluation. An accompanying Jupyter notebook provides an interactive environment for experimentation and visualisation of results.

## Features

- Data preprocessing module that loads, resizes, and normalises cat/dog images for CNN input.
- Convolutional neural network architecture defined in a reusable Python class.
- Training engine that manages epochs, loss computation, optimizer steps, and checkpoint saving.
- Evaluation script to compute accuracy, loss, and generate classification reports on validation/test sets.
- Interactive Jupyter notebook for data exploration, model training, and visualising predictions.

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
pip install -r requirements.txt

# Run the training script (adjust paths/hyper‑parameters as needed)
python src/convolutional_neural_network.py --data_dir ./data --epochs 20 --batch_size 32

# Launch the notebook for interactive exploration
jupyter notebook src/notebooks/convolutional_neural_network.ipynb
```
```

## Architecture

The pipeline is a monolithic Python package where each stage (data‑preprocessing, model‑definition, training‑engine, evaluation) lives in its own module under `src/`. The data‑preprocessing module reads raw images, applies resizing and normalisation, and outputs NumPy arrays or TensorFlow/PyTorch tensors. The model‑definition module encapsulates the CNN architecture. The training‑engine consumes the preprocessed data and the model, runs the training loop, logs metrics, and saves checkpoints. Finally, the evaluation module loads a saved model and reports performance on held‑out data. The Jupyter notebook ties these pieces together for exploratory analysis and visualisation.

---
*This file is kept in sync by [AutoScribe](https://github.com) — edits here may be overwritten on the next sync.*
