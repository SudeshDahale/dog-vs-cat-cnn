# Dog vs Cat CNN

A lightweight Python CNN for binary classification of dog and cat images.

## Overview

This repository implements a complete end‑to‑end pipeline for training, evaluating, and experimenting with a convolutional neural network that distinguishes dogs from cats. The project is organized as a single monolithic codebase under `src/`, with distinct logical sections for data preprocessing, model definition, training, and evaluation. An accompanying Jupyter notebook demonstrates interactive inference and visualisation of results.

## Features

- Data preprocessing that loads images, resizes them to a uniform shape, normalises pixel values, and creates reproducible train/validation splits.
- A handcrafted CNN architecture defined in `src/convolutional_neural_network.py` using PyTorch, tailored for binary classification.
- Training pipeline that handles loss calculation, optimizer updates, learning‑rate scheduling, checkpoint saving, and metric logging.
- Evaluation utilities for computing accuracy on a held‑out set and a Jupyter notebook (`src/notebooks/convolutional_neural_network.ipynb`) for interactive inference and result visualisation.

## Quick Start

```bash
```bash
# Clone the repository
git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
cd dog-vs-cat-cnn

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # on Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the training script (adjust paths or hyper‑parameters inside the script as needed)
python src/convolutional_neural_network.py

# Launch the Jupyter notebook for interactive inference
jupyter notebook src/notebooks/convolutional_neural_network.ipynb
```
```

## Architecture

The project follows a monolithic, linear ML pipeline: data preprocessing → model definition → training loop → evaluation/inference. All components reside in the `src/` package, enabling straightforward execution from a single entry point while keeping the logical stages clearly separated in the source code.

---
*This file is kept in sync by [AutoScribe](https://github.com) — edits here may be overwritten on the next sync.*
