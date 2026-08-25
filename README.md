# dog-vs-cat-cnn

A simple convolutional neural network for classifying dog and cat images.

## Overview

This repository provides a lightweight implementation of a convolutional neural network (CNN) that distinguishes between dog and cat images. The core model and training logic live in `src/convolutional_neural_network.py`, while an accompanying Jupyter notebook (`src/notebooks/convolutional_neural_network.ipynb`) offers an interactive environment for experimenting with the model, visualizing training curves, and testing predictions.

## Features

- Defines a CNN architecture using only standard Python ML libraries (TensorFlow/Keras or PyTorch).
- Training script that loads image data, applies augmentation, and reports accuracy and loss.
- Utility functions for preprocessing images and saving/loading model checkpoints.
- Jupyter notebook UI for step‑by‑step experimentation, visualizing sample images, training progress, and inference on custom images.

## Quick Start

```bash
git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
cd dog-vs-cat-cnn
python -m venv venv
source venv/bin/activate  # on Windows use `venv\Scripts\activate`
pip install -r requirements.txt
python src/convolutional_neural_network.py  # runs training with default settings
# Optional: launch the notebook for interactive exploration
jupyter notebook src/notebooks/convolutional_neural_network.ipynb
```

## Architecture

Monolithic Python package focused on ML training. The `src` directory houses all model definitions, training loops, and utility code, while the notebook provides a UI layer on top of the same codebase, keeping the entire project self‑contained.

---
*This file is kept in sync by [AutoScribe](https://github.com) — edits here may be overwritten on the next sync.*
