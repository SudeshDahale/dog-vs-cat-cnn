# Dog vs Cat CNN

A simple convolutional neural network that classifies images as dogs or cats.

## Overview

This repository implements a binary image classifier using a convolutional neural network (CNN) built with TensorFlow/Keras. The codebase is organized as a single monolithic Python package under the src directory, containing a data‑loading pipeline, the model definition/training script, and a Jupyter notebook for exploratory analysis. All required dependencies are listed in requirements.txt, making the project easy to set up and run locally.

## Features

- Loads and preprocesses cat and dog image datasets with automatic resizing, normalization, and train/validation split.
- Defines a configurable CNN architecture (conv layers, pooling, dropout, dense output) in src/convolutional_neural_network.py.
- Trains the model with checkpointing and reports accuracy/loss on a validation set.
- Provides an evaluation routine to predict on new images and compute classification metrics.
- Includes src/notebooks/convolutional_neural_network.ipynb for interactive visualisation of training curves, sample predictions, and model debugging.

## Quick Start

```bash
git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git && cd dog-vs-cat-cnn && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && python src/convolutional_neural_network.py
```

## Architecture

The project follows a monolithic architecture where all components live under the src folder. The data loader prepares image tensors, which are fed into the CNN model defined in convolutional_neural_network.py. Training, validation, and inference logic are encapsulated in the same script, while the notebook offers a separate exploratory interface that imports the same modules for reproducible experimentation.

---
*This file is kept in sync by [AutoScribe](https://github.com) — edits here may be overwritten on the next sync.*
