# Dog vs Cat CNN

A monolithic Python project that trains a convolutional neural network to classify dog and cat images.

## Overview

This repository provides a complete end‑to‑end pipeline for binary image classification of dogs vs. cats. It ingests and preprocesses image data, defines a CNN model using PyTorch, and runs batch‑processed training and evaluation loops. All code lives under a single `src/` package, making it simple to run the entire workflow from the command line or a Jupyter notebook.

## Features

- Data ingestion and preprocessing with automatic resizing, normalization, and train/validation split.
- CNN model definition using PyTorch, including configurable layers and activation functions.
- Training loop with batch processing, loss/accuracy monitoring, and model checkpoint saving.
- Evaluation script that reports final validation accuracy and visualizes sample predictions.
- Jupyter notebook (`src/notebooks/convolutional_neural_network.ipynb`) demonstrating the full workflow step‑by‑step.

## Quick Start

```bash
```bash
# Clone the repository
git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
cd dog-vs-cat-cnn

# Set up a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the training script (default will download data, train, and evaluate)
python src/convolutional_neural_network.py
```

```

## Architecture

The project follows a monolithic, batch‑processing architecture. All components—data handling, model definition, training, and evaluation—are implemented as Python modules within the `src/` directory and are executed sequentially in a single process. Data is loaded once, preprocessed in memory, and then fed to the model in batches during training, making the workflow straightforward and reproducible.

---
*This file is kept in sync by [AutoScribe](https://github.com) — edits here may be overwritten on the next sync.*
