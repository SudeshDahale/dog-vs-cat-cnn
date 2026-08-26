# Dog vs Cat CNN

A simple convolutional neural network to classify dog and cat images.

## Overview

This repository implements a complete end‑to‑end pipeline for binary image classification between dogs and cats. It includes data preprocessing (loading, resizing, normalizing, and train/validation split), a CNN model definition and training script, and evaluation/inference utilities. All components are organized under a monolithic Python codebase, with a Jupyter notebook for exploratory analysis.

The project is built with Python and relies on common ML libraries such as TensorFlow/Keras, NumPy, and scikit‑learn. The source code lives in the `src/` directory, and the notebook provides a convenient interactive view of the workflow.

## Features

- Data preprocessing module that reads raw images, resizes them to a uniform shape, normalizes pixel values, and splits the dataset into training and validation sets.
- Convolutional neural network architecture defined in `src/convolutional_neural_network.py` with configurable hyper‑parameters (layers, filters, dropout, optimizer, etc.).
- Training script that fits the model on the preprocessed data, saves model checkpoints, and logs training metrics.
- Evaluation utilities to compute accuracy, loss, and generate confusion matrices on the validation set.
- Inference function to predict the class of new, unseen images with a single command.
- Jupyter notebook (`src/notebooks/convolutional_neural_network.ipynb`) that demonstrates the full pipeline step‑by‑step, including visualizations of data and training curves.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
cd dog-vs-cat-cnn

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Prepare the data (assumes images are placed in `data/`)
python -c "from src.convolutional_neural_network import prepare_data; prepare_data()"

# Train the model
python src/convolutional_neural_network.py --mode train

# Evaluate the trained model
python src/convolutional_neural_network.py --mode evaluate

# Run inference on a new image
python src/convolutional_neural_network.py --mode infer --image_path path/to/image.jpg
```

## Architecture

The project follows a monolithic ML‑pipeline architecture. All stages—data preprocessing, model definition, training, evaluation, and inference—are implemented within the `src/` package, enabling a straightforward linear flow from raw images to predictions. The Jupyter notebook mirrors this pipeline for interactive exploration.

---
*This file is kept in sync by [AutoScribe](https://github.com) — edits here may be overwritten on the next sync.*
