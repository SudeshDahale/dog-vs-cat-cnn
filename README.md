# Dog vs Cat CNN

A Python monolithic CNN for classifying dog and cat images in batch mode.

## Overview

This repository implements a convolutional neural network (CNN) that classifies images as either dogs or cats. The project is organized as a single Python monolith that processes datasets in batch, providing a model definition, a training script, and a Jupyter notebook for exploratory analysis and visualisation.

## Features

- Model definition in `src/convolutional_neural_network.py` using PyTorch/Keras (as specified in requirements).
- Batch‑processing training loop that loads the image dataset, trains the CNN, and evaluates performance.
- Fully reproducible experiments via the Jupyter notebook `src/notebooks/convolutional_neural_network.ipynb`.
- All dependencies captured in `requirements.txt` for easy environment recreation.

## Quick Start

```bash
```bash
# Clone the repository
git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
cd dog-vs-cat-cnn

# Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate   # on Windows use `venv\Scripts\activate`
pip install -r requirements.txt

# Run the training script (batch processing)
python src/convolutional_neural_network.py

# Launch the exploratory notebook
jupyter notebook src/notebooks/convolutional_neural_network.ipynb
```
```

## Architecture

The project follows a monolithic, batch‑processing architecture. All code resides under the `src/` package: `convolutional_neural_network.py` holds the model definition and training loop, while the Jupyter notebook provides a separate entry point for interactive exploration. Data is loaded, transformed, and fed to the model in large batches, making the workflow suitable for offline training on a static image dataset.

---
*This file is kept in sync by [AutoScribe](https://github.com) — edits here may be overwritten on the next sync.*
