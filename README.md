# Dog vs Cat CNN

A simple Convolutional Neural Network for classifying dog and cat images using TensorFlow/Keras.

## Overview

This repository provides a minimal end‑to‑end example of training a convolutional neural network to distinguish dogs from cats. The core model logic lives in a Python script under `src/`, while an interactive Jupyter notebook under `src/notebooks/` walks through data loading, model construction, training, and visualisation of results. All dependencies are listed in `requirements.txt`, making the project easy to set up and run locally.

## Features

- CNN architecture defined in pure TensorFlow/Keras with configurable layers and hyper‑parameters.
- Training pipeline that loads the classic dog‑vs‑cat image dataset, preprocesses images, and saves trained model weights.
- Jupyter notebook for step‑by‑step exploration, including data inspection, model training progress, and prediction visualisation.
- Requirements file pinning exact library versions for reproducible results.
- Model checkpoint saving for later inference or further fine‑tuning.

## Quick Start

```bash
```bash
# Clone the repository
git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
cd dog-vs-cat-cnn

# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the training script (adjust dataset path as needed)
python src/convolutional_neural_network.py

# Or launch the interactive notebook
jupyter notebook src/notebooks/convolutional_neural_network.ipynb
```
```

## Architecture

The project follows a monolithic ML‑pipeline design: `src/convolutional_neural_network.py` contains the model definition, data preprocessing, training loop and weight serialization. The notebook `src/notebooks/convolutional_neural_network.ipynb` imports this script to reuse the same pipeline while providing inline visualisation and interactive experimentation. No separate services or micro‑services are involved; all code runs in a single Python environment.

---
*This file is kept in sync by [AutoScribe](https://github.com) — edits here may be overwritten on the next sync.*
