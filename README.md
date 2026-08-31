# Dog vs Cat CNN

A lightweight convolutional neural network for binary dog‑cat image classification.

## Overview

This repository implements a simple yet effective convolutional neural network (CNN) that distinguishes between dog and cat images. The codebase is organized as a monolithic Python project that performs batch processing of image datasets for training and evaluation. An accompanying Jupyter notebook provides an interactive environment for exploratory data analysis, model visualization, and step‑by‑step experimentation.

## Features

- Defines a custom CNN architecture in `src/convolutional_neural_network.py` tailored for binary classification.
- End‑to‑end training pipeline that loads images, preprocesses them, trains the model, evaluates validation accuracy, and persists the trained weights.
- Configurable hyperparameters (learning rate, batch size, epochs) via command‑line arguments or a simple config section.
- Comprehensive Jupyter notebook (`src/notebooks/convolutional_neural_network.ipynb`) for data exploration, training visualizations, and quick prototyping.
- Dependency management through `requirements.txt` for reproducible environments.

## Quick Start

```bash
```bash
# Clone the repository
git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
cd dog-vs-cat-cnn

# Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # on Windows use `venv\Scripts\activate`
pip install --upgrade pip
pip install -r requirements.txt

# (Optional) Launch the exploratory notebook
jupyter notebook src/notebooks/convolutional_neural_network.ipynb

# Train the model (adjust arguments as needed)
python src/convolutional_neural_network.py --data_dir path/to/dataset --epochs 20 --batch_size 32 --learning_rate 0.001
```
```

## Architecture

The project follows a monolithic structure with three primary components: (1) **Model Definition** – a Python module that builds the CNN layers using PyTorch/TensorFlow (as declared in `requirements.txt`); (2) **Training & Evaluation** – scripts that perform batch loading of image data, execute the training loop, compute validation metrics, and serialize the model weights; (3) **Exploratory Notebook** – an interactive Jupyter notebook that imports the same model code, enabling step‑wise execution, visual loss curves, and sample predictions. All components reside under the `src/` directory, keeping the codebase self‑contained for straightforward execution.

---
*This file is kept in sync by [AutoScribe](https://github.com) — edits here may be overwritten on the next sync.*
