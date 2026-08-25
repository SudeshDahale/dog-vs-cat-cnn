# Dog vs Cat CNN

A Python monolithic data‑science project that builds and evaluates a convolutional neural network for binary dog‑vs‑cat image classification.

## Overview

This repository contains a self‑contained implementation of a convolutional neural network (CNN) that classifies images as either dogs or cats. The core model logic lives in `src/convolutional_neural_network.py`, while the Jupyter notebook `src/notebooks/convolutional_neural_network.ipynb` provides step‑by‑step exploratory data analysis, model training, validation, and visualisation of results. The project is packaged with a `requirements.txt` file for reproducible dependency management, making it straightforward to set up and run on any machine with Python installed.

## Features

- Fully‑implemented CNN architecture for binary image classification in `src/convolutional_neural_network.py`.
- End‑to‑end workflow notebook (`src/notebooks/convolutional_neural_network.ipynb`) covering data loading, preprocessing, training, evaluation, and result visualisation.
- Explicit dependency list in `requirements.txt` (TensorFlow/Keras, NumPy, pandas, matplotlib, scikit‑learn, etc.).
- Clear separation of concerns: model definition, training utilities, and analysis live in the `src/` package, while the notebook orchestrates the pipeline.
- Reproducible environment setup using virtual environments or Conda.

## Quick Start

```bash
```bash
# Clone the repository
git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
cd dog-vs-cat-cnn

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Launch the notebook to run the full workflow
jupyter notebook src/notebooks/convolutional_neural_network.ipynb
```

```

## Architecture

Monolithic data‑science architecture: all code resides under the `src/` directory, with the Jupyter notebook acting as the orchestrator for the data pipeline. The project does not split into separate services; instead, model definition, training loops, and utilities are tightly coupled in a single codebase, typical for research‑oriented machine‑learning projects.

---
*This file is kept in sync by [AutoScribe](https://github.com) — edits here may be overwritten on the next sync.*
