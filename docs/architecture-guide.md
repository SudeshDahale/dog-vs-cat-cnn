# Technical Architecture Guide for dog-vs-cat-cnn

## System Overview
The *dog-vs-cat-cnn* repository implements a monolithic Python application for binary image classification (dog vs cat) using a convolutional neural network (CNN). The codebase is organized under a single `src/` package and an accompanying Jupyter notebook for interactive experimentation. Core responsibilities are divided into four logical modules: Data Preparation, CNN Model Definition, Training & Evaluation, and an Interactive Notebook that orchestrates the workflow.

Key entry points are:
- `src/convolutional_neural_network.py` – defines the CNN class, data preprocessing utilities, and training loop.
- `src/notebooks/convolutional_neural_network.ipynb` – loads the module, prepares data, trains the model, and visualizes results.

All dependencies are declared in `requirements.txt` and documented in `README.md`.

## System Layers
### Data Layer
**Technologies:** Python, NumPy, Pillow/OpenCV (image handling)

Handles raw image ingestion, resizing, normalization, and data augmentation. Implemented as utility functions within `src/convolutional_neural_network.py`. Provides NumPy/TensorFlow‑compatible tensors for downstream consumption.

### Model Layer
**Technologies:** Python, TensorFlow/Keras or PyTorch (depending on imports)

Defines the CNN architecture as a Python class. Encapsulates convolutional blocks, pooling layers, dropout (if present), and the final dense layer that outputs a single sigmoid probability for binary classification.

### Training & Evaluation Layer
**Technologies:** Python, TensorFlow/Keras or PyTorch, NumPy

Implements the training loop, loss calculation, optimizer steps, metric tracking, and model checkpointing. Also contains evaluation logic for computing accuracy on a validation split.

### Presentation / Interaction Layer
**Technologies:** Jupyter Notebook, Python, Matplotlib/Seaborn (visualisation)

The Jupyter notebook (`src/notebooks/convolutional_neural_network.ipynb`) provides a user‑friendly interface to orchestrate the data pipeline, train the model, and visualize results (loss curves, sample predictions). It imports the monolithic module, re‑uses its functions, and adds markdown explanations for experiment documentation.



## Data Flow & Pipelines
1. **Data Ingestion** – Image files (dogs and cats) are read from the local filesystem by the Data Preparation utilities. Files are loaded using Python's standard I/O and image libraries (e.g., Pillow/ OpenCV) as indicated by the resizing and normalization steps.
2. **Pre‑processing** – Each image is resized to a fixed dimension, pixel values are normalized, and optional augmentation (flip, rotate, etc.) is applied to increase training diversity.
3. **Model Construction** – The CNN Model Definition module builds a sequential stack of convolution, activation, pooling, and fully‑connected layers for binary classification.
4. **Training Loop** – The Training & Evaluation component iterates over pre‑processed batches, computes loss (e.g., binary cross‑entropy), updates weights via back‑propagation, and tracks accuracy metrics.
5. **Checkpointing** – The best‑performing model weights are saved to disk as a checkpoint file.
6. **Evaluation & Visualization** – After training, the notebook loads the checkpoint, runs inference on a validation set, and visualizes predictions and performance curves.

The entire pipeline runs within a single Python process, driven either by script execution (`python src/convolutional_neural_network.py`) or interactively via the notebook.

## Key Design Decisions
- Monolithic organization – all code resides under a single `src/` package, simplifying dependency management for a small research‑oriented project.
- Separation of concerns via logical modules (data, model, training) while keeping them in the same file for ease of experimentation.
- Use of a notebook for interactive exploration rather than a separate UI service, aligning with typical data‑science workflows.
- Checkpointing the best model based on validation accuracy to avoid over‑fitting and enable later inference without retraining.

## Scalability & Reliability
The current monolith is suitable for prototype scale and runs on a single workstation or GPU. To scale:
- **Data Size** – Introduce `tf.data`/`torch.utils.data.DataLoader` pipelines with sharding and prefetching to handle larger image collections.
- **Compute** – Leverage multi‑GPU training via `tf.distribute.MirroredStrategy` or PyTorch `DistributedDataParallel`.
- **Modularization** – Split the repository into separate packages (e.g., `data`, `model`, `train`) and expose a CLI for batch training, facilitating CI/CD pipelines.
- **Deployment** – Export the trained checkpoint to a portable format (SavedModel, ONNX) and serve via a lightweight inference service (Flask/FastAPI) if production use is required.
