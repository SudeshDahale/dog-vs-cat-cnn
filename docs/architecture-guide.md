# Technical Architecture Guide – dog-vs-cat-cnn

## System Overview
The *dog-vs-cat-cnn* repository implements a monolithic, batch‑processing machine‑learning pipeline for binary image classification (dogs vs. cats). The codebase is organized around four logical modules – Data Preparation, Model Definition, Training Engine, and an Exploratory Notebook – all written in Python and residing under the `src/` directory. The pipeline loads raw image files, applies preprocessing and augmentation, builds a convolutional neural network (CNN), trains the model on batched data, evaluates performance, and persists the trained artifact for downstream analysis in a Jupyter notebook.

## System Layers
### Data Preparation Layer
**Technologies:** Python, NumPy, Pillow / OpenCV (image I/O), TensorFlow/Keras `ImageDataGenerator` or PyTorch `torchvision.transforms` (as defined in `requirements.txt`)

Responsible for ingesting raw image datasets, applying deterministic preprocessing steps (resizing, normalization) and stochastic augmentations (flips, rotations). Implements a batched data loader that yields NumPy/TensorFlow tensors for the training engine.

### Model Definition Layer
**Technologies:** Python, TensorFlow/Keras or PyTorch (as listed in `requirements.txt`)

Encapsulates the CNN architecture used for binary classification. The module defines the network topology, activation functions, regularization (dropout, batch‑norm), and compiles the model with an appropriate loss (binary cross‑entropy) and optimizer.

### Training Engine Layer
**Technologies:** Python, TensorFlow/Keras or PyTorch training APIs, Matplotlib / Seaborn for optional inline metric plots

Orchestrates the end‑to‑end training loop: receives batched data from the Data Preparation layer, feeds it to the Model Definition, tracks metrics, performs validation, and persists the final model artifact (`.h5`, `.pt`, or similar). Also provides a small CLI entry point for reproducible runs.

### Exploratory Notebook Layer
**Technologies:** Jupyter, Python, Matplotlib / Seaborn, IPython widgets (optional for interactive hyper‑parameter tweaks)

A Jupyter notebook (`src/notebooks/convolutional_neural_network.ipynb`) that loads the saved model, visualizes training curves, inspects mis‑classifications, and optionally fine‑tunes hyper‑parameters interactively. Serves as the primary interface for data scientists to experiment with the pipeline.



## Data Flow & Pipelines
Raw image files → Data Preparation (load ➜ preprocess ➜ augment) → Model Definition (CNN architecture) → Training Engine (train/evaluate ➜ save model) → Exploratory Notebook (load model ➜ visualize results)

## Key Design Decisions
- Monolithic repository layout keeps all pipeline stages together, simplifying reproducibility and version control for a research‑oriented project.
- Explicit separation of concerns into Python modules (`convolutional_neural_network.py`) and a notebook (`convolutional_neural_network.ipynb`) enables both script‑driven execution and interactive experimentation.
- Batch processing is used for both data loading (via TensorFlow/Keras `ImageDataGenerator` or PyTorch `DataLoader`) and model training, which aligns with typical GPU‑accelerated training workflows.
- Model persistence (e.g., `model.save()` or `torch.save`) allows the notebook to load the exact trained weights for downstream analysis without re‑training.
- All third‑party dependencies are declared in `requirements.txt`, ensuring a reproducible Python environment.

## Scalability & Reliability
The pipeline is built on batch processing, making it straightforward to scale horizontally by increasing batch size or distributing data loading across multiple workers (`tf.data` pipelines or PyTorch `DataLoader` with `num_workers`). GPU acceleration is leveraged during training; the architecture can be extended to multi‑GPU or distributed training (e.g., TensorFlow `tf.distribute.Strategy` or PyTorch `DistributedDataParallel`) without altering the monolithic code layout. Data volume growth is mitigated by streaming data from disk rather than loading the entire dataset into memory, and augmentation is performed on‑the‑fly to keep storage footprints low.
