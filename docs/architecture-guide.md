# Technical Architecture Guide for dog-vs-cat-cnn

## System Overview
The *dog-vs-cat-cnn* repository implements a monolithic, batch-oriented machine‑learning pipeline for binary image classification (dog vs. cat) using a Convolutional Neural Network (CNN). The core logic lives in pure Python under the `src/` package, while interactive exploration is provided through a Jupyter notebook. The system is designed for one‑off or scheduled training runs, processing static image datasets on local or cloud compute resources.

## System Layers
### Data Ingestion & Storage
**Technologies:** Python (os, pathlib), NumPy

Static image collections (e.g., Kaggle's Dogs vs Cats dataset) are stored on disk or a mounted cloud bucket. The ingestion step consists of loading image file paths and labels into Python lists or NumPy arrays. No external services are invoked; the pipeline relies on the file system.

### Pre‑processing
**Technologies:** Pillow (PIL), NumPy, TensorFlow/Keras preprocessing utilities

Images are read, resized, normalized, and optionally augmented. The `src/convolutional_neural_network.py` module contains helper functions that use Pillow (PIL) and NumPy to transform raw pixel data into tensors suitable for TensorFlow/Keras.

### Model Definition
**Technologies:** TensorFlow, Keras

The CNN architecture (stack of Conv2D, MaxPooling, Dense layers) is defined in `src/convolutional_neural_network.py`. Hyper‑parameters such as filter count, kernel size, activation functions, and dropout rates are hard‑coded but exposed via function arguments for experimentation.

### Training (Batch Execution)
**Technologies:** TensorFlow/Keras fit(), Python logging

A single training script orchestrates data loading, model compilation, and batch‑wise fitting. The training loop runs to completion in a batch mode (no streaming or online learning). Checkpoints and final model weights are persisted to a `models/` directory (if created).

### Evaluation & Metrics
**Technologies:** TensorFlow/Keras evaluate(), Matplotlib, scikit‑learn (optional)

After training, the model is evaluated on a held‑out validation set. Accuracy, loss, and optionally ROC‑AUC are printed to console and logged. The same script can generate confusion matrices using Matplotlib.

### Interactive Exploration (Notebook)
**Technologies:** Jupyter Notebook, IPython, Matplotlib, TensorFlow/Keras

The `src/notebooks/convolutional_neural_network.ipynb` notebook imports the core module to reproduce training steps, visualize intermediate feature maps, and experiment with hyper‑parameters in an ad‑hoc fashion. It serves as documentation and a rapid‑prototyping surface for data scientists.



## Data Flow & Pipelines
1️⃣ **Load Dataset** – File system paths are enumerated → image files are read with Pillow. 2️⃣ **Pre‑process** – Images are resized, normalized, and batched as NumPy arrays. 3️⃣ **Model Build** – `build_cnn()` constructs the Keras model. 4️⃣ **Train** – `model.fit()` consumes pre‑processed batches, updates weights, and writes checkpoints. 5️⃣ **Validate** – `model.evaluate()` runs on a separate batch, producing metrics. 6️⃣ **Export** – Final weights (`.h5`/SavedModel) are saved for downstream inference. The notebook repeats steps 2‑5 interactively, allowing visual inspection after each epoch.

## Key Design Decisions
- Monolithic layout – all code resides under `src/`, simplifying dependency management for small‑scale research.
- Batch‑only training – the pipeline assumes the entire training set can be loaded (or streamed in fixed‑size batches) without needing online learning or continuous inference.
- TensorFlow/Keras as the deep‑learning backend – provides high‑level APIs for model definition, training, and checkpointing, reducing boilerplate.
- Explicit separation of core logic (`convolutional_neural_network.py`) from experimentation (`*.ipynb`). This encourages reproducibility while still supporting ad‑hoc analysis.
- Requirements pinned in `requirements.txt` to guarantee a consistent environment across local machines and CI runners.

## Scalability & Reliability
The current monolith is suitable for single‑node training on a GPU or CPU. To scale:
- **Hardware scaling** – swap the local runtime for a GPU‑enabled instance (e.g., AWS EC2 G4, GCP Compute Engine with NVIDIA GPU). TensorFlow automatically leverages the device.
- **Data parallelism** – modify the training script to use `tf.distribute.MirroredStrategy` for multi‑GPU scaling on a single node, or `tf.distribute.MultiWorkerMirroredStrategy` for a cluster.
- **Batch size tuning** – larger batches reduce per‑epoch overhead but require more memory; adjust based on GPU VRAM.
- **Containerization** – wrap the repository in a Docker image (Python base, CUDA drivers) to ensure reproducible scaling across environments.
- **Pipeline orchestration** – for recurring training (e.g., nightly retraining), embed the script in an Airflow DAG or a simple cron job, feeding new data snapshots into the same batch flow.
- **Model serving** – although out of scope for the current batch‑only design, exported SavedModel artifacts can later be served via TensorFlow Serving or a lightweight Flask API.
These steps keep the architecture consistent while allowing horizontal and vertical scaling as data volume or model complexity grows.
