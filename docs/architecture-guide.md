# Technical Architecture Guide for dog-vs-cat-cnn

## System Overview
The *dog-vs-cat-cnn* repository implements a monolithic, batch‑processing machine‑learning application for binary image classification (dog vs cat). The codebase is organized around three logical modules: (1) **Model Definition**, (2) **Training & Evaluation**, and (3) **Exploratory Notebook**. All components are written in Python and execute synchronously as a single process, leveraging the Keras API (TensorFlow backend) for deep‑learning primitives. The data pipeline is a classic offline, batch‑oriented workflow: images are loaded from disk, pre‑processed, fed into a training loop, and the resulting model weights are persisted to the filesystem.

Key files:
- `src/convolutional_neural_network.py` – core implementation (model class, training routine, evaluation utilities).
- `src/notebooks/convolutional_neural_network.ipynb` – Jupyter notebook for interactive experimentation and visualization.
- `requirements.txt` – Python dependencies.
- `README.md` – usage guide.

The guide below documents the system layers, component interactions, data flow, design decisions, and scalability considerations, each grounded in the actual repository structure.

## System Layers
### Data Ingestion & Pre‑processing Layer
**Technologies:** Python, TensorFlow (tf.data), NumPy

Handles filesystem access, image decoding, resizing, normalisation, and optional augmentation. Implemented with TensorFlow's `tf.data` API in `convolutional_neural_network.py`.

### Model Definition Layer
**Technologies:** Python, TensorFlow Keras

Encapsulates the CNN architecture for binary classification. Defined as a Keras `Sequential` or functional model within `src/convolutional_neural_network.py`.

### Training & Evaluation Layer
**Technologies:** Python, TensorFlow Keras, TensorBoard (optional)

Coordinates model compilation, training loop, callback management, checkpointing, and post‑training evaluation. All logic lives in `convolutional_neural_network.py` and is invoked via the `if __name__ == "__main__"` entry point.

### Persistence Layer
**Technologies:** Python, TensorFlow Keras I/O

Serialises the trained model weights to disk (`.h5` format) and provides a loading utility for downstream inference or notebook reuse.

### Exploratory Notebook Layer
**Technologies:** Python, Jupyter Notebook, Matplotlib/Seaborn (visualisation)

Provides an interactive environment for data inspection, model training, hyper‑parameter tuning, and result visualisation. The notebook imports the same Python modules used by the batch script, ensuring code reuse and reproducibility.



## Data Flow & Pipelines
1. **Data Ingestion** – The training script reads image files from a local dataset directory (e.g., `data/train/` and `data/validation/`). `tensorflow.keras.preprocessing.image_dataset_from_directory` (or a custom `ImageDataGenerator`) loads images into memory as batched `tf.data.Dataset` objects.  
2. **Pre‑processing** – Images are resized, normalized to `[0,1]`, and optionally augmented (flip, rotation). Pre‑processing logic lives inside `convolutional_neural_network.py` as part of the dataset pipeline.  
3. **Model Construction** – `ConvolutionalNeuralNetwork` (or similarly named class) builds a sequential/functional Keras model consisting of Conv2D, MaxPooling, Dropout, and Dense layers. This definition resides in `src/convolutional_neural_network.py`.  
4. **Training Loop** – `train()` function receives the prepared dataset, compiles the model (loss=`binary_crossentropy`, optimizer=`Adam`, metrics=`accuracy`), and runs `model.fit(...)` for a fixed number of epochs. Callbacks (ModelCheckpoint, EarlyStopping) are configured to persist the best weights to `models/`.
5. **Evaluation** – After training, `evaluate()` loads the best checkpoint and runs `model.evaluate(...)` on the validation set, printing loss/accuracy.  
6. **Persistence** – Model weights are saved as an HDF5 (`.h5`) file (`model_weights.h5`). The notebook can later load this file via `tf.keras.models.load_model`.  
7. **Exploratory Notebook** – The Jupyter notebook (`src/notebooks/convolutional_neural_network.ipynb`) reproduces steps 1‑5 interactively, allowing the developer to visualise loss curves, sample predictions, and tweak hyper‑parameters on‑the‑fly.

## Key Design Decisions
- Monolithic repository layout – all Python code resides under `src/` and the notebook under `src/notebooks/`, simplifying dependency management and version control for a single‑researcher project.
- Batch‑processing training – the workflow processes the entire dataset in a single training run rather than streaming or online learning, matching the typical offline image‑classification use case.
- Keras high‑level API – chosen for rapid prototyping, readability, and built‑in utilities (callbacks, checkpointing).
- Explicit checkpointing – `ModelCheckpoint` ensures the best model is persisted, enabling deterministic reproducibility when loading from the notebook.
- Separate notebook for experimentation – avoids polluting the production training script while still sharing the exact model definition and preprocessing code.
- Requirements pinning – `requirements.txt` lists exact package versions to guarantee reproducible environments across machines.

## Scalability & Reliability
The current architecture is oriented toward a single‑node, single‑GPU execution model. Scalability can be achieved along several axes:

* **Data volume** – `tf.data` pipelines support sharding, prefetching, and caching, allowing the system to handle larger image collections with minimal code changes.
* **Compute** – By switching the TensorFlow backend to a GPU-enabled environment (installing `tensorflow-gpu`) and optionally employing `tf.distribute.MirroredStrategy`, the training loop can leverage multiple GPUs on a single host. For multi‑node scaling, the same strategy can be extended to `MultiWorkerMirroredStrategy` with minimal refactoring.
* **Modularisation** – While the repository is monolithic, the clear separation of concerns (data, model, training) makes it straightforward to extract each layer into its own service or container if future micro‑service decomposition is required.
* **Experiment tracking** – Adding TensorBoard or MLflow callbacks would provide scalable experiment management without altering the core architecture.

Overall, the system is designed for simplicity and rapid iteration, yet it retains a clean layering that can be incrementally scaled as dataset size or compute requirements grow.
