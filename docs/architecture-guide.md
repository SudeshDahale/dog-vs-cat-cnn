# Technical Architecture Guide – Dog vs Cat CNN

## System Overview
The repository implements a monolithic Python application for binary image classification (dog vs cat) using a convolutional neural network (CNN). All components—data loading, preprocessing, model definition, training, evaluation, and interactive experimentation—reside in a single codebase under the `src/` directory. The architecture follows a straightforward layered design that separates concerns (data, model, and experimentation) while keeping the system simple for educational and research purposes.

**Key Modules**
- **Data Loader** – Handles ingestion of raw image files, applies resizing, normalization, and optional augmentation, and provides TensorFlow/Keras (or PyTorch) compatible datasets for training and inference.
- **CNN Model** – Implements the network architecture, training loop, checkpointing, and evaluation metrics.
- **Exploratory Notebook** – A Jupyter notebook (`src/notebooks/convolutional_neural_network.ipynb`) that imports the data loader and model modules to run interactive experiments, visualize training curves, and display sample predictions.

**File Map**
- `README.md` – Project description, usage instructions, and quickstart guide.
- `requirements.txt` – Pin‑pointed Python dependencies (e.g., `numpy`, `pandas`, `tensorflow`/`torch`, `opencv-python`, `jupyter`).
- `src/convolutional_neural_network.py` – Core implementation of data pipelines and the CNN model.
- `src/notebooks/convolutional_neural_network.ipynb` – Jupyter notebook for interactive analysis.

The guide below details the system layers, component interactions, data flow, design decisions, and scalability considerations.

## System Layers
### Presentation / Experimentation Layer
**Technologies:** Jupyter Notebook, Matplotlib/Seaborn for visualization

Provides a user‑facing interface via Jupyter Notebook. The notebook imports functions from `src/convolutional_neural_network.py` to launch training, plot metrics, and run inference on sample images. This layer is optional for production but essential for research and debugging.

### Application Logic Layer
**Technologies:** Python, TensorFlow/Keras or PyTorch, NumPy, OpenCV / Pillow

Encapsulated in `src/convolutional_neural_network.py`. This layer contains three logical sub‑components:
1. **Data Loader** – Reads image files from the dataset directory, applies preprocessing (resize, scaling, optional augmentation) and creates training/validation splits.
2. **Model Definition** – Constructs the CNN architecture using Keras `Sequential` or PyTorch `nn.Module`, defines loss, optimizer, and metrics.
3. **Training & Evaluation Engine** – Executes the training loop, logs epoch metrics, saves model checkpoints, and computes evaluation scores on the validation set.

### Data & Artifact Layer
**Technologies:** File system (POSIX), CSV/JSON for metadata if present

Stores raw image assets (dogs and cats) and generated artifacts such as model checkpoints (`.h5`/`.pth`), training logs, and visualizations. The data files themselves are not part of the repository but are expected to be placed in a configurable directory referenced by the Data Loader.



## Data Flow & Pipelines
1. **Data Ingestion** – The Data Loader scans a user‑specified root folder containing two sub‑folders (`/dogs` and `/cats`). It reads each image using OpenCV/Pillow, resizes to the network input shape (e.g., 128x128), converts to a NumPy array, and normalizes pixel values to `[0,1]`.
2. **Dataset Construction** – Images and labels are wrapped in a TensorFlow `tf.data.Dataset` or PyTorch `DataLoader`, with optional shuffling and batching.
3. **Model Training** – The Training Engine iterates over batches, feeds them to the CNN, computes loss, performs back‑propagation, and updates weights. After each epoch, it evaluates on the validation split and writes metrics to stdout and optional log files.
4. **Checkpointing** – At the end of each epoch (or when validation loss improves), the model weights are saved to the Data & Artifact Layer.
5. **Inference / Visualization** – The notebook loads a saved checkpoint, runs the model on a small set of held‑out images, and visualizes predictions alongside true labels using Matplotlib.

**Component Interaction Diagram** (textual):
```
[Notebook] <--import--> [convolutional_neural_network.py] --> {Data Loader, Model, Trainer}
[Data Loader] --> File System (raw images)
[Trainer] --> File System (checkpoints, logs)
[Model] <--trained weights-- [Trainer]
```

## Key Design Decisions
- **Monolithic Structure** – All code lives in a single Python module for simplicity. This reduces onboarding friction for learners and ensures that data loading, model definition, and training stay tightly coupled.
    *Benefit*: Easy to run end‑to‑end with a single command.
    *Trade‑off*: Limited reusability of individual components in other projects.

- **Explicit Data Pipeline** – The loader performs deterministic preprocessing (resize + normalization) inside the same script rather than relying on external pipelines. This guarantees reproducibility across runs.

- **Framework Choice (TensorFlow/Keras or PyTorch)** – The repository pins a single deep‑learning framework in `requirements.txt`. Using a high‑level API (Keras `Sequential` or PyTorch `nn.Module`) simplifies model construction and training loops, aligning with the educational focus.

- **Checkpoint Strategy** – Model weights are saved after each epoch (or when validation loss improves). This enables easy resumption of training and quick evaluation in the notebook without retraining.

- **Jupyter Notebook for Exploration** – Keeping an interactive notebook separate from the core module encourages exploratory analysis while preserving the deterministic training script.


## Scalability & Reliability
The current monolith is optimized for single‑GPU or CPU execution on modest datasets. To scale:
- **Data Parallelism** – Replace the single `DataLoader` with `tf.data.experimental.DistributeOptions` or PyTorch `DistributedDataParallel` to train across multiple GPUs.
- **Dataset Caching** – Add `cache()` and `prefetch()` steps in the TensorFlow pipeline (or use `num_workers` > 0 in PyTorch) to reduce I/O bottlenecks.
- **Modular Refactor** – Extract the Data Loader and Model into independent packages (`dogcat_data` and `dogcat_model`). This would allow independent versioning and reuse in other projects.
- **Cloud Storage** – Store raw images on cloud buckets (e.g., S3) and stream them via `tf.io` or `torchvision.io` to handle larger datasets without requiring local disk space.
- **Experiment Tracking** – Integrate lightweight experiment logging (e.g., TensorBoard or Weights & Biases) to manage many hyper‑parameter runs.

Even without these changes, the architecture supports straightforward scaling by increasing batch size, using a more powerful GPU, or augmenting the dataset.

---
*All sections are derived from the observed repository layout (`README.md`, `requirements.txt`, `src/convolutional_neural_network.py`, `src/notebooks/convolutional_neural_network.ipynb`) and the identified modules.*
