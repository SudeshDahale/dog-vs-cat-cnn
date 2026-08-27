# Technical Architecture Guide for dog-vs-cat-cnn

## System Overview
The *dog-vs-cat-cnn* repository implements a monolithic, batch‑processing system for training a convolutional neural network (CNN) that classifies images of dogs and cats. The codebase is written in Python and organized under the `src/` directory with a clear separation between model definition, training logic, and exploratory analysis. All components run as a single process, orchestrated by a training script that loads data in batches, feeds it to the model, updates parameters, and evaluates performance. Interactive experimentation is supported through a Jupyter notebook.

Key entry points are:
* `src/convolutional_neural_network.py` – defines the CNN architecture and provides helper functions for forward passes and weight initialization.
* `src/notebooks/convolutional_neural_network.ipynb` – an exploratory notebook that imports the model module, visualizes training curves, and runs ad‑hoc predictions.
* `README.md` and `requirements.txt` – documentation and dependency specifications for reproducing the environment.

The system follows a classic monolith pattern: all layers (data loading, model, training loop, evaluation) reside in the same Python runtime, communicating via in‑memory objects rather than network calls.

## System Layers
### Data Ingestion Layer
**Technologies:** Python, NumPy, Pillow, torchvision or tf.data (as declared in requirements)

Responsible for locating image files, applying deterministic preprocessing (resize, normalization), and emitting batched tensors to downstream components. Implemented using standard Python libraries and the data‑loader utilities referenced in `requirements.txt`.

### Model Definition Layer
**Technologies:** Python, PyTorch or TensorFlow (as per requirements)

Encapsulates the CNN architecture (convolution, pooling, fully‑connected layers) and exposes a callable class used by the training script and notebook. Defined in `src/convolutional_neural_network.py`.

### Training & Evaluation Layer
**Technologies:** Python, Optimizers from the chosen deep‑learning framework

Orchestrates the training loop: iterates over batches, forwards them through the model, computes loss, triggers back‑propagation, updates parameters, and periodically evaluates on a validation set. Also handles checkpointing of model weights.

### Exploratory & Visualization Layer
**Technologies:** Jupyter Notebook, Matplotlib/Seaborn, Python

Provides an interactive environment for data scientists to experiment with hyper‑parameters, visualise learning curves, and inspect model predictions. Implemented as a Jupyter notebook located at `src/notebooks/convolutional_neural_network.ipynb`.



## Data Flow & Pipelines
1. **Batch Data Loading** – The training script reads image files from a local dataset directory (e.g., `data/train/` and `data/validation/`). Images are pre‑processed (resize, normalization) and grouped into batches using a Python data loader (typically `torch.utils.data.DataLoader` or `tf.data.Dataset`).
2. **Model Forward Pass** – Each batch is passed to the CNN class defined in `src/convolutional_neural_network.py`. The model computes logits for the two classes (dog, cat).
3. **Loss Computation & Back‑Propagation** – The training script calculates a classification loss (e.g., cross‑entropy) and invokes the optimizer to update model parameters.
4. **Evaluation** – After each epoch, the script runs a validation pass on the held‑out batch set, computes accuracy/F1, and logs metrics.
5. **Persistence** – Trained weights are saved to disk (e.g., `model.pth` or `model.h5`). The notebook can later load these weights for inference and visualisation.
6. **Visualization** – The Jupyter notebook imports the same model module, loads saved weights, and renders loss/accuracy curves, sample predictions, and feature maps for deeper analysis.

All data movement stays inside the Python process; there are no external services or message queues.

## Key Design Decisions
- Monolithic layout – All components live in a single Python package (`src/`). This simplifies dependency management and reproducibility for a research‑oriented project.
- Batch‑processing training – The system processes the dataset in fixed‑size batches, which aligns with GPU memory constraints and enables deterministic epoch boundaries.
- Separation of concerns via modules – Model architecture is isolated in its own file, allowing the notebook to import the same class without duplicating code.
- Explicit requirement file – `requirements.txt` lists exact library versions, ensuring that the training script and notebook run with identical runtimes.
- Version‑controlled notebooks – Storing the exploratory notebook alongside source code guarantees that analysis steps are reproducible and auditable.

## Scalability & Reliability
The current monolithic, batch‑processing design is well‑suited for single‑machine training on modest datasets. To scale:
* **Data volume** – Increase batch size or employ `torch.utils.data.DataLoader` with `num_workers > 1` to parallelise I/O.
* **Compute** – Swap the CPU runtime for a GPU‑enabled instance; the code already uses framework‑level device abstractions (e.g., `torch.device('cuda')`).
* **Distributed training** – While not built‑in, the architecture can be extended by wrapping the model in a DistributedDataParallel wrapper and launching multiple processes; the modular separation of model and training logic eases this transition.
* **Pipeline parallelism** – For extremely large models, the layers could be split across devices, but the existing CNN is lightweight enough that a single GPU suffices.
* **Data storage** – Moving the image dataset to a cloud object store (S3, GCS) and streaming batches would allow the system to handle petabyte‑scale corpora, with minimal code changes (just the data loader path).

Overall, the architecture is deliberately simple for clarity and reproducibility, yet it follows best practices that make horizontal scaling feasible when needed.
