# Technical Architecture Guide – Dog vs Cat CNN

## System Overview
This repository implements a binary image classification model that distinguishes between dogs and cats using a convolutional neural network (CNN) built with Python deep‑learning libraries (PyTorch/Keras). The code is organized as a monolithic, batch‑processing application that runs end‑to‑end from data ingestion through model training and evaluation. All core components reside under the `src/` package, with a Jupyter notebook for exploratory analysis and a README that documents usage and dependencies.

## System Layers
### Data Ingestion & Preprocessing
**Technologies:** Python, Pillow/OpenCV, torchvision.transforms / tf.keras.preprocessing

Handles loading of raw image files, resizing to a fixed resolution, normalizing pixel intensity, and splitting the dataset into train/validation subsets. Implemented with Python's `os`, `glob`, `PIL`/`opencv`, and `torchvision.transforms` (or `tf.keras.preprocessing.image`).

### Model Definition
**Technologies:** Python, PyTorch, Keras (TensorFlow backend)

Encapsulates the CNN architecture in a single module. The file `src/convolutional_neural_network.py` defines a class inheriting from `torch.nn.Module` (or `tf.keras.Model`), stacking convolutional layers, ReLU activations, max‑pooling, dropout, and a final dense layer with sigmoid output for binary classification.

### Training & Evaluation
**Technologies:** Python, PyTorch Optimizer / Keras compile‑fit API, tqdm, matplotlib

Orchestrates the batch‑processing training loop, optimizer configuration, loss computation, metric tracking, and checkpointing. Uses `torch.utils.data.DataLoader` (or `tf.data.Dataset`) to feed batches, and logs progress via `tqdm` and `matplotlib` in the notebook.



## Data Flow & Pipelines
1. **Data Ingestion** – Image files are read from a local directory (or dataset path) using standard Python I/O utilities. 2. **Preprocessing** – Each image is resized to a uniform shape, pixel values are normalized, and optional data augmentations are applied. 3. **Dataset Split** – The preprocessed dataset is shuffled and split into training and validation (test) sets. 4. **Model Definition** – `src/convolutional_neural_network.py` constructs a CNN architecture (convolution → pooling → fully‑connected layers) using either PyTorch `torch.nn.Module` or Keras `tf.keras.Model`. 5. **Training Loop** – A batch‑wise training loop iterates over the training set, computes loss, back‑propagates gradients, and updates weights. 6. **Evaluation** – After each epoch (or at the end), the model is evaluated on the validation set, reporting loss and accuracy metrics. 7. **Results** – Metrics and optionally model checkpoints are saved to disk for later analysis, and the notebook (`src/notebooks/convolutional_neural_network.ipynb`) visualizes training curves.

## Key Design Decisions
- Monolithic layout – all code lives under `src/` for simplicity, avoiding micro‑service overhead for a research‑grade prototype.
- Batch‑processing training – model is trained on full epochs over static image batches rather than streaming or online learning, matching typical deep‑learning workflows.
- Choice of PyTorch vs Keras – the repository supports either framework (as reflected in imports). This flexibility lets contributors experiment with their preferred library while keeping the same high‑level architecture.
- Image resizing to a fixed size (e.g., 128×128) – reduces memory footprint and standardizes input shape across the network.
- Normalization to [0,1] (or standard mean‑std) – accelerates convergence and aligns with pretrained weight expectations if transfer learning is introduced later.
- Explicit train/validation split – ensures unbiased evaluation and prevents data leakage.
- Use of a Jupyter notebook for exploratory runs – aids rapid prototyping and visual inspection of loss/accuracy curves without modifying the core script.

## Scalability & Reliability
Although designed as a research prototype, the architecture can scale with modest adjustments:
- **Dataset size** – Replace the in‑memory list of file paths with a lazy loader (`torch.utils.data.Dataset` or `tf.data.Dataset`) that streams images from disk, allowing millions of samples.
- **Hardware acceleration** – Enable GPU usage by moving the model and tensors to `cuda` (PyTorch) or by setting the TensorFlow device context. Batch size can be increased proportionally to GPU memory.
- **Distributed training** – The monolithic script can be wrapped with PyTorch `DistributedDataParallel` or TensorFlow `tf.distribute.Strategy` to run across multiple GPUs or nodes.
- **Experiment tracking** – Integrating tools like Weights & Biases or TensorBoard would provide scalable logging and hyper‑parameter sweeps.
- **Data augmentation pipelines** – Leveraging libraries such as Albumentations can offload augmentation to the CPU, keeping GPU utilization high.
Overall, the current design is modular enough to adopt these enhancements without restructuring the codebase.
