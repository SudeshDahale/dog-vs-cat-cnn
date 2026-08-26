# Technical Architecture Guide for Dog vs Cat CNN

## System Overview
The **dog-vs-cat-cnn** repository implements a monolithic machine‑learning pipeline for binary image classification (dog vs cat) using a convolutional neural network (CNN). All components—data preprocessing, model definition, training, evaluation, and interactive inference—are written in Python and reside under the `src/` package, with an accompanying Jupyter notebook for exploratory analysis. The architecture follows a straightforward, end‑to‑end flow suitable for small‑to‑medium sized image datasets and can be executed on a single workstation with optional GPU acceleration.

## System Layers
### Data Preprocessing Layer
**Technologies:** Python, PyTorch, torchvision.transforms, Pillow (PIL)

Loads image files from the filesystem, performs resizing to a fixed input size, normalizes pixel values to the range expected by the CNN, and constructs PyTorch `Dataset` and `DataLoader` objects. Handles deterministic train/validation splitting.

### Model Definition Layer
**Technologies:** Python, PyTorch

Encapsulates the convolutional neural network architecture in a subclass of `torch.nn.Module`. The network stacks convolutional layers with ReLU activations, pooling layers, and fully‑connected output layers that produce a single logit for binary classification.

### Training Pipeline Layer
**Technologies:** Python, PyTorch, torch.optim, torch.nn.functional

Implements the end‑to‑end training loop: iterates over `DataLoader` batches, computes forward passes, calculates binary cross‑entropy loss, back‑propagates gradients, updates weights via an optimizer, and saves model checkpoints. Logs epoch‑level loss and accuracy for monitoring.

### Evaluation & Inference Layer
**Technologies:** Python, PyTorch, Jupyter Notebook, matplotlib/seaborn (optional)

Runs the trained model on held‑out validation data to compute final performance metrics. Provides a Jupyter notebook for ad‑hoc inference on user‑supplied images, visualizing predictions alongside ground‑truth labels.



## Data Flow & Pipelines
1. **Raw Image Dataset** – Images are stored on disk (e.g., in a `data/` folder, not shown in the excerpt). 2. **Data Preprocessing** (`src/convolutional_neural_network.py`) reads each image, applies resizing, pixel‑value normalization, and assembles PyTorch `Dataset` objects. 3. **Train/Validation Split** – The preprocessing module creates deterministic splits and wraps them with `DataLoader` for batched streaming. 4. **Model Definition** – A `torch.nn.Module` subclass defines the CNN architecture (convolutional layers, pooling, fully‑connected heads). 5. **Training Pipeline** – The same script orchestrates the training loop: forward pass, loss computation (e.g., `nn.BCEWithLogitsLoss`), optimizer step (e.g., `Adam`), checkpoint saving (`torch.save`) and metric logging per epoch. 6. **Evaluation & Inference** – After training, the model is evaluated on the validation set; metrics (accuracy, loss) are reported. 7. **Interactive Notebook** (`src/notebooks/convolutional_neural_network.ipynb`) loads the saved checkpoint, runs inference on new images, and visualizes predictions.

## Key Design Decisions
- Monolithic layout under `src/` keeps the entire pipeline in a single Python module, simplifying execution for educational or prototype use.
- Choice of PyTorch as the deep‑learning framework offers dynamic graph construction, easy debugging, and seamless GPU support.
- Explicit train/validation split performed during preprocessing avoids data leakage and keeps the split logic transparent.
- Model checkpointing with `torch.save` ensures training can be resumed and the best‑performing weights are preserved for inference.
- A dedicated notebook (`src/notebooks/convolutional_neural_network.ipynb`) separates exploratory analysis from the production‑grade script, enabling rapid experimentation without modifying core code.

## Scalability & Reliability
The current monolithic design targets a single‑node environment. Scalability can be improved by:
- **Data Loading:** Increase `num_workers` in `DataLoader` to parallelize image I/O and transformation.
- **GPU Utilization:** Move tensors to a CUDA device (`.to('cuda')`) and optionally use mixed‑precision training (`torch.cuda.amp`).
- **Model Size:** Replace the custom CNN with a deeper architecture (e.g., ResNet) while reusing the same training loop.
- **Distributed Training:** Refactor the training pipeline to use `torch.nn.parallel.DistributedDataParallel` for multi‑GPU or multi‑node scaling.
- **Containerization:** Wrap the pipeline in a Docker image that pins Python and library versions from `requirements.txt`, enabling reproducible deployments.
- **Data Management:** For larger datasets, store images in an object store (e.g., S3) and stream them via `torchvision.datasets.ImageFolder` or a custom dataset that reads from the cloud.
