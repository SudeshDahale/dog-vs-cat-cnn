# Technical Architecture Guide – Dog vs Cat CNN

## System Overview
The **dog-vs-cat-cnn** repository implements a binary image classification pipeline that distinguishes dogs from cats using a Convolutional Neural Network (CNN). The codebase is written in Python and follows a monolithic, end‑to‑end ML‑workflow architecture. Core functionality is split across three logical modules—**data‑prep**, **cnn‑model**, and **evaluation**—and is orchestrated via the main script `src/convolutional_neural_network.py`. Interactive exploration and result visualisation are provided in the Jupyter notebook `src/notebooks/convolutional_neural_network.ipynb`. The repository is self‑contained: data loading, model definition, training, persistence, and evaluation all live in the same code tree, making it straightforward to run on a single machine or a modest GPU‑enabled environment.

## System Layers
### Data Ingestion & Preparation
**Technologies:** Python, TensorFlow / Keras, NumPy, Pandas (optional for CSV manifests), Pillow / OpenCV (image handling)

Loads raw image files, applies resizing, normalization, and optional data‑augmentation, and splits the dataset into training and validation subsets. The logic resides in the **data‑prep** module (e.g., functions in `src/convolutional_neural_network.py` that use `tf.keras.preprocessing.image_dataset_from_directory` or `ImageDataGenerator`).

### Model Definition, Training & Persistence
**Technologies:** TensorFlow / Keras, Python, CUDA / cuDNN (when GPU is available)

Defines a sequential CNN architecture, compiles it with an appropriate loss (binary cross‑entropy) and optimizer, and trains the model on the prepared dataset. Training callbacks such as EarlyStopping and ModelCheckpoint are used. The trained model is serialized to disk (SavedModel or HDF5). This logic lives in the **cnn‑model** module, primarily within `src/convolutional_neural_network.py`.

### Evaluation & Visualisation
**Technologies:** Matplotlib / Seaborn, Scikit‑learn (metrics), TensorFlow / Keras

Loads the persisted model, runs inference on the validation set, computes performance metrics (accuracy, precision, recall, confusion matrix), and generates visual artefacts such as ROC curves and sample prediction grids. Implemented in the **evaluation** module and exercised from the notebook `src/notebooks/convolutional_neural_network.ipynb`.

### Interactive Exploration (Notebook)
**Technologies:** Jupyter Notebook, IPython widgets (optional), Matplotlib

Provides a step‑by‑step walkthrough of the pipeline, enabling users to inspect data samples, tweak hyper‑parameters, and visualise training history. Implemented in `src/notebooks/convolutional_neural_network.ipynb`.



## Data Flow & Pipelines
1️⃣ **Raw Image Directory** → `data‑prep` reads image files and creates a TensorFlow `tf.data.Dataset` (train/val). 2️⃣ **Prepared Dataset** → passed to `cnn‑model` where the CNN is instantiated, compiled, and trained. 3️⃣ **Model Checkpoint** → after training, the model is saved to `models/` (or similar path). 4️⃣ **Evaluation** → `evaluation` loads the saved model, runs inference on the validation `tf.data.Dataset`, computes metrics, and writes visualisations (e.g., PNG plots) to `outputs/`. 5️⃣ **Notebook** → optionally reproduces steps 1‑4, displaying intermediate results for human inspection.

## Key Design Decisions
- Monolithic repository layout – all code, data pipelines, and notebooks live under a single `src/` tree, simplifying setup and version control for a research‑oriented project.
- Use of TensorFlow/Keras high‑level API – enables concise model definition and leverages built‑in data pipelines (`image_dataset_from_directory`) and callbacks for robust training.
- Explicit separation of concerns via logical modules (data‑prep, cnn‑model, evaluation) – improves readability, testability, and future extensibility without introducing micro‑service overhead.
- Model checkpointing and early stopping – prevents over‑fitting and provides a recoverable training artefact.
- Reproducibility measures – fixed random seeds (NumPy, TensorFlow) and deterministic data splits are configured at the top of the script.
- Notebook for exploratory analysis – offers a low‑code entry point for data scientists while keeping the production script headless.

## Scalability & Reliability
The current monolithic design is optimal for a single‑node, modest‑size dataset (e.g., the standard Kaggle Dogs vs Cats set). To scale:
- **Data volume**: Switch to TensorFlow `tf.data` pipelines with `.cache()`, `.prefetch()`, and sharding for larger image corpora.
- **Compute**: Leverage GPU or multi‑GPU training by enabling `tf.distribute.MirroredStrategy` in the training script; the codebase already uses Keras callbacks that are compatible with distributed strategies.
- **Model size**: Replace the simple Sequential network with a deeper architecture (e.g., ResNet, EfficientNet) using Keras Applications, keeping the same training loop.
- **Deployment**: Export the SavedModel and serve it via TensorFlow Serving or convert to TensorFlow Lite for edge deployment; the persistence format is already compatible.
- **CI/CD**: Integrate the script into a GitHub Actions workflow that runs unit tests, validates training on a small subset, and pushes the model artefact to a model registry.
Overall, the architecture’s clear module boundaries make incremental scaling straightforward without a full rewrite.
