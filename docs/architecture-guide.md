# Technical Architecture Guide for Dog vs Cat CNN

## System Overview
The **dog-vs-cat-cnn** repository implements a monolithic machine‑learning pipeline for binary image classification (dog vs cat) using a convolutional neural network (CNN) built with Keras/TensorFlow. The codebase is organized under a simple `src/` package and an exploratory Jupyter notebook. The pipeline consists of three logical modules: Data Loading & Preprocessing, Model Definition, and Training & Evaluation. All components are written in Python and execute sequentially within a single process, suitable for single‑node training on CPU or GPU.

Key files:
- `README.md` – high‑level description, usage instructions, and dataset acquisition notes.
- `requirements.txt` – pinned Python dependencies (TensorFlow, Keras, numpy, pandas, matplotlib, etc.).
- `src/convolutional_neural_network.py` – core implementation containing data loaders, model architecture, training loop, and evaluation utilities.
- `src/notebooks/convolutional_neural_network.ipynb` – exploratory notebook that demonstrates end‑to‑end execution of the pipeline and visualises results.

The guide below details the system layers, their interactions, data flow, design rationale, and scalability considerations.

## System Layers
### Data Loading & Preprocessing
**Technologies:** Python, TensorFlow (tf.keras), NumPy, Pillow

Responsible for ingesting raw image files, applying deterministic transformations (resize, normalization) and stochastic augmentations, and exposing TensorFlow `tf.data.Dataset` objects for downstream consumption.

### Model Definition
**Technologies:** Python, TensorFlow (Keras)

Encapsulates the CNN architecture. Uses Keras layers to define a depth‑wise feature extractor followed by a dense classifier. The model object is reusable across training and inference contexts.

### Training & Evaluation
**Technologies:** Python, TensorFlow (Keras), Matplotlib, scikit‑learn (optional for reports)

Executes the learning loop, manages callbacks, persists the best‑performing checkpoint, and computes quantitative metrics on a separate test split.



## Data Flow & Pipelines
1. **Dataset Acquisition** – Raw image files (dogs & cats) are stored on disk (or downloaded by the notebook).  
2. **Data Loading & Preprocessing** (`src/convolutional_neural_network.py`):  
   - Uses `tf.keras.preprocessing.image_dataset_from_directory` (or custom `ImageDataGenerator`) to read images.
   - Applies resizing to a fixed input shape, pixel value scaling (0‑1) and optional augmentation (random flips, rotations, zoom).  
   - Splits data into training, validation, and test subsets.
3. **Model Definition** (`src/convolutional_neural_network.py`):  
   - Constructs a sequential/functional Keras model with stacked Conv2D, MaxPooling2D, BatchNormalization, Dropout, and Dense layers.
   - Compiles the model with a binary cross‑entropy loss and an optimizer (e.g., Adam).
4. **Training & Evaluation** (`src/convolutional_neural_network.py`):  
   - Calls `model.fit()` on the training dataset, supplying validation data.
   - Tracks metrics such as accuracy and loss via Keras callbacks (EarlyStopping, ModelCheckpoint).
   - After training, evaluates the final model on the held‑out test set and outputs classification reports and confusion matrices.
5. **Result Visualization** (`src/notebooks/convolutional_neural_network.ipynb`):  
   - Plots training curves, sample predictions, and performance metrics for stakeholder review.

All stages are orchestrated in a linear fashion; there is no external service or micro‑service communication.

## Key Design Decisions
- Monolithic pipeline – All stages live in a single Python module (`convolutional_neural_network.py`) to keep the project lightweight and easy to run on a personal workstation.
- Keras high‑level API – Chosen for rapid prototyping, clear layer semantics, and built‑in training utilities (fit, callbacks).
- TensorFlow `tf.data` pipelines – Provide efficient batched loading, prefetching, and optional GPU‑accelerated preprocessing.
- Image augmentation at training time – Improves generalisation without expanding the stored dataset.
- Model checkpointing with EarlyStopping – Prevents over‑fitting and reduces unnecessary epochs on limited hardware.
- Separate Jupyter notebook – Offers an interactive environment for exploratory data analysis, hyper‑parameter tuning, and result visualisation without affecting the core script.

## Scalability & Reliability
The current monolithic design targets single‑node execution, but several pathways exist to scale:
- **Hardware scaling** – Switching the TensorFlow runtime to GPU (via CUDA) or TPU yields up to 10‑30× faster training.
- **Data pipeline scaling** – Leveraging `tf.data` features like `cache()`, `shuffle(buffer_size)`, and `prefetch(tf.data.AUTOTUNE)` maximises CPU‑GPU overlap.
- **Distributed training** – Minimal code changes are required to enable `tf.distribute.MirroredStrategy` for multi‑GPU training or `tf.distribute.MultiWorkerMirroredStrategy` for multi‑node clusters.
- **Modularisation** – Extracting each logical layer into its own package (`data/`, `model/`, `training/`) would facilitate independent testing, CI pipelines, and reuse in other image‑classification projects.
- **Dataset versioning** – Integrating DVC or TensorFlow Datasets can manage large image collections and enable reproducible experiments.

Even without these extensions, the repository remains performant for typical educational or prototype workloads (e.g., 20k images, 32‑64 GB RAM, single GPU).
