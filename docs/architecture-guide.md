# Technical Architecture Guide – Dog vs Cat CNN

## System Overview
The *dog-vs-cat-cnn* repository implements a convolutional neural network (CNN) that classifies images as either a dog or a cat. The codebase follows a monolithic ML‑pipeline pattern where data ingestion, preprocessing, model definition, training, and evaluation are bundled in a single Python module (`src/convolutional_neural_network.py`). An accompanying Jupyter notebook (`src/notebooks/convolutional_neural_network.ipynb`) provides an interactive environment for exploratory data analysis, step‑by‑step model training, and visualisation of results. The stack is pure Python, leveraging TensorFlow/Keras for model construction and training, and standard scientific libraries for data handling.

## System Layers
### Data Ingestion & Pre‑processing
**Technologies:** Python, TensorFlow, NumPy

Handles reading raw image files, applying resizing, normalisation, and optional augmentation. Implemented with TensorFlow's `image_dataset_from_directory` and `tf.image` utilities in `src/convolutional_neural_network.py`.

### Model Definition & Training
**Technologies:** Python, TensorFlow/Keras

Encapsulates the CNN architecture definition (convolutional, pooling, dense layers) and the training loop (compile, fit, callbacks). All logic resides in `src/convolutional_neural_network.py`.

### Model Persistence & Serving Artifact
**Technologies:** TensorFlow/Keras

Serialises trained weights to a portable `.h5` file using `model.save_weights`. This artifact can be loaded by downstream scripts or notebooks for inference without re‑training.

### Interactive Exploration & Reporting
**Technologies:** Jupyter Notebook, Python, Matplotlib, TensorBoard

Provides an iterative, visual workflow for data inspection, model training, and result visualisation. Implemented in `src/notebooks/convolutional_neural_network.ipynb` with inline Matplotlib/Seaborn plots and TensorBoard callbacks.



## Data Flow & Pipelines
1. **Dataset Loading** – Image files from the public Dog vs Cat dataset are read from a local directory (or downloaded via `tensorflow.keras.utils.get_file`). 2. **Pre‑processing** – Images are resized, normalised to [0,1] range, and optionally augmented (random flips, rotations). 3. **Train/Test Split** – A deterministic split creates training and validation subsets. 4. **Model Construction** – `src/convolutional_neural_network.py` defines a sequential CNN architecture (convolution → pooling → dense layers) using TensorFlow/Keras APIs. 5. **Training Loop** – The model is compiled with a loss function (`binary_crossentropy`), optimizer (e.g., `Adam`), and metric (`accuracy`). Training proceeds over a configurable number of epochs, with callbacks for early stopping and checkpointing. 6. **Model Persistence** – After training, model weights are saved to disk (`model.save_weights`), enabling later reuse. 7. **Evaluation & Visualisation** – The notebook loads the saved weights, runs inference on a held‑out set, and visualises metrics and sample predictions.

## Key Design Decisions
- Use of TensorFlow/Keras as the single deep‑learning framework to keep the stack lightweight and avoid multi‑framework integration overhead.
- Monolithic pipeline layout in `convolutional_neural_network.py` for simplicity; the entire flow (pre‑process → train → save) is executed sequentially, which eases reproducibility for a single‑researcher project.
- Separate Jupyter notebook for exploratory work, avoiding clutter in the production script while still leveraging the same source module for consistency.
- Model checkpointing via `ModelCheckpoint` callback ensures that the best‑performing weights are persisted, mitigating the risk of over‑fitting during longer training runs.
- Binary classification loss (`binary_crossentropy`) matches the two‑class problem, reducing computational cost compared to a categorical approach.

## Scalability & Reliability
The current monolithic design works well for the modest dataset size (~25k images) used in typical Dog vs Cat tutorials. To scale:
- **Data Parallelism**: Replace the single‑GPU training loop with `tf.distribute.MirroredStrategy` to leverage multiple GPUs or TPU pods.
- **Batch Size Tuning**: Larger batch sizes can improve GPU utilisation but require more memory; the script exposes `batch_size` as a configurable argument.
- **Dataset Pipeline Optimisation**: Persist pre‑processed TFRecord files and use `tf.data` pipelines with `prefetch` and `cache` to minimise I/O bottlenecks.
- **Modularisation**: Extract preprocessing, model architecture, and training utilities into separate modules (e.g., `src/preprocess.py`, `src/model.py`, `src/training.py`) to enable independent scaling of each stage and easier integration with orchestrators like Airflow or Kubeflow.
- **Model Export**: Convert the trained Keras model to TensorFlow SavedModel format for serving via TensorFlow Serving or Cloud AI Platform, facilitating production‑grade inference at scale.
