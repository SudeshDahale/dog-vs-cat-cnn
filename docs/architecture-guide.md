# Technical Architecture Guide for dog-vs-cat-cnn

## System Overview
The **dog-vs-cat-cnn** repository implements a binary image classification solution (dog vs cat) using a Convolutional Neural Network (CNN) built in pure Python. The codebase follows a monolithic, data‑science style architecture: all core logic lives under the **src/** package, while exploratory analysis, model training, and evaluation are performed interactively in a Jupyter notebook. The solution is intended for research and prototype‑level workloads and can be scaled to larger datasets or production environments with modest refactoring.

## System Layers
### Data Ingestion & Pre‑processing
**Technologies:** Python, NumPy, Pillow (PIL), scikit‑learn (train_test_split)

Handles file system traversal, image decoding, resizing, normalisation and optional augmentation. Implemented in pure Python functions inside `src/convolutional_neural_network.py`.

### Model Definition & Training
**Technologies:** TensorFlow, Keras, Python

Encapsulates the CNN architecture, loss function, optimizer, and training loop. Uses a high‑level deep‑learning API (TensorFlow/Keras) declared in `requirements.txt`.

### Evaluation & Experimentation
**Technologies:** Jupyter Notebook, matplotlib, seaborn, pandas

Interactive analysis performed in the Jupyter notebook. Loads the trained model, runs predictions, visualises results, and records experiment metrics.

### Orchestration & Utilities
**Technologies:** Python standard library (logging, argparse)

Utility functions for logging, checkpointing, and command‑line execution. All utilities are co‑located in the monolithic `src/` package to keep the prototype simple.



## Data Flow & Pipelines
1. **Data Acquisition** – Raw image files (dog and cat photos) are placed in a local directory structure (e.g., `data/train/dog`, `data/train/cat`).\n2. **Pre‑processing** – The script `src/convolutional_neural_network.py` loads images, resizes them to a fixed shape, normalises pixel values, and optionally applies augmentation. The processed arrays are split into training/validation sets.\n3. **Model Construction** – A CNN architecture is defined in the same module, using Keras/TensorFlow layers (or an equivalent deep‑learning backend as declared in `requirements.txt`).\n4. **Training** – The model is compiled with a binary‑cross‑entropy loss and fitted on the pre‑processed tensors. Callbacks such as `ModelCheckpoint` and `EarlyStopping` may be employed.\n5. **Evaluation** – After training, the notebook `src/notebooks/convolutional_neural_network.ipynb` loads the saved model, runs inference on a hold‑out test set, and computes metrics (accuracy, ROC‑AUC, confusion matrix). Visualisations of predictions are rendered inline.\n6. **Inference (optional)** – The trained model can be loaded in a Python script to classify new images on demand.

## Key Design Decisions
- Monolithic layout under `src/` – simplifies dependency management for a research prototype and avoids the overhead of a micro‑service split.
- Single‑file CNN implementation – keeps the model definition readable and reduces the cognitive load for newcomers.
- Use of Jupyter notebook for EDA and model validation – enables rapid iteration, visual feedback, and easy sharing of results with stakeholders.
- Binary classification with a single sigmoid output – aligns with the problem statement (dog vs cat) and reduces model complexity.
- Explicit checkpointing of the best model – ensures reproducibility and provides a fallback for later inference without retraining.

## Scalability & Reliability
Although the repository is designed as a prototype, several avenues exist to scale the solution:

* **Data Volume** – Replace in‑memory NumPy arrays with `tf.data.Dataset` pipelines or TensorFlow `TFRecord` files to stream large image collections without exhausting RAM.
* **Compute Resources** – Leverage GPU acceleration by running the training script on a machine with CUDA‑enabled TensorFlow. The code can be launched inside Docker containers or on managed services such as Google AI Platform.
* **Modularisation** – Extract the data pipeline, model definition, and training logic into separate packages or modules. This makes unit testing easier and supports reuse in other computer‑vision projects.
* **Hyper‑parameter Search** – Integrate with libraries such as `Keras Tuner` or `Optuna` to automate experimentation at scale.
* **Production Deployment** – Export the trained model to TensorFlow SavedModel or ONNX format and serve it via TensorFlow Serving, FastAPI, or a lightweight Flask endpoint.

By addressing these points, the codebase can evolve from a research notebook into a production‑ready image‑classification service.
