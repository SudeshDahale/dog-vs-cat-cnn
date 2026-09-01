# Technical Architecture Guide – dog-vs-cat-cnn

## System Overview
The **dog-vs-cat-cnn** repository provides a monolithic, prototype‑grade machine‑learning system for binary image classification (dog vs. cat). Implemented in pure Python, the solution consists of a single codebase under `src/` that defines the convolutional neural network (CNN), handles data loading, preprocessing, augmentation, model training, evaluation, and inference. Interactive exploration and documentation are facilitated via a Jupyter notebook in `src/notebooks/`. The architecture follows a straightforward linear pipeline suitable for rapid experimentation and educational purposes.

## System Layers
### Data Layer
**Technologies:** Python, TensorFlow/Keras, NumPy, Pillow

Handles raw image discovery, loading, and on‑the‑fly augmentation. Implemented via Keras `ImageDataGenerator` in `convolutional_neural_network.py`.

### Model Layer
**Technologies:** TensorFlow/Keras

Defines the CNN architecture, compiles the model, and encapsulates training logic. All code resides in `src/convolutional_neural_network.py`.

### Evaluation & Inference Layer
**Technologies:** Matplotlib, scikit-learn (optional for confusion matrix)

Contains functions for assessing validation metrics, generating plots, and running predictions on unseen images.

### Presentation Layer
**Technologies:** Jupyter Notebook, IPython, Matplotlib

A Jupyter notebook (`src/notebooks/convolutional_neural_network.ipynb`) that ties together data preparation, model training, evaluation, and inference for interactive experimentation.



## Data Flow & Pipelines
The data flow description is included above in the `data_flow` field.

## Key Design Decisions
- Use of a **single monolithic script** (`convolutional_neural_network.py`) to keep the prototype lightweight and easy to run end‑to‑end.
- Leverage **Keras high‑level API** for rapid model definition and data pipeline construction, reducing boilerplate code.
- Adopt **ImageDataGenerator** for on‑the‑fly augmentation, avoiding the need for a separate preprocessing step and minimizing I/O overhead.
- Training hyper‑parameters (batch size, epochs, learning rate) are hard‑coded for reproducibility, reflecting the prototype nature of the project.
- All configuration (paths, image size, class mode) is defined at the top of the script, enabling quick adjustments without external config files.

## Scalability & Reliability
The current monolithic design is sufficient for the modest dataset used in the classic dog‑vs‑cat challenge (≈25 k images). For larger datasets or production use, scalability can be addressed by:
* **Data Layer** – Switching to `tf.data.Dataset` pipelines for better parallelism and prefetching.
* **Model Layer** – Modularizing the model definition into separate files and employing model checkpointing to resume long‑running training on distributed GPU clusters.
* **Training** – Using mixed‑precision training and multi‑GPU strategies (MirroredStrategy) to accelerate epochs.
* **Inference Service** – Extracting the inference utilities into a lightweight Flask/FastAPI microservice, enabling RESTful predictions.
* **Experiment Tracking** – Integrating with MLflow or TensorBoard for systematic logging of hyper‑parameters and metrics.
These incremental steps preserve the existing codebase while allowing the system to grow beyond the prototype scope.
