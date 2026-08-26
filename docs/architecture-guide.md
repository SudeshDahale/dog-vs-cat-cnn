# Technical Architecture Guide for dog-vs-cat-cnn

## System Overview
The dog-vs-cat-cnn repository implements a monolithic machine‑learning pipeline for binary image classification (dogs vs. cats). The codebase is written in Python and organized as a single application that handles data preprocessing, model definition/training, and evaluation/inference. All components live under the `src/` directory, with the main CNN implementation in `src/convolutional_neural_network.py` and an accompanying Jupyter notebook (`src/notebooks/convolutional_neural_network.ipynb`) for exploratory analysis.

The pipeline follows a linear data‑flow: raw image files → preprocessing → train/validation split → CNN definition → training → evaluation → inference. The repository ships a `README.md` for usage instructions and a `requirements.txt` that pins the Python libraries required (e.g., TensorFlow/Keras, NumPy, pandas, scikit‑learn, Pillow).

## System Layers
### Data Layer
**Technologies:** Python, Pillow, NumPy, scikit-learn

Handles raw image loading, resizing, normalization, and train/validation splitting. Implemented with Pillow (image I/O) and NumPy for array manipulation; scikit‑learn assists with data splitting.

### Model Layer
**Technologies:** TensorFlow, Keras

Defines the CNN architecture, compiles the model, and runs the training loop. Utilises Keras (high‑level TensorFlow API) for model building, callbacks for checkpointing, and TensorBoard for optional logging.

### Evaluation & Inference Layer
**Technologies:** TensorFlow, Keras, scikit-learn

Loads the persisted model, computes performance metrics on the validation set, and provides a prediction API for new images. Uses Keras utilities for evaluation and standard Python for result formatting.



## Data Flow & Pipelines
1. **Data Ingestion** – The pipeline starts by loading raw image files (dog and cat pictures) from a local directory specified by the user.
2. **Preprocessing (`data‑preprocessing` module)** – Images are resized to a uniform shape, pixel values are normalized (e.g., scaling to [0,1]), and labels are encoded. The preprocessed data is then split into training and validation sets (typically using scikit‑learn's `train_test_split`).
3. **Model Definition & Training (`model‑definition‑training` module)** – `src/convolutional_neural_network.py` defines a sequential Keras CNN architecture (convolutional layers, pooling, dropout, dense layers). The model is compiled with a binary‑crossentropy loss and an optimizer (e.g., Adam). Training runs for a configurable number of epochs, persisting the best model checkpoint to disk.
4. **Evaluation & Inference (`evaluation‑inference` module)** – After training, the saved model is loaded to compute validation accuracy, loss curves, and a classification report. A helper function accepts a new image path, applies the same preprocessing steps, and returns the predicted class (dog or cat).
5. **Artifacts** – Trained model files (`.h5` or TensorFlow SavedModel) and optional plots (training history) are stored alongside the source code for downstream use.

## Key Design Decisions
- Monolithic organization – All pipeline stages reside in a single repository and share common configuration, simplifying reproducibility for a small‑scale academic project.
- Explicit preprocessing function – Guarantees that training and inference data undergo identical transformations, reducing data‑drift risk.
- Keras Sequential API – Chosen for its readability and rapid prototyping; suitable for the relatively shallow network required for the dog‑vs‑cat task.
- Model checkpoint callback – Persists the best validation loss model, enabling easy rollback and inference without retraining.
- Requirements pinning – `requirements.txt` ensures consistent library versions across environments, critical for reproducible model behavior.

## Scalability & Reliability
The current monolithic design works well for a modest dataset (few thousand images). To scale:
- **Data volume**: Replace local file loading with a data ingestion layer that reads from cloud storage (e.g., GCS/AWS S3) and streams batches using `tf.data` pipelines.
- **Model complexity**: Swap the Sequential model for a functional API or TensorFlow Estimator to support larger architectures and distributed training.
- **Compute**: Leverage TensorFlow's multi‑GPU strategy (`tf.distribute.MirroredStrategy`) or shift to a managed training service (e.g., Google AI Platform) by extracting the training logic into a separate script.
- **Modularity**: Break the monolith into micro‑services (preprocessing service, training service, inference service) and containerize each with Docker, orchestrated by Kubernetes for horizontal scaling.
- **Experiment tracking**: Integrate MLflow or TensorBoard for systematic logging of hyper‑parameters, metrics, and artifacts when scaling experiments.
