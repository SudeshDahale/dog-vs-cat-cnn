# Technical Architecture Guide for dog-vs-cat-cnn

## System Overview
The *dog-vs-cat-cnn* repository implements a monolithic machine‑learning pipeline in Python for binary image classification (dog vs. cat). Core functionality lives in `src/convolutional_neural_network.py`, which defines, trains, and persists a convolutional neural network (CNN). Interactive data exploration, model training runs and result visualisation are provided through the Jupyter notebook `src/notebooks/convolutional_neural_network.ipynb`. The repository follows a straightforward, single‑process architecture suitable for prototyping and small‑scale experiments.

## System Layers
### Data Ingestion & Preprocessing
**Technologies:** Python, NumPy, Pillow, scikit-learn

Loads raw image files from a user‑specified directory, rescales them to a uniform size, normalises pixel values, and optionally augments the dataset (flip, rotation, zoom). This logic is encapsulated in helper functions within `convolutional_neural_network.py` and reused by the notebook for quick visual checks.

### Model Definition & Training
**Technologies:** Python, TensorFlow/Keras, NumPy

Defines a Keras Sequential/Functional CNN architecture (conv layers, pooling, dropout, dense output) tailored for binary classification. Handles compilation (binary cross‑entropy, Adam optimizer) and executes model.fit() with training/validation splits. Model artefacts (weights, architecture) are saved to disk (`model.h5`).

### Experiment & Visualization
**Technologies:** Python, Jupyter, Matplotlib, Seaborn

The Jupyter notebook (`src/notebooks/convolutional_neural_network.ipynb`) imports the training module, runs experiments with different hyper‑parameters, and visualises training history, confusion matrices and sample predictions using Matplotlib/Seaborn.

### Model Persistence & Deployment Stubs
**Technologies:** Python, TensorFlow/Keras

After training, the model is persisted as an HDF5 file (`model.h5`). While the repository does not include a serving layer, the saved model can be loaded by any downstream Python service or exported to TensorFlow SavedModel format for future deployment.



## Data Flow & Pipelines
1. **Dataset Acquisition** – User places dog and cat images in a folder structure (e.g., `data/train/dog`, `data/train/cat`). 2. **Loading** – `load_images()` reads files, converts them to NumPy arrays, and assigns binary labels (0=cat, 1=dog). 3. **Preprocessing** – Images are resized (e.g., 150×150), pixel values scaled to [0,1], and optionally shuffled/augmented. 4. **Train‑Validation Split** – `train_test_split` creates separate sets. 5. **Model Construction** – CNN architecture built in `convolutional_neural_network.py`. 6. **Training** – `model.fit()` consumes the preprocessed tensors, logs loss/accuracy per epoch, and validates on the hold‑out set. 7. **Evaluation & Visualization** – Notebook loads the trained model, computes metrics, plots learning curves, and displays sample predictions. 8. **Persistence** – `model.save('model.h5')` writes the trained artefact for reuse.

## Key Design Decisions
- Keep the entire pipeline in a single Python module to simplify reproducibility for academic/learning purposes.
- Choose a binary‑crossentropy loss with a sigmoid output for clear dog vs. cat separation.
- Persist the trained model as an HDF5 file (`model.h5`) for easy loading in both script and notebook contexts.
- Leverage a Jupyter notebook for exploratory analysis, allowing rapid iteration on hyper‑parameters without modifying source code.
- Use standard Python data‑science libraries (NumPy, Pillow, scikit‑learn) to avoid heavy external dependencies.

## Scalability & Reliability
The current monolithic design is optimal for small datasets and single‑GPU environments. To scale:
- **Data Pipeline**: Replace in‑memory NumPy arrays with TensorFlow `tf.data.Dataset` pipelines and TFRecord files to stream large image collections.
- **Distributed Training**: Adopt `tf.distribute.Strategy` (e.g., MirroredStrategy) to leverage multiple GPUs or multi‑node clusters.
- **Modularisation**: Split the training logic into separate packages (data loader, model factory, trainer) to enable independent testing and CI integration.
- **Model Serving**: Export the model to TensorFlow SavedModel format and serve via TensorFlow Serving or a lightweight Flask/FASTAPI endpoint for production inference.
- **Experiment Tracking**: Integrate with MLflow or TensorBoard for systematic logging of hyper‑parameters and metrics as the experiment count grows.
