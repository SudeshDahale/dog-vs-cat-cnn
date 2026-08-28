# Technical Architecture Guide for dog-vs-cat-cnn

## System Overview
The repository implements a monolithic machine‑learning pipeline in Python that trains a convolutional neural network (CNN) to classify dog and cat images. All code resides under the src package, while exploratory analysis lives in a Jupyter notebook. The pipeline follows a linear flow: data loading → preprocessing → model definition → training → evaluation.

## System Layers
### Data Loading & Preprocessing
**Technologies:** Python, NumPy, Pillow, scikit-learn

Loads image files from the dataset directory, resizes, normalises and augments them using NumPy/Pillow. Exposed through functions in src/convolutional_neural_network.py.

### Model Definition & Training
**Technologies:** Python, TensorFlow, Keras, NumPy

Defines the CNN architecture (Conv2D, MaxPooling, Dense layers) and runs the training loop with back‑propagation, loss calculation and metric logging.

### Experimentation & Visualization
**Technologies:** Jupyter, Matplotlib, Seaborn

Interactive Jupyter notebook (src/notebooks/convolutional_neural_network.ipynb) that imports the pipeline modules, visualises training curves, and performs quick inference on sample images.

### Packaging & Dependencies
**Technologies:** pip, requirements.txt

Project metadata in README.md and requirements.txt ensures reproducible environment setup.



## Data Flow & Pipelines
Raw image files → Data Loading module reads files → Preprocessing functions resize and normalise → Preprocessed tensors are passed to the Model Definition module → Training loop consumes tensors, updates weights, writes checkpoints → After training, the notebook loads the saved model to run inference and plot metrics.

## Key Design Decisions
- Keep the entire pipeline in a single repository for simplicity (monolith).
- Use a single Python module (src/convolutional_neural_network.py) to expose both data utilities and model functions, reducing import overhead.
- Separate exploratory code into a notebook to avoid polluting production scripts.
- Pin exact library versions in requirements.txt to guarantee reproducibility.

## Scalability & Reliability
The current design is suitable for a single‑GPU workstation. To scale horizontally, data loading could be parallelised with tf.data or multiprocessing, model code can be moved to a separate package, and training can be dispatched to distributed strategies (e.g., TensorFlow MirroredStrategy) without changing the external API.
