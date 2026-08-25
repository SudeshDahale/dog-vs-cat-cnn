# Technical Architecture Guide – Dog vs Cat CNN

## System Overview
This repository implements a monolithic machine‑learning application for training a convolutional neural network (CNN) that classifies images of dogs and cats. The codebase is organised around two primary modules – the model/training logic in src/convolutional_neural_network.py and an interactive Jupyter notebook (src/notebooks/convolutional_neural_network.ipynb) for experimentation, visualization and ad‑hoc training runs. All components are written in Python and rely on standard scientific‑ML libraries.

## System Layers
### Data Ingestion & Pre‑processing
**Technologies:** Python, NumPy, Pillow, TensorFlow/Keras ImageDataGenerator

Loads the dog‑cat image dataset from a local directory, applies deterministic train/validation splits, and performs image resizing, normalization and optional augmentation. The logic lives in the data‑handling section of convolutional_neural_network.py and is invoked by the training script. 

### Model Definition
**Technologies:** Python, TensorFlow, Keras

Encapsulates the CNN architecture – a stack of Conv2D, MaxPooling, BatchNormalization and Dense layers – in a reusable function/class. The definition is pure Python/TensorFlow code, allowing the notebook or the training script to instantiate the model with a single call.

### Training Engine
**Technologies:** Python, TensorFlow, Keras, TensorBoard

Orchestrates the end‑to‑end training loop: compiles the model with loss and optimizer, feeds pre‑processed batches from the data pipeline, tracks metrics, and persists checkpoints and final weights to the src/ directory. Configurable hyper‑parameters (learning rate, epochs, batch size) are declared near the top of the script for easy tweaking.

### Notebook UI & Visualization
**Technologies:** Jupyter Notebook, Python, matplotlib, seaborn

Provides an interactive Jupyter notebook that imports the same model and data utilities, runs quick training experiments, and visualises loss/accuracy curves, sample predictions and model architecture diagrams. This layer is a thin wrapper around the core library, ensuring reproducibility while giving data scientists a sandbox.



## Data Flow & Pipelines
1. The entry point (either the Python script or the notebook) invokes the data ingestion routine, which reads raw image files, applies resizing (e.g., 150×150) and pixel scaling (0‑1). 2. A TensorFlow/Keras ImageDataGenerator yields batched tensors for the training and validation splits. 3. The training engine creates the CNN model via the Model Definition layer, compiles it, and streams batches from the generator into the model.fit() loop. 4. During each epoch, training metrics are logged to TensorBoard and optional callbacks (e.g., ModelCheckpoint) persist the best weights. 5. After training, the notebook can reload the saved model, run inference on a held‑out test set, and render visualisations of predictions and performance metrics.

## Key Design Decisions
- Separate concerns: data handling, model architecture, and training orchestration live in distinct code blocks within a single module for clarity while keeping the monolith simple.
- Reuse the same Python functions in both the script and the notebook to guarantee that experimental runs mirror production training.
- Leverage TensorFlow/Keras high‑level APIs (ImageDataGenerator, model.fit) to minimise boilerplate and focus on architecture experimentation.
- Persist model checkpoints and final weights to enable incremental training and reproducibility.
- Include deterministic seeding (numpy.random.seed, tf.random.set_seed) to make notebook experiments repeatable.

## Scalability & Reliability
The current monolithic design runs on a single machine but scales horizontally by: • Switching the ImageDataGenerator to tf.data pipelines for better CPU/GPU throughput; • Running the training script on machines equipped with NVIDIA GPUs to accelerate convolutional kernels; • Increasing batch size up to GPU memory limits; • Extending the training engine with TensorFlow’s tf.distribute.Strategy (e.g., MirroredStrategy) for multi‑GPU or multi‑node training without changing the core model code; • Externalising the dataset to cloud storage (e.g., S3) and using streaming data loaders to handle larger image corpora.
