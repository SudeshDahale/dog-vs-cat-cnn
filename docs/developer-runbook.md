# Dog vs Cat CNN Repository – Developer Runbook

## Prerequisites
- Git installed (for cloning the repository)
- Python 3.9+ installed and available in PATH
- pip (Python package installer)
- Virtual environment tool (venv or conda)
- For GPU acceleration (optional): NVIDIA driver, CUDA Toolkit, and cuDNN compatible with TensorFlow version specified in requirements.txt

## Environment Variables
| Variable | Status | Description |
| :--- | :--- | :--- |
| `DATA_DIR` | Optional | Path to the root folder containing the `train` and `validation` sub‑folders of the dog‑vs‑cat image dataset. If omitted, the script defaults to `./data`. |
| `MODEL_OUTPUT_DIR` | Optional | Directory where trained model weights and checkpoint files will be saved. Defaults to `./models` if not set. |


## Local Setup & Development
1. 1. Clone the repository:
   ```bash
   git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
   cd dog-vs-cat-cnn
   ```
2. 2. Create and activate a virtual environment (using venv):
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows use `.venv\Scripts\activate`
   ```
3. 3. Install the Python dependencies defined in `requirements.txt`:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. 4. Verify TensorFlow installation (CPU vs GPU). In a Python REPL run:
   ```python
   import tensorflow as tf
   print(tf.__version__)
   print('GPU available:', tf.config.list_physical_devices('GPU'))
   ```
5. 5. (Optional) Download the Dog vs Cat dataset if not already present. The repository expects the dataset under `data/` – follow the instructions in the notebook `src/notebooks/convolutional_neural_network.ipynb` to download and extract it.
6. 6. Run the model script to ensure the pipeline works end‑to‑end:
   ```bash
   python src/convolutional_neural_network.py --train --epochs 1
   ```
   This will train for a single epoch and write model weights to `models/` (or the path defined inside the script).
7. 7. Open the exploratory Jupyter notebook for interactive development:
   ```bash
   jupyter notebook src/notebooks/convolutional_neural_network.ipynb
   ```
   Execute cells sequentially to explore data loading, model architecture, training metrics, and visualisations.

## Running Tests
```bash
There are no dedicated unit tests for this repository. A quick sanity‑check can be performed by running the script with a reduced dataset and a single epoch:
```bash
python src/convolutional_neural_network.py --train --epochs 1 --batch_size 16
```
Successful execution and the creation of a weight file indicates the core pipeline is functional.
```

## Troubleshooting
### ImportError: No module named 'tensorflow'
**Resolution:** Ensure the virtual environment is activated and that `tensorflow` (or `tensorflow-gpu` for GPU support) is listed in `requirements.txt`. Re‑run `pip install -r requirements.txt` inside the active environment.

### TensorFlow reports "Failed to load the CUDA driver" even though a GPU is present.
**Resolution:** Confirm that the NVIDIA driver, CUDA Toolkit, and cuDNN versions match the TensorFlow build you installed. Refer to TensorFlow's compatibility matrix and install the matching versions, or fall back to CPU‑only TensorFlow by installing `tensorflow-cpu`.

### Training script aborts with "FileNotFoundError: [Errno 2] No such file or directory: 'data/..."
**Resolution:** Set the `DATA_DIR` environment variable to point at the folder that contains the extracted image dataset, or place the dataset under the repository's default `./data` directory. Follow the download instructions inside the Jupyter notebook.

### Jupyter notebook cannot find the `src` module when importing `convolutional_neural_network`.
**Resolution:** Start the notebook from the repository root (`dog-vs-cat-cnn`) so that the Python path includes the `src` package, or add the following to a notebook cell before imports:
```python
import sys, os
sys.path.append(os.path.abspath('../src'))
```


