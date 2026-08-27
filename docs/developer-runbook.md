# Developer Runbook – dog-vs-cat-cnn

## Prerequisites
- Git
- Python 3.8 or newer
- Virtual environment tool (venv, virtualenv, or conda)
- Git Large File Storage (optional, if dataset stored via LFS)
- Access to a GPU (recommended for training, but not required for development)

## Environment Variables
| Variable | Status | Description |
| :--- | :--- | :--- |
| `DATA_ROOT` | Optional | Root directory for the dataset. If set, the training script will read images from `${DATA_ROOT}/train` and `${DATA_ROOT}/validation`. |


## Local Setup & Development
1. 1. Clone the repository:
   ```bash
   git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
   cd dog-vs-cat-cnn
   ```
2. 2. Create and activate a virtual environment:
3.    ```bash
   python -m venv .venv
   # On macOS/Linux
   source .venv/bin/activate
   # On Windows
   .venv\Scripts\activate
   ```
4. 3. Install the Python dependencies defined in `requirements.txt`:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. 4. (Optional) Install Jupyter Lab/Notebook if you intend to run the exploratory notebook:
   ```bash
   pip install notebook   # or `pip install jupyterlab`
   ```
6. 5. Download the Dogs vs. Cats dataset and place it under a `data/` directory at the repository root. The expected structure is:
   ```
   data/
   ├── train/
   │   ├── dogs/
   │   └── cats/
   └── validation/
       ├── dogs/
       └── cats/
   ```
7. 6. Verify the installation by running a quick sanity‑check script (provided in `src/convolutional_neural_network.py`):
   ```bash
   python - <<'PY'
   from src.convolutional_neural_network import build_model
   model = build_model()
   model.summary()
   PY
   ```

## Running Tests
```bash
There are no dedicated unit‑test files in this repository. The recommended "test" is to run a short training epoch to ensure the pipeline works:
```bash
python src/convolutional_neural_network.py \
    --data-dir data \
    --epochs 1 \
    --batch-size 32
```
If the script finishes without errors and prints validation accuracy, the development environment is correctly set up.
```

## Troubleshooting
### ImportError: No module named 'src'
**Resolution:** Make sure you are running commands from the repository root and that the virtual environment is activated. The `src` directory is a package; you may need to add the repository root to `PYTHONPATH`:
```bash
export PYTHONPATH=$(pwd)
```

### ModuleNotFoundError: No module named 'tensorflow' (or 'torch')
**Resolution:** The project depends on a deep‑learning framework listed in `requirements.txt`. Re‑run `pip install -r requirements.txt` inside the activated virtual environment. Verify the correct Python version (>=3.8).

### FileNotFoundError: Dataset directory not found
**Resolution:** Ensure the Dogs vs. Cats dataset is downloaded and placed under a `data/` folder as described in step 5. Alternatively, set the `DATA_ROOT` environment variable to point to the correct location.

### Jupyter notebook cannot find `src` modules
**Resolution:** Launch the notebook from the repository root and set the kernel’s working directory accordingly. You can also add the following at the top of the notebook:
```python
import sys, os
sys.path.append(os.path.abspath('..'))
```

### Training is extremely slow or runs out of memory
**Resolution:** If a GPU is available, ensure the appropriate drivers and CUDA/cuDNN libraries are installed. Otherwise, reduce `--batch-size` or use a smaller image size in the training script.


