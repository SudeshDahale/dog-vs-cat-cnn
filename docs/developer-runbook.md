# Developer Runbook – dog-vs-cat-cnn

## Prerequisites
- Git installed (for cloning the repository)
- Python 3.8 or newer
- pip (Python package installer)
- Virtual environment tool (venv, virtualenv, or conda)
- Access to a GPU with CUDA drivers (optional, for faster training)

## Environment Variables
| Variable | Status | Description |
| :--- | :--- | :--- |
| `DATA_ROOT` | Optional | Root directory that contains the `train` and `val` sub‑folders with `dogs` and `cats` images. If not set, the code falls back to `./data`. |


## Local Setup & Development
1. 1. Clone the repository:
   ```bash
   git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
   cd dog-vs-cat-cnn
   ```
2. 2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv   # or `conda create -n dogcat python=3.9`
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```
3. 3. Install the required Python packages:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. 4. Verify the installation by importing the model module:
   ```bash
   python -c "import src.convolutional_neural_network as cnn; print('Import OK')"
   ```
5. 5. (Optional) If you have a CUDA‑capable GPU and want to use it, ensure the correct PyTorch build is installed. The `requirements.txt` pins the CPU‑only build; replace it with the appropriate `torch` wheel from https://pytorch.org/ for your CUDA version and reinstall:
6.    ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118   # example for CUDA 11.8
   ```
7. 6. (Optional) Download or prepare the dog/cat image dataset. The training script expects a folder structure:
   ```
   data/
     train/
       dogs/
       cats/
     val/
       dogs/
       cats/
   ```
   Adjust the path in `src/convolutional_neural_network.py` if your data lives elsewhere.

## Running Tests
```bash
python -m unittest discover -s tests || echo "No test suite found – run a quick training sanity check instead:"

# Quick sanity check (runs a single epoch on a small subset)
python src/convolutional_neural_network.py --epochs 1 --batch-size 8 --debug
```

## Troubleshooting
### ImportError: No module named 'torch'
**Resolution:** Ensure the virtual environment is activated and `torch` is installed via `pip install -r requirements.txt`. If you need GPU support, reinstall the correct CUDA‑compatible torch wheel as described in step 5.

### FileNotFoundError when loading data
**Resolution:** Set the `DATA_ROOT` environment variable to point at the directory containing the `train`/`val` sub‑folders, or place the dataset under a `data/` folder at the project root.

### CUDA out of memory error during training
**Resolution:** Reduce the batch size (`--batch-size` flag) or switch to CPU by uninstalling the GPU torch build and reinstalling the CPU‑only version (`pip install torch==<version>+cpu`).

### Jupyter notebook kernel dies when opening `src/notebooks/convolutional_neural_network.ipynb`
**Resolution:** Make sure the notebook is launched from the activated virtual environment:
   ```bash
   source .venv/bin/activate
   jupyter notebook
   ```
   Also verify that the notebook's import cells reference `src.convolutional_neural_network` correctly.


