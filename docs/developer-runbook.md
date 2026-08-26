# Dog vs Cat CNN – Developer Runbook

## Prerequisites
- Python 3.8 or higher installed on your system.
- Git client for cloning the repository.
- Access to a GPU (optional but recommended) with CUDA drivers for accelerated training.
- Sufficient disk space to store the Cat vs Dog dataset (≈ 800 MB).

## Environment Variables
| Variable | Status | Description |
| :--- | :--- | :--- |
| `DATA_ROOT` | Optional | Absolute path to the root folder that contains the `train/` and `val/` sub‑folders. If omitted, the code falls back to `./data` relative to the repository root. |


## Local Setup & Development
1. 1. **Clone the repository**
   ```bash
   git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
   cd dog-vs-cat-cnn
   ```
2. 2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```
3. 3. **Install the Python dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. 4. **Download the dataset**
   - The project expects the dataset under `data/` with the following structure:
     ```
     data/
       train/
         cats/
         dogs/
       val/
         cats/
         dogs/
     ```
   - If you do not have the dataset, download it from Kaggle ("Dogs vs. Cats") and extract it to the `data/` folder.
   - Alternatively, set the `DATA_ROOT` environment variable to point to any directory that follows the above layout.
   
5. 5. **Verify the installation**
   ```bash
   python -c "import src.convolutional_neural_network as cnn; print('Import OK')"
   ```
6. 6. **Run the notebook (optional)**
   ```bash
   jupyter notebook src/notebooks/convolutional_neural_network.ipynb
   ```

## Running Tests
```bash
python -m unittest discover -s tests || echo 'No test suite found – repository is primarily experimental.'
```

## Troubleshooting
### ImportError: No module named 'src'
**Resolution:** Make sure you are running commands from the repository root and that the virtual environment is activated. The `src` package is a top‑level module; you can also add the repository root to `PYTHONPATH`:
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### torch.cuda.is_available() returns False despite having a GPU
**Resolution:** Verify that the correct CUDA toolkit version matches the PyTorch wheel installed. Reinstall PyTorch with the appropriate CUDA version, e.g.:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0 --extra-index-url https://download.pytorch.org/whl/cu121
```

### FileNotFoundError: Dataset directory not found
**Resolution:** Either place the dataset under `./data` following the expected structure or set the `DATA_ROOT` environment variable to the correct location before running any script.
```bash
export DATA_ROOT=/path/to/your/dataset
```

### MemoryError while loading images
**Resolution:** The data‑preprocessing module loads images in batches. Reduce the `BATCH_SIZE` constant in `src/convolutional_neural_network.py` or increase your system’s swap space.


