# Developer Runbook – dog-vs-cat-cnn

## Prerequisites
- Git
- Python 3.8 or newer
- Virtual environment tool (venv, virtualenv, or conda)
- Internet connection (to install packages and optionally download the dataset)
- Optional: NVIDIA GPU with CUDA drivers (for faster training)

## Local Setup & Development
1. 1. **Clone the repository**
2.    ```bash
3.    git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
4.    cd dog-vs-cat-cnn
5.    ```
6. 
7. 2. **Create and activate a virtual environment** (recommended)
8.    ```bash
9.    # Using venv (built‑in)
10.    python3 -m venv .venv
11.    source .venv/bin/activate   # macOS / Linux
12.    .\\venv\\Scripts\\activate # Windows
13. 
14.    # Or using conda
15.    conda create -n dog-cat-cnn python=3.9 -y
16.    conda activate dog-cat-cnn
17.    ```
18. 
19. 3. **Install Python dependencies**
20.    ```bash
21.    pip install --upgrade pip
22.    pip install -r requirements.txt
23.    ```
24. 
25. 4. **(Optional) Install Jupyter Notebook for the exploratory notebook**
26.    ```bash
27.    pip install notebook  # already in requirements, but explicit if you need it
28.    ```
29. 
30. 5. **(Optional) Acquire the Dogs vs Cats dataset**
31.    - The training script expects a folder `data/` with sub‑folders `train/` and `validation/` containing images.
32.    - You can download the Kaggle “Dogs vs Cats” dataset and unzip it:
33.      ```bash
34.      mkdir -p data/train data/validation
35.      # Example using Kaggle CLI (requires kaggle API token)
36.      kaggle competitions download -c dogs-vs-cats
37.      unzip dogs-vs-cats.zip -d data
38.      # Then split the images into train/validation as you prefer
39.      ```
40.    - If you do not provide a dataset, the script will raise a clear `FileNotFoundError`.

## Running Tests
```bash
No automated test suite is shipped with this repository.  The primary way to verify a working environment is to run the training script (see *Local Development Loop* below) and confirm that it starts without import or runtime errors.
```

## Troubleshooting
### ImportError: No module named <package>
**Resolution:** Make sure you activated the virtual environment before running any Python command. Re‑run `source .venv/bin/activate` (or the conda activate command) and reinstall dependencies with `pip install -r requirements.txt`.

### torch.cuda.is_available() returns False on a GPU‑enabled machine
**Resolution:** Verify that the correct CUDA toolkit version is installed and that the NVIDIA driver matches the version required by the PyTorch wheel. Re‑install PyTorch with the appropriate CUDA build, e.g.: `pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html`.

### FileNotFoundError: [Errno 2] No such file or directory: 'data/train/'
**Resolution:** Create the expected data directory structure and place the training images there, or edit `src/convolutional_neural_network.py` to point to the correct path.

### MemoryError or extremely slow training on CPU
**Resolution:** Reduce the batch size in the script (e.g., `batch_size = 16`) or switch to a GPU. If you must stay on CPU, consider using a smaller model architecture.

### Jupyter notebook refuses to start or shows `ImportError` for project modules
**Resolution:** Start the notebook from the repository root so that `src/` is on the Python path, or add the repository root to `PYTHONPATH` before launching Jupyter:


