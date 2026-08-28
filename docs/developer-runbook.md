# Dog vs Cat CNN - Developer Runbook

## Prerequisites
- Git
- Python 3.9+
- Virtual environment tool (venv or conda)
- Git LFS (optional, if the image dataset is stored as large files)
- NVIDIA GPU with CUDA drivers (recommended for training speed) – optional for CPU‑only runs

## Environment Variables
| Variable | Status | Description |
| :--- | :--- | :--- |
| `DATA_DIR` | Optional | Path to the root folder that contains the `dogs/` and `cats/` sub‑folders. If not set, the code defaults to `./data`. |
| `CUDA_VISIBLE_DEVICES` | Optional | Comma‑separated list of GPU IDs to make visible to PyTorch. Useful on multi‑GPU machines. |


## Local Setup & Development
1. 1. **Clone the repository**
2.    ```bash
3.    git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
4.    cd dog-vs-cat-cnn
5.    ```
6. 
7. 2. **Create and activate a virtual environment**
8.    ```bash
9.    # Using venv
10.    python -m venv .venv
11.    source .venv/bin/activate   # On Windows: .venv\Scripts\activate
12. 
13.    # OR using conda
14.    conda create -n dogcat-cnn python=3.9 -y
15.    conda activate dogcat-cnn
16.    ```
17. 
18. 3. **Install Python dependencies**
19.    ```bash
20.    pip install --upgrade pip
21.    pip install -r requirements.txt
22.    ```
23. 
24. 4. **Prepare the image dataset**
25.    - The repository expects a directory named `data/` with two sub‑folders:
26.      - `data/dogs/` – containing dog images
27.      - `data/cats/` – containing cat images
28.    - If the dataset is not included, download a public dog‑vs‑cat dataset (e.g., Kaggle’s *Dogs vs. Cats* dataset) and place the extracted images in the structure above.
29.    - Optionally, create a symbolic link if you keep the data elsewhere:
30.      ```bash
31.      ln -s /path/to/your/dataset data
32.      ```
33. 
34. 5. **(Optional) Verify GPU availability**
35.    ```bash
36.    python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
37.    ```
38.    If you see `CUDA available: False` but you have a GPU, ensure the correct CUDA toolkit and matching PyTorch version are installed.

## Running Tests
```bash
```bash
# Run a quick sanity‑check training for 1 epoch on a small subset of data
python src/convolutional_neural_network.py --epochs 1 --batch-size 16 --debug

# Open the experiment notebook (requires Jupyter) to visualise results
jupyter notebook src/notebooks/convolutional_neural_network.ipynb
```
```

## Troubleshooting
### ImportError: No module named 'torch'
**Resolution:** Make sure the virtual environment is activated and that `requirements.txt` was installed. Run `pip install torch` with a version compatible with your CUDA toolkit (e.g., `pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html`).

### FileNotFoundError: Unable to locate dataset directory
**Resolution:** Create the `data/` folder with `dogs/` and `cats/` sub‑folders, or set the `DATA_DIR` environment variable to point at the correct location.

### torch.cuda.OutOfMemoryError: CUDA out of memory
**Resolution:** Reduce the batch size (`--batch-size`), or run the script on CPU by adding `--device cpu` (if the script supports it).

### Jupyter notebook does not start or cannot import the project modules
**Resolution:** Launch Jupyter from the activated virtual environment (`jupyter notebook`). If import errors persist, add the repository root to `PYTHONPATH`:
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```


