# Developer Runbook – Dog vs Cat CNN

## Prerequisites
- Git
- Python 3.8 or newer
- pip (compatible with the Python version)
- Virtual environment tool (e.g., `venv` or `conda`)
- Optional: NVIDIA GPU with CUDA toolkit if you intend to train on GPU

## Environment Variables
| Variable | Status | Description |
| :--- | :--- | :--- |
| `DATASET_ROOT` | Optional | Absolute path to the root folder that contains `train/` and `validation/` sub‑folders. If omitted, the code falls back to a relative `./data` path. |


## Local Setup & Development
1. 1. **Clone the repository**
2.    ```
3.    git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
4.    cd dog-vs-cat-cnn
5.    ```
6. 
7. 2. **Create and activate a virtual environment** (recommended)
8.    ```
9.    python -m venv .venv
10.    # On Linux / macOS
11.    source .venv/bin/activate
12.    # On Windows
13.    .venv\Scripts\activate
14.    ```
15. 
16. 3. **Install Python dependencies**
17.    ```
18.    pip install --upgrade pip
19.    pip install -r requirements.txt
20.    ```
21. 
22. 4. **(Optional) Install CUDA / cuDNN** if you plan to use the GPU. Ensure the versions match the TensorFlow / PyTorch wheel pulled in via `requirements.txt`.
23. 
24. 5. **Prepare the dataset** – the training script expects a directory structure similar to:
25.    ```
26.    data/
27.      train/
28.        dogs/   <image files>
29.        cats/   <image files>
30.      validation/
31.        dogs/   <image files>
32.        cats/   <image files>
33.    ```
34.    If you store the data elsewhere, set the `DATASET_ROOT` environment variable to point at the top‑level `data` folder (see *Environment Variables* below).

## Running Tests
```bash
Run the main training script to verify the installation:
```bash
python src/convolutional_neural_network.py
```
The script will load the dataset (or abort with a clear error if the path is missing), build the model, and start a short training run. Successful execution ends with a printed summary of the model architecture and final loss/accuracy values.

You can also launch the exploratory notebook to visualise the model and sample predictions:
```bash
jupyter notebook src/notebooks/convolutional_neural_network.ipynb
```
```

## Troubleshooting
### ImportError / ModuleNotFoundError for a package listed in `requirements.txt`
**Resolution:** Make sure the virtual environment is activated and re‑run `pip install -r requirements.txt`. If the problem persists, upgrade `pip` and retry.

### CUDA‑related errors (e.g., `Failed to load the CUDA runtime library`)
**Resolution:** Verify that the NVIDIA driver, CUDA toolkit, and cuDNN versions are compatible with the deep‑learning framework version installed. If you lack a GPU, force CPU execution by setting the environment variable `CUDA_VISIBLE_DEVICES=` (empty) before running the script.

### FileNotFoundError: dataset directory not found
**Resolution:** Either place the dataset under `./data` following the expected structure or set the `DATASET_ROOT` environment variable to the correct path.

### Out‑of‑memory (OOM) error during training
**Resolution:** Reduce the batch size in `src/convolutional_neural_network.py`, or switch to CPU execution which uses less GPU memory. Alternatively, ensure no other GPU‑intensive processes are running.

### Jupyter notebook refuses to start or cannot locate the kernel
**Resolution:** Install the IPython kernel inside the virtual environment: `pip install ipykernel && python -m ipykernel install --user --name=dogcat-env`. Then select the `dogcat‑env` kernel inside the notebook UI.


