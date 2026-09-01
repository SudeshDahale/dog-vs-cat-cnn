# Dog vs Cat CNN - Developer Runbook

## Prerequisites
- Git installed (>=2.20)
- Python 3.8 or newer
- Virtual environment tool (venv or virtualenv)
- Internet connectivity for pip package installation
- Optional: NVIDIA GPU with compatible CUDA drivers (for GPU acceleration)
- Access to the Dog vs Cat image dataset (Kaggle Dogs vs Cats or equivalent)

## Environment Variables
| Variable | Status | Description |
| :--- | :--- | :--- |
| `DOG_CAT_DATA_DIR` | Required | Absolute path to the root folder that contains the `train/` and `validation/` image sub‑directories for the Dog vs Cat dataset. |


## Local Setup & Development
1. 1. Clone the repository:
2.    ```
3.    git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
4.    cd dog-vs-cat-cnn
5.    ```
6. 
7. 2. Create and activate a virtual environment:
8.    ```
9.    python -m venv .venv            # create virtualenv in .venv directory
10.    source .venv/bin/activate       # Linux/macOS
11.    .\\venv\\Scripts\\activate   # Windows PowerShell
12.    ```
13. 
14. 3. Install the required Python packages:
15.    ```
16.    pip install --upgrade pip
17.    pip install -r requirements.txt
18.    ```
19. 
20. 4. Download the Dog vs Cat dataset and place it in a known location, e.g.:
21.    - `~/datasets/dog_vs_cat/` (contains `train/` and `validation/` sub‑folders).
22. 
23. 5. Configure the environment variable that points to the dataset root:
24.    ```
25.    export DOG_CAT_DATA_DIR=~/datasets/dog_vs_cat   # Linux/macOS
26.    set DOG_CAT_DATA_DIR=%%USERPROFILE%%\datasets\dog_vs_cat   # Windows CMD
27.    $env:DOG_CAT_DATA_DIR = "C:\datasets\dog_vs_cat"   # PowerShell
28.    ```
29. 
30. 6. Verify the installation by running a quick training pass (1 epoch, small batch):
31.    ```
32.    python src/convolutional_neural_network.py \
33.           --epochs 1 \
34.           --batch-size 32 \
35.           --data-dir $DOG_CAT_DATA_DIR
36.    ```
37. 
38.    The script should download any missing pretrained weights, start training, and print the loss/accuracy for the epoch.

## Running Tests
```bash
There are no dedicated unit‑test suites in this prototype. To sanity‑check the code you can run a minimal training job:
```bash
python src/convolutional_neural_network.py --epochs 1 --batch-size 16 --data-dir $DOG_CAT_DATA_DIR
```
```

## Troubleshooting
### ImportError / ModuleNotFoundError for tensorflow / keras / numpy etc.
**Resolution:** Make sure the virtual environment is activated and that `pip install -r requirements.txt` completed without errors. Re‑run the install step or upgrade pip and retry.

### RuntimeError: GPU device not found or CUDA related errors.
**Resolution:** If you intend to run on GPU, verify that the CUDA toolkit and cuDNN versions match the TensorFlow build installed (see TensorFlow GPU compatibility matrix). Otherwise, force CPU execution by adding `--device cpu` flag (if supported) or uninstalling the GPU‑specific TensorFlow package.

### FileNotFoundError: Dataset directory does not exist.
**Resolution:** Confirm that the `DOG_CAT_DATA_DIR` environment variable points to the correct absolute path and that the directory contains `train/` and `validation/` sub‑folders with image files. Adjust the variable and restart the script.

### MemoryError / OOM during training.
**Resolution:** Reduce the batch size (`--batch-size`) or switch to CPU execution. For GPU runs, ensure the GPU has enough free memory or use gradient accumulation tricks (not implemented in this prototype).

### The notebook `src/notebooks/convolutional_neural_network.ipynb` cannot locate the data.
**Resolution:** Within the notebook, set the `data_dir` variable to the same path used for `DOG_CAT_DATA_DIR`, e.g.: `data_dir = os.getenv('DOG_CAT_DATA_DIR')` or manually assign the path.


