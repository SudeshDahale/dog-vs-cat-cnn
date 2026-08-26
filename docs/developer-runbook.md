# Dog vs Cat CNN – Developer Runbook

## Prerequisites
- Git
- Python 3.8+ installed on the development machine
- Virtual environment tool (venv or conda)
- Access to the image dataset (e.g., a folder containing `train/` and `validation/` sub‑folders)

## Environment Variables
| Variable | Status | Description |
| :--- | :--- | :--- |
| `DATA_ROOT` | Optional | Absolute path to the root folder that contains the raw image dataset. The data‑prep module will look for `train/` and `validation/` sub‑folders inside this directory. |


## Local Setup & Development
1. 1. Clone the repository
2. ```bash
git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
cd dog-vs-cat-cnn
```
3. 2. Create and activate a virtual environment
4. ```bash
# Using venv
python -m venv .venv
source .venv/bin/activate   # On Windows use `.venv\Scripts\activate`

# Or using conda
conda create -n dogcatcnn python=3.8 -y
conda activate dogcatcnn
```
5. 3. Install the Python dependencies defined in `requirements.txt`
6. ```bash
pip install -r requirements.txt
```
7. 4. (Optional) Export any environment variables required for data locations – see the "Environment Variable Configuration" section.
8. 5. Verify the installation by running the sanity‑check script (if present) or importing the main module:
9. ```bash
python -c "import src.convolutional_neural_network as cnn; print('Import successful')"
```

## Running Tests
```bash
There is no dedicated unit‑test suite in this repository. The quickest way to verify that the workflow is functional is to run a short end‑to‑end training and evaluation cycle on a small subset of the data:
```bash
# Step 1 – Prepare the data (the script respects the DATA_ROOT env var if set)
python src/data_preparation.py   # If the file exists; otherwise skip – the training script will load data directly.

# Step 2 – Train the CNN for a few epochs (use a reduced batch size for speed)
python src/convolutional_neural_network.py --epochs 2 --batch-size 16

# Step 3 – Evaluate the trained model
python src/evaluation.py --model-path outputs/model.h5
```
If the notebook is preferred, you can launch it locally:
```bash
jupyter notebook src/notebooks/convolutional_neural_network.ipynb
```
```

## Troubleshooting
### ImportError / ModuleNotFoundError for packages listed in `requirements.txt`.
**Resolution:** Ensure the virtual environment is activated (`source .venv/bin/activate` or `conda activate dogcatcnn`) and that `pip install -r requirements.txt` completed without errors. Re‑run the install command if any packages failed.

### CUDA/GPU related error when running the training script.
**Resolution:** The repository falls back to CPU if a compatible GPU is not detected. If you intended to use GPU, verify that the correct version of `tensorflow`/`torch` (as specified in `requirements.txt`) matches your CUDA driver. Alternatively, force CPU execution by setting the environment variable `CUDA_VISIBLE_DEVICES=` (empty) before running the script.

### FileNotFoundError: Unable to locate training images.
**Resolution:** Either place your image dataset under a folder named `data/` at the repository root with `train/` and `validation/` sub‑folders, or export the absolute path to your dataset via the `DATA_ROOT` environment variable before invoking any script:
```bash
export DATA_ROOT=/absolute/path/to/your/dataset
```

### Jupyter notebook fails to import `src` modules (ImportError: No module named 'src').
**Resolution:** Launch the notebook from the repository root so that Python’s module search path includes the `src` package, or add the repository root to `PYTHONPATH`:
```bash
export PYTHONPATH=$(pwd)
```

### Training script exits immediately without training.
**Resolution:** Check that the script received the expected command‑line arguments. Run the script with the `--help` flag to view usage:
```bash
python src/convolutional_neural_network.py --help
```


