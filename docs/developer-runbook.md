# Dog vs Cat CNN Development Runbook

## Prerequisites
- Python 3.8 or higher
- git
- virtualenv or conda
- CUDA-enabled GPU (optional for faster training)

## Environment Variables
| Variable | Status | Description |
| :--- | :--- | :--- |
| `DATA_DIR` | Required | Absolute path to the directory containing the training and validation image folders (e.g., /path/to/dataset) |


## Local Setup & Development
1. git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
2. cd dog-vs-cat-cnn
3. python -m venv venv   # or conda create -n dogcat python=3.8
4. source venv/bin/activate   # on Windows use venv\Scripts\activate
5. pip install -r requirements.txt
6. export DATA_DIR=/full/path/to/your/dataset   # adjust to your environment
7. # Verify dataset structure: $DATA_DIR/train/dog, $DATA_DIR/train/cat, $DATA_DIR/val/dog, $DATA_DIR/val/cat
8. # Run a quick sanity check by training a small model
9. python src/convolutional_neural_network.py --epochs 1 --batch-size 8
10. # Launch Jupyter for notebook exploration
11. jupyter notebook src/notebooks/convolutional_neural_network.ipynb

## Running Tests
```bash
python -m pytest || echo "No test suite found; run the training script to validate functionality"
```

## Troubleshooting
### ImportError: No module named 'tensorflow' or 'torch'
**Resolution:** Ensure the virtual environment is activated and all dependencies are installed via pip install -r requirements.txt. If using GPU, install the matching CUDA version of the deep‑learning framework.

### CUDA driver not found or GPU not detected
**Resolution:** Install the appropriate NVIDIA driver and CUDA toolkit, or run the script with the flag --device cpu to force CPU execution.

### Dataset not found or path errors
**Resolution:** Set the DATA_DIR environment variable to point to the root folder containing train/ and val/ subfolders. Verify folder names and image file permissions.

### Jupyter notebook kernel dies when loading large tensors
**Resolution:** Increase the memory limit or run the notebook on a machine with more RAM. Alternatively, reduce batch size or image resolution in the notebook parameters.


