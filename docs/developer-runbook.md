# Developer Runbook – dog-vs-cat-cnn

## Prerequisites
- Git
- Python 3.8+
- pip
- Virtual environment tool (venv or conda)
- Access to a GPU (optional but recommended for training)
- Dataset of dog and cat images (e.g., Kaggle Dogs vs Cats) placed in a directory referenced by the code

## Environment Variables
| Variable | Status | Description |
| :--- | :--- | :--- |
| `DATA_DIR` | Required | Absolute or relative path to the root folder that contains the `train` and `test` sub‑folders with dog and cat images. |


## Local Setup & Development
1. 1. Clone the repository:
   ```bash
   git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
   cd dog-vs-cat-cnn
   ```
2. 2. Create and activate a virtual environment:
3.    - Using `venv`:
     ```bash
     python -m venv .venv
     source .venv/bin/activate   # Linux/macOS
     .\\venv\\Scripts\\activate   # Windows
     ```
4.    - Or using `conda`:
     ```bash
     conda create -n dogcat python=3.8
     conda activate dogcat
     ```
5. 3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
6. 4. Verify that the required image data is available:
7.    - The code expects a directory structure similar to:
     ```
     data/
       train/
         dogs/
         cats/
       test/
         dogs/
         cats/
     ```
   - Adjust the `DATA_DIR` path in `src/convolutional_neural_network.py` if you place the data elsewhere.
8. 5. (Optional) If you have a CUDA‑enabled GPU and want to use it, ensure that the appropriate `torch` version is installed. The `requirements.txt` pins `torch` without CUDA; replace it with e.g. `torch==2.2.0+cu118` from the official PyTorch channel and reinstall.
9. 6. Launch the exploratory notebook to confirm the environment works:
   ```bash
   jupyter notebook src/notebooks/convolutional_neural_network.ipynb
   ```

## Running Tests
```bash
No automated unit tests are shipped with this repository. Validation is performed by running a short training session or by executing the notebook cells. Example manual test:
```bash
python - <<'PY'
import torch
from src.convolutional_neural_network import CatDogCNN
model = CatDogCNN()
# Create a dummy batch: 4 RGB images of size 128x128
x = torch.randn(4, 3, 128, 128)
logits = model(x)
print('Logits shape:', logits.shape)
PY
```
```

## Troubleshooting
### ImportError: No module named 'src'
**Resolution:** Make sure you are executing commands from the repository root or add the project root to PYTHONPATH:
```bash
export PYTHONPATH=$(pwd)
```

### torch.cuda.is_available() returns False even though a GPU is present
**Resolution:** Install a CUDA‑compatible build of PyTorch. Replace the `torch` entry in `requirements.txt` with the version matching your CUDA toolkit, e.g. `torch==2.2.0+cu118`, then run `pip install -r requirements.txt` again.

### FileNotFoundError when loading images
**Resolution:** Confirm that the `DATA_DIR` environment variable points to the correct location and that the expected `train`/`test` sub‑folders exist. Adjust the path in `convolutional_neural_network.py` if needed.

### Jupyter kernel crashes while running the notebook
**Resolution:** Check memory usage – training a CNN on the full Dogs vs Cats dataset can exceed typical laptop RAM. Reduce the batch size in the notebook (e.g., `batch_size = 16`) or work with a smaller subset of the data.


