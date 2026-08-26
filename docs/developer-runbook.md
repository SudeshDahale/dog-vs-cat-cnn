# Developer Runbook – dog-vs-cat-cnn

## Prerequisites
- Python 3.8+ installed on the system
- Git client
- Virtual environment tool (venv or virtualenv)
- Optionally, NVIDIA GPU with CUDA toolkit if GPU acceleration is desired

## Environment Variables
| Variable | Status | Description |
| :--- | :--- | :--- |
| `DATA_ROOT` | Required | Absolute path to the directory containing the raw image dataset. The data‑preprocessing module reads images from this location. |
| `CUDA_VISIBLE_DEVICES` | Optional | Comma‑separated list of GPU device IDs to make visible to PyTorch (e.g., `0,1`). Required only when training on GPU. |


## Local Setup & Development
1. 1. Clone the repository:
2.    ```
3.    git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
4.    cd dog-vs-cat-cnn
5.    ```
6. 2. Create and activate a Python virtual environment:
7.    ```
8.    python -m venv .venv
9.    # On Windows
10.    .venv\Scripts\activate
11.    # On macOS/Linux
12.    source .venv/bin/activate
13.    ```
14. 3. Install the required Python packages:
15.    ```
16.    pip install --upgrade pip
17.    pip install -r requirements.txt
18.    ```
19. 4. (Optional) Verify GPU availability:
20.    ```
21.    python -c "import torch; print(torch.cuda.is_available())"
22.    ```
23. 5. Prepare the dataset:
24.    - Place the `train` and `validation` image folders (or a single dataset folder) under a directory of your choice.
25.    - Note the absolute path; it will be used as the `DATA_ROOT` environment variable (see below).

## Running Tests
```bash
There are no dedicated unit‑test files in this repository. Validation of the pipeline can be performed by executing a short training run on a subset of the data:
```bash
python src/convolutional_neural_network.py train \
    --data_dir ./processed_data \
    --epochs 2 \
    --batch_size 8 \
    --learning_rate 0.001 \
    --checkpoint_dir ./tmp_checkpoints
```
If the script finishes without errors and a checkpoint file appears in `./tmp_checkpoints`, the core pipeline is functional.
```

## Troubleshooting
### ImportError: No module named 'torch'
**Resolution:** Ensure that PyTorch is installed. Run `pip install torch torchvision` (or follow the PyTorch website for a GPU‑compatible wheel).

### CUDA runtime error – out of memory
**Resolution:** Reduce `--batch_size` or switch to CPU by unsetting `CUDA_VISIBLE_DEVICES` and adding `--device cpu` if the script supports it.

### FileNotFoundError for image paths
**Resolution:** Verify that the `DATA_ROOT` environment variable points to the correct dataset directory and that the expected sub‑folders (`train`, `validation`) exist.

### Jupyter notebook cannot locate the source modules
**Resolution:** Start the notebook from the repository root (as shown above) or add the repo root to `sys.path` inside the notebook:
```python
import sys, os
sys.path.append(os.path.abspath('..'))
```

### Training script exits immediately with no output
**Resolution:** Check that the required command‑line arguments (`--data_dir`, etc.) are provided. Run `python src/convolutional_neural_network.py --help` to view the expected interface.


