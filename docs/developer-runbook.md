# Dog vs Cat CNN Developer Runbook

## Prerequisites
- Git installed (>=2.20)
- Python 3.9 or newer
- Virtual environment tool (venv or conda)
- Git Large File Storage (optional, if dataset stored via LFS)
- CUDA-enabled GPU and NVIDIA drivers (optional, for GPU acceleration)
- Internet access to install Python dependencies

## Environment Variables
| Variable | Status | Description |
| :--- | :--- | :--- |
| `DATASET_ROOT` | Required | Absolute path to the root folder containing the image data sub‑folders (`train/`, `test/`). The code uses this variable to locate images during ingestion and preprocessing. |


## Local Setup & Development
1. 1. Clone the repository:
   ```bash
   git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
   cd dog-vs-cat-cnn
   ```
2. 2. Create and activate a Python virtual environment:
3.    - Using **venv**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate   # Linux/macOS
     .venv\Scripts\activate    # Windows PowerShell
     ```
4.    - Using **conda** (if preferred):
     ```bash
     conda create -n dogcat python=3.9
     conda activate dogcat
     ```
5. 3. Install the required packages:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. 4. (Optional) Install Jupyter Lab/Notebook for the exploratory notebook:
   ```bash
   pip install jupyterlab
   ```
7. 5. Prepare the image dataset:
   - Place the raw image folders (e.g., `train/dog`, `train/cat`, `test/dog`, `test/cat`) under a directory of your choice.
   - Set the absolute path to this directory in the environment variable `DATASET_ROOT` (see *Environment Variables* below).
   - If you do not have a dataset yet, you can download the standard Kaggle "Dogs vs. Cats" dataset and extract it.
   - Ensure the folder structure matches the expectations of the ingestion code in `src/convolutional_neural_network.py`.
   
8. 6. Verify the installation by running a quick sanity‑check script (see *Local Development Loop*).
   ```bash
   python src/convolutional_neural_network.py --dry-run
   ```

## Running Tests
```bash
Run the training script on a reduced dataset to verify the pipeline works end‑to‑end:
```bash
# Use a small subset for quick feedback (e.g., 200 images per class)
python src/convolutional_neural_network.py \
    --epochs 2 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --max-samples-per-class 200
```
The script prints training/validation loss and accuracy after each epoch. A successful run ends with a summary like:
```
Training completed. Final validation accuracy: 0.xx
```
```

## Troubleshooting
### ImportError: No module named 'torch' (or 'tensorflow/keras')
**Resolution:** Ensure the Python environment is activated and that `requirements.txt` has been installed. Re‑run `pip install -r requirements.txt`. If you need GPU support, install the appropriate CUDA‑compatible version of PyTorch (`pip install torch==<version>+cuXX -f https://download.pytorch.org/whl/torch_stable.html`).

### FileNotFoundError: Could not find dataset directory
**Resolution:** Verify that `DATASET_ROOT` points to the correct absolute path and that the directory contains the expected sub‑folders (`train/`, `test/`). You can echo the variable to confirm:
```bash
echo $DATASET_ROOT
```

### CUDA out of memory / torch.cuda.CudaError
**Resolution:** Reduce the `--batch-size` argument, or run the script on CPU by adding the flag `--device cpu`. Ensure that other GPU processes are not hogging memory.

### ValueError: Unexpected number of channels in image (e.g., 4 instead of 3)
**Resolution:** The preprocessing step expects RGB images. Remove or convert images with an alpha channel (e.g., PNG with transparency) to standard 3‑channel JPEGs before training.

### Training loop hangs or runs extremely slowly
**Resolution:** Check that the data loader is not bottlenecked by disk I/O. If using a large dataset on a mechanical HDD, consider copying a smaller subset to an SSD for development. Also, confirm that `--num-workers` (if exposed) matches the number of CPU cores you wish to use.


