# Developer Runbook for dog-vs-cat-cnn

## Prerequisites
- Git
- Python 3.8 or newer
- Virtualenv (or Conda) for isolated Python environments
- Access to the internet to download required packages and the dataset

## Environment Variables
| Variable | Status | Description |
| :--- | :--- | :--- |
| `DATA_ROOT` | Required | Absolute or relative path to the folder that contains the `train` (and optionally `test`) sub‑folders with cat and dog images. |
| `PYTHONPATH` | Optional | Ensures the `src` package is discoverable when launching notebooks or scripts from the repository root. |


## Local Setup & Development
1. 1. **Clone the repository**
   ```bash
   git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
   cd dog-vs-cat-cnn
   ```
2. 2. **Create and activate a virtual environment** (using `venv` as an example)
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```
3. 3. **Install the Python dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. 4. **Download the Cat vs. Dog dataset**
   - The repository expects the raw images to be placed under a directory referenced by the `DATA_ROOT` environment variable (default: `./data`).
   - You can obtain the dataset from Kaggle ("Dogs vs. Cats") or any compatible source. After download, extract the archives so that the structure resembles:
     ```
     data/
       train/
         cats/   # cat images
         dogs/   # dog images
       test/    # optional test images for inference
     ```
5. 5. **(Optional) Set environment variables**
   - If you place the data in a location other than `./data`, export `DATA_ROOT` accordingly (see *Environment Variables* section).
   - For Jupyter notebooks, you may also set `PYTHONPATH` to include the `src` package:
     ```bash
     export PYTHONPATH=$(pwd)/src:$PYTHONPATH
     ```
6. 6. **Run a quick sanity check**
   ```bash
   python src/convolutional_neural_network.py --mode test
   ```
   This will load a small subset of the data, build the CNN model, and run a single forward pass to verify that the pipeline works.

## Running Tests
```bash
python -m unittest discover -s tests || echo "No unit tests found – run a quick training sanity check instead."
```

## Troubleshooting
### ImportError: No module named 'cv2' or 'PIL'
**Resolution:** Install the missing imaging libraries. For OpenCV: `pip install opencv-python`. For Pillow: `pip install Pillow`. On Linux you may also need system packages (`libjpeg-dev`, `zlib1g-dev`).

### FileNotFoundError: Dataset directory not found
**Resolution:** Ensure the `DATA_ROOT` environment variable points to the correct location and that the directory contains `train/cats` and `train/dogs` sub‑folders. Verify with `ls $DATA_ROOT/train`.

### CUDA/cuDNN related errors when trying to use GPU
**Resolution:** The code defaults to CPU if a compatible GPU is not detected. Either install a matching version of `torch`/`tensorflow` with CUDA support, or run the script with the `--device cpu` flag (if the script provides such an argument).

### MemoryError when loading the full dataset
**Resolution:** The data loader uses lazy loading, but if you force loading all images at once, memory can blow up. Reduce the batch size (`--batch-size`) or use a smaller subset of the data for development.


