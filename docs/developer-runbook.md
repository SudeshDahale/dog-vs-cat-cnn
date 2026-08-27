# Developer Runbook – dog-vs-cat-cnn

## Prerequisites
- Git installed
- Python 3.8+
- Virtual environment tool (venv or conda)
- Git LFS (optional, if dataset stored via LFS)

## Local Setup & Development
1. 1. Clone the repository:
2.    ```
3.    git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
4.    cd dog-vs-cat-cnn
5.    ```
6. 2. Create and activate a virtual environment:
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
19. 4. (Optional) Install Jupyter Notebook to explore the example notebook:
20.    ```
21.    pip install notebook
22.    ```
23. 5. Verify the installation by running a quick import:
24.    ```
25.    python -c "import tensorflow as tf; print('TF version:', tf.__version__)"
26.    ```

## Running Tests
```bash
Run the training script on a small subset of data to ensure the pipeline works:
```bash
python src/convolutional_neural_network.py --epochs 1 --batch_size 8 --test_mode True
```
The `--test_mode` flag (if implemented) should load a minimal in‑memory dataset rather than the full image directory. The script will print training loss/accuracy and exit.

To launch the interactive notebook:
```bash
jupyter notebook src/notebooks/convolutional_neural_network.ipynb
```
```

## Troubleshooting
### ImportError: No module named 'tensorflow'
**Resolution:** Ensure the virtual environment is activated and that `tensorflow` appears in `pip list`. Re‑install with `pip install tensorflow`.

### CUDA / GPU related errors (e.g., "Failed to load the CUDA runtime library")
**Resolution:** If you do not have a GPU, install the CPU‑only TensorFlow package:
```bash
pip uninstall tensorflow
pip install tensorflow-cpu
```
If you have a GPU, verify that the CUDA and cuDNN versions match the TensorFlow version listed in `requirements.txt`.

### FileNotFoundError when loading image data
**Resolution:** The repository does not ship the raw image dataset. Place the dog/cat image folders under a directory named `data/` (e.g., `data/train/dog`, `data/train/cat`). Adjust the path in `src/convolutional_neural_network.py` if needed.

### MemoryError or OOM during training
**Resolution:** Reduce the batch size (e.g., `--batch_size 4`) or resize images to a smaller dimension in the preprocessing step.


