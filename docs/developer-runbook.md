# Dog vs Cat CNN – Developer Runbook

## Prerequisites
- Python 3.8+ installed on the development machine
- Git for version control
- Virtual environment tool (venv or conda) to isolate dependencies
- Access to the dataset (e.g., Kaggle Dogs vs Cats) – placed in a local `data/` directory as described in the README
- Optional: Jupyter Notebook support for interactive experimentation

## Environment Variables
| Variable | Status | Description |
| :--- | :--- | :--- |
| `DATA_DIR` | Optional | Absolute or relative path to the folder containing the `train` and `test` sub‑folders with dog and cat images. |
| `MODEL_SAVE_PATH` | Optional | Path where the trained CNN model (`.h5` file) will be persisted. Defaults to `models/` if not set. |


## Local Setup & Development
1. 1. Clone the repository:
2.    ```
3.    git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
4.    cd dog-vs-cat-cnn
5.    ```
6. 2. Create and activate a virtual environment:
7.    - Using `venv`:
8.      ```
9.      python -m venv .venv
10.      source .venv/bin/activate   # On Windows: .venv\Scripts\activate
11.      ```
12.    - Or using `conda`:
13.      ```
14.      conda create -n dogcat python=3.8
15.      conda activate dogcat
16.      ```
17. 3. Install required Python packages:
18.    ```
19.    pip install -r requirements.txt
20.    ```
21.    The `requirements.txt` lists core libraries such as `tensorflow`/`keras`, `numpy`, `pandas`, `matplotlib`, and `jupyter`.
22. 4. Verify the installation:
23.    ```
24.    python -c "import tensorflow as tf; print('TF version:', tf.__version__)"
25.    ```
26. 5. (Optional) Launch Jupyter Lab/Notebook for the exploratory notebook:
27.    ```
28.    jupyter notebook src/notebooks/convolutional_neural_network.ipynb
29.    ```

## Running Tests
```bash
The project does not contain a separate test suite, but you can run a quick sanity check by training the model on a small subset:
```bash
python src/convolutional_neural_network.py --epochs 1 --batch-size 32 --subset 0.1
```
This script will:
- Load data from `DATA_DIR` (or the default `data/` folder).
- Build the CNN defined in `src/convolutional_neural_network.py`.
- Train for 1 epoch on 10 % of the data.
- Save the model to `MODEL_SAVE_PATH` (or `models/`).
If the script finishes without errors, the development environment is correctly configured.

For notebook‑based verification, run the cells in `src/notebooks/convolutional_neural_network.ipynb` and ensure the training loss decreases.
```

## Troubleshooting
### ImportError: No module named 'tensorflow'
**Resolution:** Make sure the virtual environment is activated and `tensorflow` is installed. Re‑run `pip install -r requirements.txt`. If you are on a GPU machine, install the GPU‑compatible package (`tensorflow-gpu`).

### MemoryError or OOM during model training
**Resolution:** Reduce `batch-size` (e.g., `--batch-size 16`) or use a smaller image size in the data loader. Alternatively, run the training on a machine with more RAM or a GPU.

### FileNotFoundError: data directory not found
**Resolution:** Set the `DATA_DIR` environment variable to point to the location of the Dogs vs Cats dataset, or place the dataset under the repository's `data/` folder as described in the README.

### Jupyter notebook kernel dies when executing the training cell
**Resolution:** Ensure the notebook is using the same Python kernel as the virtual environment. In Jupyter, select the kernel from `.venv/bin/python` (or the conda environment). Restart the kernel after any package changes.


