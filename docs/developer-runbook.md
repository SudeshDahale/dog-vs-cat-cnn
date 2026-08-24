# Dog vs Cat CNN - Developer Runbook

## Prerequisites
- Git
- Python 3.8+
- Virtual environment tool (venv, virtualenv, or conda)
- Jupyter Notebook (optional for exploratory notebook)
- Access to a GPU is optional but recommended for faster training

## Local Setup & Development
1. 1. Clone the repository
2.    ```
3.    git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
4.    cd dog-vs-cat-cnn
5.    ```
6. 2. Create and activate a virtual environment
7.    ```
8.    python -m venv .venv
9.    # On macOS/Linux
10.    source .venv/bin/activate
11.    # On Windows
12.    .venv\Scripts\activate
13.    ```
14. 3. Install the required Python packages
15.    ```
16.    pip install -r requirements.txt
17.    ```
18. 4. (Optional) Install Jupyter kernel for the virtual environment
19.    ```
20.    pip install ipykernel
21.    python -m ipykernel install --user --name=dogcat-cnn
22.    ```
23. 5. Verify the package installation
24.    ```
25.    python -c "import tensorflow as tf; print(tf.__version__)"
26.    ```
27. 6. Prepare the image dataset (dog vs. cat). The training scripts expect a directory structure like:
28.    ```
29.    data/
30.    ├── train/
31.    │   ├── dogs/
32.    │   └── cats/
33.    └── validation/
34.        ├── dogs/
35.        └── cats/
36.    ```
37.    Place your dataset accordingly or modify the path arguments in `src/convolutional_neural_network.py`.
38. 7. Run the training engine (see *Local Development Loop* below).

## Running Tests
```bash
There are no dedicated unit‑test suites in this repository. To validate the pipeline you can execute a quick end‑to‑end run:
```bash
python src/convolutional_neural_network.py \
    --train_dir data/train \
    --val_dir data/validation \
    --epochs 1 \
    --batch_size 32
```
If the script completes without errors and prints training/validation accuracy, the environment is correctly set up.
```

## Troubleshooting
### ImportError: No module named 'tensorflow'
**Resolution:** Ensure the virtual environment is activated and `requirements.txt` was installed successfully. Re‑run `pip install -r requirements.txt`.

### FileNotFoundError: Dataset directory not found
**Resolution:** Verify that the dataset follows the expected folder layout and that the paths passed to the script match the actual locations.

### MemoryError or OOM on CPU
**Resolution:** Reduce `batch_size` or `image_size` arguments, or run the script on a machine with a GPU. Install GPU‑enabled TensorFlow if a compatible GPU is available.

### Jupyter notebook cannot import local modules
**Resolution:** Launch the notebook from the repository root (`jupyter notebook`) so that `src/` is on the Python path, or add `import sys, os; sys.path.append(os.path.abspath('../src'))` at the top of the notebook.


