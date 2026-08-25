# Developer Runbook for dog-vs-cat-cnn

## Prerequisites
- Python 3.9 or later installed
- Git installed
- Virtual environment tool (venv or conda)
- Optional: NVIDIA GPU with CUDA drivers for GPU acceleration

## Local Setup & Development
1. 1. Clone the repository
2.    ```bash
3.    git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
4.    cd dog-vs-cat-cnn
5.    ```
6. 2. Create and activate a virtual environment
7.    - Using **venv**:
8.      ```bash
9.      python -m venv .venv
10.      source .venv/bin/activate   # On Windows: .venv\Scripts\activate
11.      ```
12.    - Or using **conda**:
13.      ```bash
14.      conda create -n dog-cat-cnn python=3.9
15.      conda activate dog-cat-cnn
16.      ```
17. 3. Install the required Python packages
18.    ```bash
19.    pip install -r requirements.txt
20.    ```
21. 4. (Optional) Verify GPU availability
22.    ```bash
23.    python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
24.    ```
25. 5. Launch Jupyter Notebook for interactive experimentation
26.    ```bash
27.    jupyter notebook src/notebooks/convolutional_neural_network.ipynb
28.    ```
29. 6. Run the training script directly (if you prefer CLI)
30.    ```bash
31.    python src/convolutional_neural_network.py
32.    ```

## Running Tests
```bash
There are no dedicated unit tests in this repository. Validation is performed by running the training script or the notebook and confirming that the model trains without errors.
```

## Troubleshooting
### ImportError: No module named 'torch' (or any other dependency)
**Resolution:** Make sure the virtual environment is activated and all dependencies are installed with `pip install -r requirements.txt`. If you are using a new terminal, reactivate the environment.

### CUDA not detected despite having an NVIDIA GPU
**Resolution:** Install the matching CUDA toolkit and cuDNN versions for your PyTorch build. Refer to the PyTorch "Get Started" page for the correct pip wheel, e.g., `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118` for CUDA 11.8.

### Jupyter notebook fails to start or cannot find the kernel
**Resolution:** Ensure the notebook is launched from within the activated virtual environment. If using conda, run `conda install ipykernel && python -m ipykernel install --user --name dog-cat-cnn`.

### MemoryError during training on large batch sizes
**Resolution:** Reduce the `batch_size` parameter in the training script/notebook or switch to a GPU with more VRAM. Alternatively, enable gradient accumulation to simulate larger batches.

### Training loss does not decrease (model not learning)
**Resolution:** Check that the dataset path is correct and that the data loader is properly shuffling data. Verify learning rate and optimizer settings in `convolutional_neural_network.py`.


