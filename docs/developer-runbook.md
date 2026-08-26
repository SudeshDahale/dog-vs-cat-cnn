# Dog vs Cat CNN – Developer Runbook

## Prerequisites
- Git
- Python 3.8+ installed and accessible via `python` command
- Virtual environment tool (venv or conda)
- A CUDA‑compatible GPU (optional, for faster training) with appropriate drivers if you plan to use GPU acceleration

## Environment Variables
| Variable | Status | Description |
| :--- | :--- | :--- |
| `DATA_ROOT` | Optional | Absolute or relative path to the root folder that contains `train/` and `validation/` sub‑folders. If omitted, the code defaults to `./data`. |
| `MODEL_SAVE_PATH` | Optional | Path where the trained model (`.h5`/`.pt`) will be saved. Defaults to `./model/dog_vs_cat_cnn.h5`. |
| `CUDA_VISIBLE_DEVICES` | Optional | Comma‑separated list of GPU IDs to expose to TensorFlow/PyTorch. Set to an empty string to force CPU execution. |


## Local Setup & Development
1. 1. **Clone the repository**
   ```bash
2.    git clone https://github.com/SudeshDahale/dog-vs-cat-cnn.git
3.    cd dog-vs-cat-cnn
4.    ```
5. 
6. 2. **Create and activate a virtual environment** (recommended). Example with `venv`:
7.    ```bash
8.    python -m venv .venv
9.    # Windows
10.    .venv\Scripts\activate
11.    # macOS/Linux
12.    source .venv/bin/activate
13.    ```
14. 
15. 3. **Install Python dependencies**
16.    ```bash
17.    pip install --upgrade pip
18.    pip install -r requirements.txt
19.    ```
20. 
21. 4. **Prepare the dataset**
   The code expects a directory structure similar to:
22.    ```
23.    data/
24.    ├─ train/
25.    │   ├─ dogs/
26.    │   └─ cats/
27.    └─ validation/
28.        ├─ dogs/
29.        └─ cats/
30.    ```
31.    If you have the original Kaggle *Dogs vs. Cats* dataset, extract it and organize it as above. By default the scripts look for the data folder at the project root; you can override the location with the `DATA_ROOT` environment variable (see below).
32. 
33. 5. **(Optional) Set environment variables** – see the *Environment Variables* section.
34. 
35. 6. **Run the notebook to explore the pipeline** (optional but recommended for the first run).
36.    ```bash
37.    jupyter notebook src/notebooks/convolutional_neural_network.ipynb
38.    ```
39. 
40. 7. **Start the training script** – the entry point is `src/convolutional_neural_network.py`.
41.    ```bash
42.    python src/convolutional_neural_network.py --mode train
43.    ```
44. 
45. 8. **Evaluate the trained model**
46.    ```bash
47.    python src/convolutional_neural_network.py --mode evaluate
48.    ```
49. 
50. 9. **Run inference on a new image**
51.    ```bash
52.    python src/convolutional_neural_network.py --mode predict --image_path path/to/image.jpg
53.    ```

## Running Tests
```bash
pytest tests/  # (If a `tests/` folder exists – add unit tests here)
# Quick sanity check – run a single‑epoch training on a tiny subset:
python src/convolutional_neural_network.py --mode train --epochs 1 --sample_fraction 0.01
```

## Troubleshooting
### ImportError: No module named 'tensorflow' (or 'torch')
**Resolution:** Make sure the virtual environment is activated and that `pip install -r requirements.txt` completed without errors. If you need GPU support, install the matching `tensorflow-gpu` or `torch` version for your CUDA toolkit.

### FileNotFoundError: Data directory not found
**Resolution:** Verify that the `data/` folder follows the expected layout. Either place the folder at the repository root or set `DATA_ROOT` to point to its location.

### CUDA out‑of‑memory error during training
**Resolution:** Reduce the batch size (e.g., `--batch_size 16`) or set `CUDA_VISIBLE_DEVICES=''` to fall back to CPU. Alternatively, use gradient accumulation to simulate a larger batch.

### Jupyter notebook cannot find the project modules
**Resolution:** Start the notebook from the repository root (as shown above) or add the project `src/` directory to `sys.path` inside the notebook:
```python
import sys, os
sys.path.append(os.path.abspath('../src'))
```


