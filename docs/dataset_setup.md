# Dataset Setup Guide for FADE-Net

This guide explains how to prepare the datasets (**AFAD**, **AAF**, **UTKFace**) required for training and evaluating FADE-Net.

## 1. Directory Structure

The project expects the following structure after preprocessing:

```
code/
├── datasets/           # Processed & Aligned images (Generated)
│   ├── AFAD/
│   ├── AAF/
│   └── UTKFace/
├── data/               # Raw downloaded datasets (Recommended)
│   ├── UTKFace/
│   ├── AFAD-Full/
│   └── All-Age-Faces/
```

## 2. Preparing Raw Data

Please ensure you have the raw datasets downloaded.

### AFAD (Asian Face Age Dataset)
*   **Source**: [GitHub / Official Site]
*   **Format**: Nested folders (Age -> Gender -> Images).
*   **Path in Script**: Currently hardcoded or expects a specific path. Please check `scripts/preprocess.py` to point to your `AFAD-Full` directory.

### AAF (All-Age-Faces Dataset)
*   **Source**: [GitHub]
*   **Format**: Flat folder with aligned faces.
*   **Path in Script**: Please check `scripts/preprocess.py`.

### UTKFace
*   **Source**: [Susanqq/UTKFace]
*   **Path**: Expects `./data/UTKFace/train` and `./data/UTKFace/val`.

## 3. Running Preprocessing

We provide a script to **clean, align, and organize** the datasets into the standard format required by `src/dataset.py`.

1.  **Edit Paths**: Open `scripts/preprocess.py` and modify the `raw_afad_dir` and `raw_aaf_dir` variables to point to where you downloaded the datasets.
    ```python
    # scripts/preprocess.py
    raw_afad_dir = "path/to/AFAD-Full"
    raw_aaf_dir = "path/to/AAF/aglined_faces"
    ```

2.  **Run the Script**:
    ```bash
    python scripts/preprocess.py
    ```

3.  **Result**:
    The script will create a `datasets/` directory in the project root containing the processed images.

## 4. Verification

After processing, you should verify the `datasets/` folder exists and contains valid images.
You can modify `src/config.py` if you need to change the location of the `datasets` folder, but it defaults to `ROOT/datasets`.
