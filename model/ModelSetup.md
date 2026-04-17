# Model Setup Guide

Guide for preparing dataset and training Devanagari character recognition model.

## Automatic Setup

### 1. Download Dataset:

- Download from Kaggle: [Devanagari Handwritten Character Dataset](https://www.kaggle.com/datasets/medahmedkrichen/devanagari-handwritten-character-datase)
- Save the ZIP file into the `model/classes` folder.

### 2. Run Auto Setup:

```bash
python model/setup.py
```

- This script will extract, rename, preprocess, and clean up automatically.
- It will also prompt you if you want to start training immediately.

---

## Manual Setup

### 1. Get Dataset

- Download from Kaggle: [Devanagari Handwritten Character Dataset](https://www.kaggle.com/datasets/medahmedkrichen/devanagari-handwritten-character-datase)
- Save ZIP in `model/classes` folder.

### 2. Prepare Folders

- Extract ZIP in `model/classes`.
- Rename folders:
    - `train` → `training`
    - `test` → `testing`

### 3. Preprocess Data

- Run the preprocessing script:
    ```bash
    python model/01-prepare-classes.py
    ```
- This script cleans images (shirorekha removal) and automatically creates `train` and `test` folders inside `classes` from your `training` and `testing` sources.

### 4. Cleanup

- Delete raw folders `model/classes/training` and `model/classes/testing`.

### 5. Train Model

- **Local**:
    ```bash
    python model/02-train-model.py
    ```
- **Google Colab**:
    - Zip the `classes` folder.
    - Upload to Google Drive / Capstone2.
    - Run [Google Colab Notebook](https://colab.research.google.com/drive/1jA5Dw27gYUvwUcF0EjvPbY5d5Xz6Qbl6).
    - Download `resnet18.pth`.

### 6. Update Model

- Move the new `resnet18.pth` into the `model` folder.
- Ensure filename is `resnet18.pth` for pipeline compatibility.
