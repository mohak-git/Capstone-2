---
license: cc-by-nc-sa-4.0
task_categories:
    - image-classification
    - image-to-text
    - image-feature-extraction
language:
    - sa
tags:
    - sanskrit
    - ocr
    - text-recognition
    - text-extraction
    - image-classification
pretty_name: "Sanskrit OCR Dataset "
size_categories:
    - 1K<n<10K
---

# Sanskrit OCR Dataset

This dataset contains Sanskrit text images paired with their corresponding text labels, designed for OCR (Optical Character Recognition) tasks.

## Dataset Structure

The dataset is split into training and validation sets:

- Training set: Contains unique Sanskrit text images
- Validation set: Contains separate unique Sanskrit text images

## Features

- `image`: The image containing Sanskrit text
- `label`: The corresponding Sanskrit text label
- `filename`: Original image filename

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("your-username/sanskrit-ocr-dataset")

# Access the data
train_data = dataset["train"]
validation_data = dataset["validation"]
```

#### Licensing Information

MIT License

Copyright (c) 2024 ProcessVenue AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated data (the "Dataset"), to deal
in the Dataset without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Dataset, and to permit persons to whom the Dataset is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Dataset.

THE DATASET IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE DATASET OR THE USE OR OTHER DEALINGS IN THE
DATASET.

#### How to Cite

If you use this dataset in your research or project, please cite it as follows:

```bibtex
@misc{sanskrit-ocr-dataset,
  title     = {Sanskrit OCR Dataset},
  author    = {ProcessVenue},
  year      = {2024},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/datasets/processvenue/Sanskrit-OCR-Dataset},
  organization = {ProcessVenue},
  contact   = {info@predusk.com},
  Website  = {https://www.processvenue.com/}
}
```

For use in publications, please also include:

"This work uses the Sanskrit OCR Dataset created by Team ProcessVenue, available under MIT License at https://huggingface.co/datasets/processvenue/Sanskrit-OCR-Dataset"

#### Usage Terms

While this dataset is released under the MIT License, we kindly request that you:

1. Cite the dataset using the provided citation format when used in research or projects
2. Include a link to the original dataset repository
3. Mention Team ProcessVenue as the dataset creator
4. Notify us of any publications or projects using this dataset at info@predusk.com
