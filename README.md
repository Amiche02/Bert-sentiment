# Bert-sentiment

This repository contains a PyTorch implementation for tweet sentiment extraction using a BERT-based model. It has been updated to use recent versions of Hugging Face Transformers (v4.35.0+), and it includes a Dockerfile based on Ubuntu 22.04 with CUDA support and a Conda environment.

The project supports:
- Data preprocessing using a custom dataset class.
- Model definition based on `bert-base-uncased`.
- Training and evaluation using a simple extraction head (predicting start and end indices).
- Inference for extracting sentiment-bearing phrases from tweets.
- A helper script to generate a JSON representation of the project directory structure.

---

## Table of Contents

- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
  - [Using Docker](#using-docker)
  - [Local Setup](#local-setup)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Directory Structure Script](#directory-structure-script)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Modern Transformers:** Utilizes `transformers>=4.35.0` and Torch 2.0 for state-of-the-art model performance.
- **BERT-based Extraction:** Implements a model that extracts sentiment-specific phrases using a start/end prediction approach.
- **Docker Support:** Includes a Dockerfile built on Ubuntu 22.04 with CUDA runtime and Conda environment for easy reproducibility.
- **Easy Inference:** Provides an inference script (`inference.py`) that loads a trained model and extracts text based on sentiment.
- **Project Structure Helper:** A Python script (`structure.py`) to output a JSON representation of the repository structure.

---

## Directory Structure

```
amiche02-bert-sentiment/
├── README.md
├── Dockerfile
├── bert-base-uncased-using-pytorch.ipynb  # Notebook version of the code
├── config.py               # Configuration settings (paths, hyperparameters, etc.)
├── dataset.py              # Dataset and data preprocessing code
├── engine.py               # Training and evaluation loops
├── inference.py            # Inference script for prediction
├── model.py                # BERT model definition
├── requirements.txt        # Python dependencies
├── structure.py            # Script to dump directory structure as JSON
└── train.py                # Main training script
```

---

## Installation

### Using Docker

1. **Build the Docker image:**

   ```bash
   docker build -t bert-sentiment:latest .
   ```

2. **Run the Docker container:**

   ```bash
   docker run --gpus all -it --rm bert-sentiment:latest
   ```

   Inside the container, your Conda environment (`ml`) is activated. You can run:

   ```bash
   python train.py    # To train the model
   python inference.py  # To test inference on an example tweet
   ```

### Local Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Amiche02/Bert-sentiment.git
   cd Bert-sentiment
   ```

2. **Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you haven’t yet.**

3. **Create the Conda environment and install dependencies:**

   ```bash
   conda create -n ml python=3.10 -y
   conda activate ml
   pip install -r requirements.txt
   ```

4. **Run the training or inference scripts:**

   ```bash
   python train.py
   python inference.py
   ```

---

## Usage

### Training

- **Data:**  
  Place your training data (e.g., `train.csv` and `valid.csv`) in the repository root or adjust the paths in `config.py`.

- **Run Training:**  

  ```bash
  python train.py
  ```

  This script loads your data using `dataset.py`, trains the model using the functions in `engine.py`, and saves the best model to the path specified in `config.py`.

### Inference

- **Run Inference on a Single Example:**

  ```bash
  python inference.py
  ```

  This script will load the trained model and run a sample tweet through the model, printing the extracted text.

- **Integration with a Flask App:**  
  If you wish to serve the model via a REST API, use the provided `app.py` as a starting point.

### Directory Structure Script

- **Generate a JSON representation of the repository structure:**

  ```bash
  python structure.py
  ```

  Enter your project path when prompted. The script will create a file named `project_structure.json`.

---
