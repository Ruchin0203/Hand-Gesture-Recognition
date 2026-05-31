# 🖐️ Hand Gesture Recognition — Deep Neural Network

A multi-class image classification project that recognizes hand gestures from grayscale images using a fully connected deep neural network built with TensorFlow/Keras.

---

## 📌 Project Overview

This project uses the [Leap Motion Hand Gesture dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) to train a neural network capable of classifying **10 subjects × 5 gesture types** into 10 output classes. Images are grayscale infrared captures from a Leap Motion controller, and the model is a dense feedforward network trained end-to-end on flattened pixel features.

**Task:** Multi-class hand gesture classification (10 classes)  
**Approach:** Pixel normalisation → Flatten → Dense layers with LeakyReLU → Softmax  
**Dataset:** leapGestRecog (Kaggle — Leap Motion Hand Gesture Recognition)

---

## 🗂️ Project Structure

```
ML_04/
├── ML_04.ipynb                  # Main Jupyter notebook
├── archive.zip                  # Dataset archive
└── leapGestRecog/               # Extracted dataset
    ├── 00/                      # Subject 0
    │   ├── 01_palm/
    │   ├── 02_l/
    │   ├── 03_fist/
    │   ├── 04_fist_moved/
    │   └── 05_thumb/
    ├── 01/                      # Subject 1
    │   └── ...
    └── ...                      # Subjects 02–09
```

### Gesture Classes

| Folder | Gesture |
|--------|---------|
| `01_palm` | Open palm |
| `02_l` | L-shape |
| `03_fist` | Closed fist |
| `04_fist_moved` | Moved fist |
| `05_thumb` | Thumbs up |

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install numpy pandas matplotlib opencv-python tqdm scikit-learn tensorflow seaborn scikit-image
```

Or with a virtual environment:

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install numpy pandas matplotlib opencv-python tqdm scikit-learn tensorflow seaborn scikit-image
```

### Dataset

1. Download the dataset from [Kaggle — leapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog).
2. Place `archive.zip` in the project root (`ML_04/`).
3. The notebook extracts it automatically on first run.

> **Note:** Update the hardcoded Windows path in the notebook's zip extraction and folder traversal cells to match your local directory before running.

---

## 🚀 Usage

Launch Jupyter and open the notebook:

```bash
jupyter notebook ML_04.ipynb
```

Run all cells in order. The notebook will:

1. Extract the dataset archive
2. Load and visualise sample images for each gesture class
3. Build the labelled training dataset
4. Train the neural network
5. Plot training loss and accuracy curves
6. Evaluate on the test set and display a confusion matrix

---

## 🔬 Methodology

### 1. Data Loading & Exploration
Images are loaded in **grayscale** using OpenCV from 10 subject folders, each containing 5 gesture subfolders. A consistency check confirms all images share the same resolution (**240 × 640 pixels**). Sample grids (3 images per gesture per class) are visualised before training.

### 2. Dataset Construction
All images and their class labels (subject number 0–9) are collected into a list and shuffled randomly to prevent ordering bias.

### 3. Preprocessing
- **Normalisation:** Pixel values scaled from `[0, 255]` → `[0.0, 1.0]`
- **Train/Test Split:** 80% training / 20% testing (`random_state=42`)

### 4. Model Architecture

```
Input: (240 × 640) grayscale image
    ↓
Flatten → 153,600 features
    ↓
Dense(64) → LeakyReLU(α=0.1)
    ↓
Dense(32) → LeakyReLU(α=0.1)
    ↓
Dense(16) → LeakyReLU(α=0.1)
    ↓
Dense(10, activation='softmax')   ← 10 output classes
```

| Layer | Output Shape | Notes |
|-------|-------------|-------|
| Flatten | 153,600 | Raw pixel vector |
| Dense + LeakyReLU | 64 | Hidden layer 1 |
| Dense + LeakyReLU | 32 | Hidden layer 2 |
| Dense + LeakyReLU | 16 | Hidden layer 3 |
| Dense (Softmax) | 10 | Class probabilities |

**Optimizer:** Adam  
**Loss:** Sparse Categorical Cross-Entropy  
**Epochs:** 3 | **Batch Size:** 32 | **Validation Split:** 10%

### 5. Evaluation
- Training and validation **loss / accuracy curves**
- Predicted vs. actual comparison DataFrame
- **Confusion matrix** heatmap (seaborn)
- Full **classification report** (precision, recall, F1 per class)

---

## 📊 Results

The notebook outputs a classification report and confusion matrix after evaluation. Training curves are plotted for both loss and accuracy across epochs.

> **Note:** With only 3 epochs and a fully connected architecture on 240×640 raw pixels, results serve as a strong baseline. See the improvements section below for paths to higher accuracy.

---

## 🧠 Key Libraries

| Library | Role |
|---------|------|
| `tensorflow` / `keras` | Model building and training |
| `opencv-python` | Grayscale image loading |
| `numpy` / `pandas` | Array operations and result comparison |
| `scikit-learn` | Train/test split, classification report |
| `matplotlib` / `seaborn` | Training curves and confusion matrix |
| `tqdm` | Progress bars during data loading |
| `scikit-image` | Image utility support |

---

## 💡 Potential Improvements

- **CNN architecture** — Convolutional layers would capture spatial patterns far more effectively than raw pixel flattening
- **More epochs** — Only 3 epochs were used; training longer (with early stopping) would improve convergence
- **Data augmentation** — Random flips, rotations, and brightness shifts to improve generalisation
- **Batch normalisation** — Stabilise and accelerate training between dense layers
- **Deeper hidden layers** — The commented-out larger layers (1024→512→256→128) in the notebook can be re-enabled for experimentation
- **Transfer learning** — Adapt a pretrained model (e.g., MobileNet) to gesture features

---

## 📄 License

This project is for educational purposes. The leapGestRecog dataset is provided by GTI-UPM via Kaggle under their respective terms.
