# Cervical Cancer Classification Using Deep Learning

This repository presents a machine learning project for the **binary classification** of cervical cells as **Normal** or **Abnormal**. The approach combines **Deep Learning for Feature Extraction** (using a pre-trained CNN) with a **Classical Machine Learning Classifier** (Logistic Regression).

## 🚀 Key Technologies

* **PyTorch** (`torch`, `torchvision`): Used for loading the pre-trained ResNet152 model and performing feature extraction.
* **Scikit-learn** (`scikit-learn`): Used for training the Logistic Regression classifier and applying `StandardScaler`.
* **Flask** (`flask`): Used to create a simple web application for image prediction.
* **ResNet152** (as a feature extractor).
* **Logistic Regression** (as the final classifier).

## 📂 Project Structure

| Path | Description |
| :--- | :--- |
| `src/data_preperation/binary_folders.py` | Maps 5 original cell types to 2 binary classes: **Normal** and **Abnormal**. |
| `src/data_preperation/split_dataset.py` | Splits the binary dataset into 80% **train** and 20% **test** subsets. |
| `src/feature_extraction/feature_extraction.py`| Extracts features using the pre-trained **ResNet152** CNN (final FC layer removed) and saves them as `.npz` files. |
| `src/classical_models/train_classifier.py` | Trains the **Logistic Regression** model, applies `StandardScaler` to the features, and saves both as `.joblib` files. |
| `src/predictor/image_predictor.py` | Encapsulates the prediction logic for a single image, loading the saved models. |
| `src/webapp/app.py` | Main Flask application file for the web interface. |
| `requirements.txt` | Project dependencies. |

## 🛠️ Setup and Installation

### Prerequisites

* Python 3.x
* A virtual environment is recommended.

### Installation

1.  Clone the repository.
2.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## 🧠 Training Workflow

**NOTE:** The provided scripts use **absolute file paths** (e.g., `C:\\Users\\farza\\PycharmProjects\\...`). You must update these paths in all relevant Python files (`split_dataset.py`, `binary_folders.py`, `feature_extraction.py`, `train_classifier.py`, `image_predictor.py`, `test_image.py`) to match your local structure before running the training steps.

### 1. Data Preparation (Binary Classification)

The original 5 classes are consolidated into two:
* **Abnormal:** `Koilocytotic`, `Dyskeratotic`, `Metaplastic`.
* **Normal:** `Parabasal`, `Superficial-Intermediate`.

```bash
# Consolidate 5 classes into 'normal' and 'abnormal' folders
python src/data_preperation/binary_folders.py
