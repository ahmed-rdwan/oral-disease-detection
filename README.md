# 🦷 Oral Disease Classifier

A deep learning pipeline that classifies photos of oral/dental conditions into **6 categories**, benchmarked across four architectures and shipped as a live, interactive web app.

**🔗 Live demo:** https://oral-disease-detection-cnn.streamlit.app/
**💻 Repository:** https://github.com/ahmed-rdwan/oral-disease-detection
**📓 Training notebook:** [`Model.ipynb`](./Model.ipynb)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project builds an image classifier that identifies **6 oral health conditions** from a single photo:

| Class | Description |
|---|---|
| Calculus | Hardened dental plaque (tartar) |
| Caries | Tooth decay / cavities |
| Gingivitis | Gum inflammation |
| Hypodontia | Congenitally missing teeth |
| Mouth Ulcer | Oral sores/ulcers |
| Tooth Discoloration | Staining/discoloration of teeth |

Four architectures were trained and benchmarked on the same data to pick the best-performing model for production, which was then wrapped in a **Streamlit web app** anyone can try from a browser — upload a photo, get instant predictions with a full confidence breakdown across all 6 classes.

> ⚠️ **Disclaimer:** This is an educational/portfolio project, **not a medical device**. Predictions must not be used for real diagnosis — always consult a dentist or doctor for oral health concerns.

---

## 🧠 Approach

### 1. Dataset

- Source: [Oral Diseases dataset](https://www.kaggle.com/datasets/salmansajid05/oral-diseases) (Kaggle, via `kagglehub`).
- The raw dataset ships with a messy mix of original images, pre-augmented images, and YOLO-format annotation folders spread across inconsistent subfolder structures.

### 2. Data Preprocessing & Restructuring Pipeline

To avoid data leakage (mixing augmented copies of the same source image across train/val) and build a clean `ImageFolder`-style structure:

- **Cleansing** — discarded all pre-augmented and YOLO-annotated directories, keeping only raw original images per class.
- **Restructuring** — consolidated raw images into a master directory with one clean subfolder per class (`Calculus`, `Caries`, `Gingivitis`, `Hypodontia`, `Mouth_Ulcer`, `Tooth_Discoloration`).
- **Train/Val split** — a strict, randomized **80% / 20%** split using `split-folders` (`seed=42`), so validation images are never seen during training.
- **Class imbalance** — the raw classes are heavily imbalanced (e.g. far fewer Caries/Ulcer images than Gingivitis). Handled via:
  - `sklearn.utils.class_weight.compute_class_weight("balanced", ...)` during training (not by duplicating images), so the validation set stays untouched and representative.
  - On-the-fly data augmentation applied **only** to the training set (random flip, rotation, zoom, translation, contrast, brightness, Gaussian noise) via a `keras.Sequential` augmentation block.

### 3. Data Pipeline

- Images loaded with `tf.keras.utils.image_dataset_from_directory`, resized to **224×224**, batch size **32**.
- `cache()` + `prefetch(AUTOTUNE)` for fast, non-blocking training I/O.

### 4. Models Trained & Compared

| Model | Type | Notes |
|---|---|---|
| **Custom CNN** | Built from scratch | 4 convolutional blocks (32→64→128→256 filters), BatchNorm + ReLU after every conv, MaxPooling + increasing Dropout (0.20→0.35) per block, `GlobalAveragePooling2D`, Dense(256, L2-regularized) head, Dropout(0.5), Softmax(6). Hyperparameter search over learning rate `{1e-3, 1e-4}` × dropout `{0.3, 0.5}` before the full training run. |
| **EfficientNetB0** | Transfer learning (ImageNet) | Frozen base + custom classification head |
| **ResNet50** | Transfer learning (ImageNet) | Frozen base + custom classification head — **best performer, selected for production** |
| **MobileNetV2** | Transfer learning (ImageNet) | Frozen base + custom classification head |

All models were trained with:
- `Adam` optimizer
- `sparse_categorical_crossentropy` loss
- Class weights to counter imbalance
- Callbacks: `ReduceLROnPlateau`, `EarlyStopping` (restore best weights), `ModelCheckpoint` (best val_accuracy), `CSVLogger`

### 5. Evaluation & Model Selection

Each model was evaluated on the held-out validation set using validation accuracy, validation loss, and **F1-score (macro-averaged)** — chosen specifically because of the class imbalance, so minority classes (Caries, Mouth Ulcer, Tooth Discoloration) aren't masked by majority-class performance.

**Selected model: ResNet50**

| Metric | Value |
|---|---|
| Validation Accuracy | **78.1%** |
| Validation Loss | 0.571 |
| F1-score (macro) | **77.6%** |

The best model (by F1-macro) is exported with its preprocessing config:
- `model.keras` — the trained model weights + architecture
- `metadata.json` — class names, input image size, which backbone-specific preprocessing to apply, and the model's evaluation metrics

---

## 🚀 Deployment

The production model is served through a **Streamlit** app (`app.py`):

1. User uploads a photo.
2. The image is resized to the model's expected input size and preprocessed with ResNet50's preprocessing function.
3. The model returns a probability distribution across all 6 classes.
4. Predictions are displayed ranked by confidence, with a progress bar per class.

Hosted for free on **Streamlit Community Cloud**, connected directly to this GitHub repo — every push to `main` redeploys automatically.

---

## 🗂️ Repository Structure

```
.
├── Model.ipynb          # Full training pipeline: data prep, training, evaluation, export
├── app.py                # Production inference app (Streamlit)
├── requirements.txt      # Python dependencies
├── packages.txt          # System-level packages for Streamlit Cloud
├── runtime.txt           # Python version pin
├── model.keras           # Best model (ResNet50, selected by F1-macro)
├── metadata.json          # Class names, image size, preprocessing, metrics
└── README.md
```

---

## 🛠️ Tech Stack

- **Modeling:** TensorFlow / Keras, `EfficientNetB0`, `ResNet50`, `MobileNetV2` (ImageNet transfer learning)
- **Data handling:** NumPy, Pandas, `split-folders`, `kagglehub`
- **Evaluation:** scikit-learn (`classification_report`, `confusion_matrix`, `f1_score`, `class_weight`)
- **Visualization:** Matplotlib, Seaborn
- **Serving / Demo:** Streamlit, Pillow
- **Deployment:** Streamlit Community Cloud

---

## ▶️ Run Locally

```bash
git clone https://github.com/ahmed-rdwan/oral-disease-detection.git
cd oral-disease-detection
pip install -r requirements.txt
streamlit run app.py
```

`model.keras` and `metadata.json` are already included in the repo root, so no extra setup is needed.

---

## 📈 Possible Next Steps

- Fine-tune ResNet50 (unfreeze the last N layers) instead of using it purely as a frozen feature extractor.
- Add Grad-CAM visualizations so predictions are explainable (show *where* the model is looking).
- Expand the dataset / collect more minority-class samples (Caries, Mouth Ulcer, Tooth Discoloration) to close the gap with the majority classes.
- Add automated tests and a CI pipeline for the inference app.

---

## 🙏 Acknowledgments

- Dataset: [Oral Diseases](https://www.kaggle.com/datasets/salmansajid05/oral-diseases) by Salman Sajid on Kaggle.


## 👤 Author

**Ahmed Radwan** — [GitHub](https://github.com/ahmed-rdwan)
