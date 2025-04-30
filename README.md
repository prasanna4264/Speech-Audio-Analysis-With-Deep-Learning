
# ANALYZING SPEECH AUDIO FILES WITH DEEP LEARNING

This project implements a deep learning pipeline for **Speech Emotion Recognition (SER)** using multiple public datasets and audio features such as **MFCCs**, **Zero Crossing Rate (ZCR)**, and **Root Mean Square (RMS)** Energy. 
We provide two implementations:

- A **Jupyter Notebook version (TensorFlow)** for experimentation and visualization
- A **Script-based version (PyTorch)** for production-ready training and evaluation

---

## Datasets Used
- **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset)
- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **SAVEE** (Surrey Audio-Visual Expressed Emotion)
- **TESS** (Toronto Emotional Speech Set)

---

## Feature Extraction
Each audio file is processed to extract using the **featuresExtraction.py** script:
- **MFCCs** (Mel Frequency Cepstral Coefficients)
- **Zero Crossing Rate (ZCR)**
- **RMS Energy**


Use `notebooks/featuresExtraction.py` to extract and store these features as `.npy` files.

---

## Data Distribution
- Emotions: `neutral`, `happy`, `sad`, `angry`, `fear`, `disgust`
- Option to train gender-specific models (e.g., female-only subset)

---
## Model Architectures

### Notebook Version (TensorFlow)
A **Conv1D Neural Network** with:
- Two Conv1D layers
- GlobalAveragePooling
- Dense + Dropout layers
- Softmax output layer (6 emotion classes)

**Performance**:
- Validation Accuracy: ~92%
- Test Accuracy: ~83%
- Best result after Optuna tuning: **~85.4% Accuracy**

**Hyperparameter Optimization**:
Model performance is enhanced using **Optuna** with:
- **PatientPruner + MedianPruner**
- Optimized filters, kernel sizes, dropout rates, learning rates, batch size

Reached an accuracy ~ 94%.

---

### Script Version (PyTorch)
Structured for fast training and deployment:
- Modular files: `model.py`, `data_loading.py`, `train.py`, `evaluate.py`
- Saved best model: `best_model.pth`
- Auto-generated plots:
  - `accuracy.png`
  - `confusion_matrix.png`

Run via:
python main.py --csv_path path/to/df.csv

---


## Evaluation
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix Visualizations
- Accuracy/Loss Graphs

---

## Folder Structure
```
project/
├── mfccs/                   # MFCC .npy files
├── zcr/                     # ZCR .npy files
├── rms/                     # RMS Energy .npy files
├── ravdess/, crema/, ...    # Audio datasets
├── df.csv                   # Metadata CSV
├── main.py                  # PyTorch training entry point
├── accuracy.png             # Accuracy curve
├── confusion_matrix.png     # Confusion matrix
├── best_model.pth           # Best PyTorch model weights
├── notebooks/               # TensorFlow + notebook version
│   ├── featuresExtraction.py
│   ├── speech-emotion-recognition.ipynb
│   ├── conv1dModel_female.keras
│   └── best_optuna_conv1d_model.keras
├── src/                     # PyTorch scripts
│   ├── model.py
│   ├── data_loading.py
│   ├── evaluate.py
│   └── train.py
└── README.md
```

---

## Acknowledgements
- Datasets by CREMA-D, RAVDESS, TESS, and SAVEE
- Inspired by various SER research papers and open-source repositories

---

## Contact
Maintained by **Prasanna Adhikari**  
[adhikapa@mail.uc.edu]

---

## ⭐ If you found this helpful, star this repo!
