
# ANALYZING SPEECH AUDIO FILES WITH DEEP LEARNING

This project implements a deep learning pipeline for **Speech Emotion Recognition (SER)** using multiple public datasets and audio features such as **MFCCs**, **Zero Crossing Rate (ZCR)**, and **Root Mean Square (RMS)** Energy. 
The model architecture is built using TensorFlow and fine-tuned with Optuna, incorporating pruning strategies to explore the hyperparameter space and accelerate training efficiently. All development and experimentation are conducted in Jupyter Notebooks, with a focus on reproducibility and performance optimization.

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

Features are stored as `.npy` files and loaded into memory for model training.

---

## Data Distribution
- Emotions: `neutral`, `happy`, `sad`, `angry`, `fear`, `disgust`
- Option to train gender-specific models (e.g., female-only subset)

---

## Model Architecture
A baseline **Conv1D Neural Network** with:
- Two Conv1D layers
- GlobalAveragePooling
- Dense + Dropout layers
- Softmax output layer (6 emotion classes)

### Performance:
- Validation Accuracy: ~92%
- Test Accuracy: ~83%
- Best result after Optuna tuning: **~85.4% Accuracy**

---

## Hyperparameter Optimization
Model performance is enhanced using **Optuna** with:
- **PatientPruner + MedianPruner**
- Optimized filters, kernel sizes, dropout rates, learning rates, batch size

Reached an accuracy ~ 94%.
---

## Evaluation
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix Visualizations
- Accuracy/Loss Graphs

---

## Folder Structure
```
project/
├── mfccs/             # MFCC feature .npy files
├── zcr/               # ZCR feature .npy files
├── rms/               # RMS Energy feature .npy files
├── ravdess/, crema/, tess/, savee/  # Audio datasets
├── df.csv             # DataFrame with file paths, gender, emotion
├── featuresExtraction.py
├── conv1dModel_female.keras
├── best_optuna_conv1d_model.keras
├── speech-emotion-recognition.ipynb
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
