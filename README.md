# MasterThesis_AffectiveComputing
- Title: A prototype design of a biofeedback system to elicit emotional responses in artwork spectators
- Author: Ana Maria Posada Ramirez
- February 11, 2025
- Master Thesis with HUMAN-IST group, DIUF, University of Fribourg
  
# Emotion Recognition from Biosignals in Art Perception

This project explores how affective computing and biosignal analysis can help understand emotional responses to artworks. It integrates data collected from the Empatica E4 sensor with machine learning models trained using the DREAMER dataset to predict levels of arousal and valence during artistic experiences.

---

## Project Structure

### Streamlit Frontend (Main Prototype)
- **`Main.py`**: User interface developed with [Streamlit](https://streamlit.io/) to collect self-reported emotions and visualize basic biosignal statistics.
- **`config.py`**: Stores constants and configuration parameters.

---

## Dataset Preprocessing

### DREAMER Dataset  
Four types of preprocessing were applied:
- **`DREAMER_raw.py`**: Uses raw ECG signals.
- **`DREAMER_extremos.py`**: Applies IQR-based outlier removal.
- **`DREAMER_Norm.py`**: Normalizes using baseline segment.
- **`DREAMER_HR.py`**: Extracts HRV features (e.g., RMSSD, LF, HF, LF/HF).

### Empatica E4 Dataset
- **`Emp_Raw.py`**: Applies similar processing (Raw data, baseline split, normalization and HRV extraction) to BVP signals from the wearable sensor Empatica e4.

---

## Machine Learning Models

- Models Trained:
  - **SVM**
  - **1D-CNN**
  - **LSTM**

- Labels:
  - Arousal and Valence (binary classes derived from ratings)

- Evaluation Metrics:
  - **Accuracy**
  - **F1-Score**

- Best-performing models were saved using:
  - `joblib` (`.pkl`) for SVM
  - `.keras` format for CNN/LSTM

---

## Emotion Prediction

- **`prediction.py`**: Loads pre-trained models and scalers to predict arousal and valence from new BVP signals collected with Empatica E4.
- Predictions are appended to the user dataset and saved as CSV.

---

## Tools and Libraries

- **Language**: Python
- **Key Libraries**:
  - `pandas`, `numpy` – data manipulation
  - `scikit-learn` – ML models, metrics, scalers
  - `tensorflow/keras` – deep learning (CNN/LSTM)
  - `neurokit2` – HRV feature extraction
  - `joblib` – model persistence
  - `streamlit` – frontend web interface

---

## Visualizations & Outputs

- Metrics (accuracy and F1-score) are printed during training.
- Predictions are saved and later compared with self-assessments and artwork's intended emotions using agreement metrics.

---

## To Do / Future Work

- [ ] Finalize and integrate `emotion_detection.py` for real-time classification.
- [ ] Implement `emotion_alignment.py` with complete Krippendorff’s Alpha support.
- [ ] Improve visualization with interactive confusion matrices and prediction feedback.
- [ ] Extend model evaluation with other datasets (e.g., DEAP, AMIGOS).
- [ ] Expand to multi-label or continuous emotion prediction.

---

## License

This project is part of a master’s thesis and is for academic use only.
