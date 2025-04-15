import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Rutas de archivos
scaler_arousal_path = "scaler_arousal.pkl"
model_svm_arousal_path = "model_svm_arousal.pkl"
scaler_valence_path = "scaler_valence.pkl"
model_cnn_valence_path = "model_cnn_valence.keras"

empatica_input_path = r"C:\Thesis\Code\data\e4_data_normalized.csv"
output_path = r"C:\Thesis\Code\data\e4_data_predict.csv"

# Cargar los modelos entrenados
print("Cargando modelos entrenados y scalers...")
scaler_arousal = joblib.load(scaler_arousal_path)
model_svm_arousal = joblib.load(model_svm_arousal_path)

scaler_valence = joblib.load(scaler_valence_path)
model_cnn_valence = load_model(model_cnn_valence_path)

# Cargar datos de Empatica
print("Cargando datos de Empatica...")
empatica_data = pd.read_csv(empatica_input_path)

# Preprocesar los datos nuevos
print("Preprocesando datos de Empatica...")
features_empatica = empatica_data.drop(columns=['Participant', 'Video', 'Gender', 'Age', 'Valence', 'Arousal'])

# Escalado para Arousal
features_empatica_scaled_arousal = scaler_arousal.transform(features_empatica)

# Escalado para Valence
features_empatica_scaled_valence = scaler_valence.transform(features_empatica)
features_empatica_scaled_valence = features_empatica_scaled_valence.reshape(-1, features_empatica_scaled_valence.shape[1], 1)

# Predicción para Arousal (SVM)
print("Realizando predicciones para Arousal (SVM)...")
arousal_classes = model_svm_arousal.predict(features_empatica_scaled_arousal)

# Predicción para Valence (1D-CNN)
print("Realizando predicciones para Valence (1D-CNN)...")
valence_probabilities = model_cnn_valence.predict(features_empatica_scaled_valence)
valence_classes = np.argmax(valence_probabilities, axis=1)

# Agregar predicciones al DataFrame de Empatica
print("Agregando predicciones al archivo de datos...")
empatica_data["Predicted_Arousal"] = arousal_classes
empatica_data["Predicted_Valence"] = valence_classes

# Guardar el archivo con predicciones
empatica_data.to_csv(output_path, index=False)
print(f"Resultados guardados en {output_path}")
