import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib
from tensorflow.keras.models import load_model

# Ruta del archivo
file_path = r"C:\Thesis\Code\data\dreamer data\dreamer_data_normalized_signal2.csv"

# Cargar el dataset
data = pd.read_csv(file_path)

# Procesamiento de datos
features = data.drop(columns=['Valence', 'Arousal', 'Participant', 'Video', 'Gender', 'Age'])
valence = data['Valence']
arousal = data['Arousal']

# Convertir las etiquetas en clases discretas (Bajo: 1-2, Alto: 3-5)
valence_classes = pd.cut(valence, bins=[0, 2, 5], labels=[0, 1], include_lowest=True)  # 0 = Bajo, 1 = Alto
arousal_classes = pd.cut(arousal, bins=[0, 2, 5], labels=[0, 1], include_lowest=True)  # 0 = Bajo, 1 = Alto

# Escalar los datos
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train_valence, y_test_valence = train_test_split(
    features_scaled, valence_classes, test_size=0.2, random_state=42
)
X_train, X_test, y_train_arousal, y_test_arousal = train_test_split(
   features_scaled, arousal_classes, test_size=0.2, random_state=42
)

# Convertir las etiquetas a formato categórico para modelos de red neuronal
y_train_valence_categorical = to_categorical(y_train_valence)
y_test_valence_categorical = to_categorical(y_test_valence)
y_train_arousal_categorical = to_categorical(y_train_arousal)
y_test_arousal_categorical = to_categorical(y_test_arousal)

# Modelo 1: 1D Convolutional Neural Network (1D-CNN)
def create_cnn(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Modelo 2: LSTM
def create_lstm(input_shape, num_classes):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        LSTM(64),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Modelo 3: SVM Classifier
svm_classifier = SVC(kernel='rbf', probability=True)

# Entrenamiento y evaluación
def train_and_evaluate_cnn(X_train, y_train, X_test, y_test):
    input_shape = (X_train.shape[1], 1)
    X_train = X_train.reshape(-1, X_train.shape[1], 1)
    X_test = X_test.reshape(-1, X_test.shape[1], 1)
    model = create_cnn(input_shape, y_train.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    _, accuracy = model.evaluate(X_test, y_test)
    print(f"---> 1D-CNN Accuracy: {accuracy:.4f}")
    
    # Calcular las predicciones
    y_pred = (model.predict(X_test) > 0.5).astype(int)  # Convertir a clases binarias
    
    # Calcular el F1-score
    f1 = f1_score(y_test, y_pred, average='weighted')  # 'weighted' para clases desbalanceadas
    print(f"---> 1D-CNN F1-Score: {f1:.4f}")
    return model

def train_and_evaluate_lstm(X_train, y_train, X_test, y_test):
    input_shape = (X_train.shape[1], 1)
    X_train = X_train.reshape(-1, X_train.shape[1], 1)
    X_test = X_test.reshape(-1, X_test.shape[1], 1)
    model = create_lstm(input_shape, y_train.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    _, accuracy = model.evaluate(X_test, y_test)
    print(f"---> LSTM Accuracy: {accuracy:.4f}")
    
    # Calcular las predicciones
    y_pred = (model.predict(X_test) > 0.5).astype(int)  # Convertir a clases binarias
    
    # Calcular el F1-score
    f1 = f1_score(y_test, y_pred, average='weighted')  # 'weighted' para clases desbalanceadas
    print(f"---> LSTM F1-Score: {f1:.4f}")
    return model

def train_and_evaluate_svm(X_train, y_train, X_test, y_test):
    svm_classifier.fit(X_train, y_train)
    predictions = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"---> SVM train & Test Accuracy: {accuracy:.4f}")
    #print(classification_report(y_test, predictions))

    # Calcular el F1-score
    f1 = f1_score(y_test, predictions, average='weighted')  # 'weighted' para clases desbalanceadas
    print(f"---> SVM train & Test F1-Score: {f1:.4f}")
    return svm_classifier


def train_and_evaluate_svm_with_cv(X, y, n_splits=10):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    accuracies = []
    f1_scores = []
    
    fold = 1
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Entrenar el modelo SVM
        svm_classifier.fit(X_train, y_train)
        
        # Realizar predicciones
        predictions = svm_classifier.predict(X_test)
        
        # Calcular Accuracy y F1-Score
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        
        print(f"Fold {fold} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        
        accuracies.append(accuracy)
        f1_scores.append(f1)
        fold += 1
    
    # Resultados promedio
    mean_accuracy = np.mean(accuracies)
    mean_f1 = np.mean(f1_scores)
    print(f"\n---> SVM 10-Fols CV Avg Accuracy: {mean_accuracy:.4f}")
    print(f"---> SVM 10-Fols CV Avg F1-Score: {mean_f1:.4f}")
    
    # Retornar métricas
    return {
        "fold_accuracies": accuracies,
        "fold_f1_scores": f1_scores,
        "mean_accuracy": mean_accuracy,
        "mean_f1_score": mean_f1
    }



print("-------------------------")
print("-------------------------")
print("-----Training models-----")
print("-------------------------")
print("-------------------------")

#Entrenar y evaluar modelos para Valence
print("Training models for Valence...")

model_cnn_valence = train_and_evaluate_cnn(X_train, y_train_valence_categorical, X_test, y_test_valence_categorical)
# Guardar el modelo 1D-CNN para Valence Normalized (signal 2)
#joblib.dump(scaler, "scaler_valence.pkl")
#model_cnn_valence.save("model_cnn_valence.keras")

lstm_valence_model = train_and_evaluate_lstm(X_train, y_train_valence_categorical, X_test, y_test_valence_categorical)
svm_valence_model = train_and_evaluate_svm(X_train, y_train_valence, X_test, y_test_valence)
# Combinar X_train y X_test, y_train_valence y y_test_valence
X = np.concatenate((X_train, X_test), axis=0)
y_valence = np.concatenate((y_train_valence, y_test_valence), axis=0)
# Llamar al método corregido
svm_10_valence_model = train_and_evaluate_svm_with_cv(X, y_valence)


# Entrenar y evaluar modelos para Arousal
print("Training models for Arousal...")
cnn_arousal_model = train_and_evaluate_cnn(X_train, y_train_arousal_categorical, X_test, y_test_arousal_categorical)
lstm_arousal_model = train_and_evaluate_lstm(X_train, y_train_arousal_categorical, X_test, y_test_arousal_categorical)
#joblib.dump(scaler, "scaler_lstm.pkl")
#lstm_arousal_model.save("model_lstm_arousal.keras")

model_svm_arousal = train_and_evaluate_svm(X_train, y_train_arousal, X_test, y_test_arousal)
# Guardar el modelo SVM para Arousal  Normalized (signal 1)
#joblib.dump(scaler, "scaler_arousal.pkl")
#joblib.dump(model_svm_arousal, "model_svm_arousal.pkl")

# Combinar X_train y X_test, y_train_arousal y y_test_arousal
X = np.concatenate((X_train, X_test), axis=0)
y_arousal = np.concatenate((y_train_arousal, y_test_arousal), axis=0)
# Llamar al método corregido
svm_10_arousal_model = train_and_evaluate_svm_with_cv(X, y_arousal)
#joblib.dump(scaler, "scaler_arousal.pkl")
#joblib.dump(svm_10_arousal_model, "model_svm_arousal.pkl")