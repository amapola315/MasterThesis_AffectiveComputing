import scipy.io as sio
import pandas as pd
import numpy as np
import os
import logging

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_ecg_data(ecg_data):
    """
    Limpia los datos de ECG eliminando valores extremos basados en el rango intercuartílico (IQR).
    
    Parameters:
        ecg_data (np.array): Datos de ECG (baseline o estímulo).
    
    Returns:
        np.array: Datos de ECG sin valores extremos.
    """
    q1 = np.percentile(ecg_data, 25)  # Primer cuartil
    q3 = np.percentile(ecg_data, 75)  # Tercer cuartil
    iqr = q3 - q1  # Rango intercuartílico
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Contar valores extremos
    outliers = ((ecg_data < lower_bound) | (ecg_data > upper_bound)).sum()
    if outliers > 0:
        logging.info(f"Se detectaron {outliers} valores extremos en ECG.")
    
    # Reemplazar valores fuera del rango con la mediana
    cleaned_data = np.where(
        (ecg_data < lower_bound) | (ecg_data > upper_bound),
        np.median(ecg_data),
        ecg_data
    )
    return cleaned_data

def load_dreamer_data(dreamer_matfile):
    """
    Carga la base de datos DREAMER desde un archivo .MAT.
    
    Parameters:
        dreamer_matfile (str): Ruta del archivo .MAT de DREAMER.
    
    Returns:
        dict: Contenido cargado del archivo .MAT.
    """
    try:
        raw = sio.loadmat(dreamer_matfile)
        if 'DREAMER' not in raw:
            raise ValueError("El archivo no contiene la variable 'DREAMER'.")
        return raw['DREAMER']
    except Exception as e:
        logging.error(f"Error al cargar el archivo {dreamer_matfile}: {e}")
        raise

def extract_data(dreamer_data):
    """
    Extrae las puntuaciones de emoción y características de ECG de la base de datos DREAMER.
    
    Parameters:
        dreamer_data (dict): Contenido de la base de datos DREAMER.
    
    Returns:
        pd.DataFrame: Datos procesados en formato de DataFrame.
    """
    emotion_data = []
    no_of_participants = len(dreamer_data[0, 0]['Data'][0])

    for participant in range(no_of_participants):
        try:
            participant_data = dreamer_data[0, 0]['Data'][0, participant]
            
            # Extraer metadata del participante
            age = participant_data['Age'][0][0][0]
            gender = participant_data['Gender'][0][0][0]

            # Extraer puntuaciones de emoción
            valence_scores = participant_data['ScoreValence'][0][0]
            arousal_scores = participant_data['ScoreArousal'][0][0]
            dominance_scores = participant_data['ScoreDominance'][0][0]

            # Procesar ECG por cada video
            for video in range(18):
                try:
                    # Verificar que existan grabaciones de ECG para el video
                    if not (len(participant_data['ECG'][0][0]['baseline'][0][0]) > video and 
                            len(participant_data['ECG'][0][0]['stimuli'][0][0]) > video):
                        logging.warning(f"Datos de ECG faltantes para video {video + 1} del participante {participant + 1}.")
                        continue
                    
                    # Obtener grabaciones de ECG
                    ecg_baseline = participant_data['ECG'][0][0]['baseline'][0][0][video][0]
                    ecg_stimulus = participant_data['ECG'][0][0]['stimuli'][0][0][video][0]

                    # Limpiar datos de ECG de valores extremos
                    ecg_baseline = clean_ecg_data(ecg_baseline)
                    ecg_stimulus = clean_ecg_data(ecg_stimulus)

                    # Calcular estadísticas de ECG
                    baseline_mean = np.mean(ecg_baseline, axis=0)
                    baseline_std = np.std(ecg_baseline, axis=0)
                    baseline_median = np.median(ecg_baseline, axis=0)  # Mediana
                    
                    stimulus_mean = np.mean(ecg_stimulus, axis=0)
                    stimulus_std = np.std(ecg_stimulus, axis=0)
                    stimulus_median = np.median(ecg_stimulus, axis=0)  # Mediana

                    # Agregar datos procesados a la lista
                    emotion_data.append({
                        "Age": age,
                        "Gender": gender,
                        "Participant": participant + 1,
                        "Video": video + 1,
                        "Valence": valence_scores[video][0],
                        "Arousal": arousal_scores[video][0],
                        "Dominance": dominance_scores[video][0],
                        "ECG_Baseline_Mean_1": baseline_mean[0],
                        "ECG_Baseline_Mean_2": baseline_mean[1],
                        "ECG_Baseline_Std_1": baseline_std[0],
                        "ECG_Baseline_Std_2": baseline_std[1],
                        "ECG_Baseline_Median_1": baseline_median[0],
                        "ECG_Baseline_Median_2": baseline_median[1],
                        "ECG_Stimulus_Mean_1": stimulus_mean[0],
                        "ECG_Stimulus_Mean_2": stimulus_mean[1],
                        "ECG_Stimulus_Std_1": stimulus_std[0],
                        "ECG_Stimulus_Std_2": stimulus_std[1],
                        "ECG_Stimulus_Median_1": stimulus_median[0],
                        "ECG_Stimulus_Median_2": stimulus_median[1],
                    })
                except (IndexError, ValueError) as e:
                    logging.warning(f"Error al procesar el video {video + 1} del participante {participant + 1}: {e}")
        except Exception as e:
            logging.error(f"Error al procesar los datos del participante {participant + 1}: {e}")

    return pd.DataFrame(emotion_data)
    
def main(dreamer_matfile, output_csv_path):
    """
    Carga, procesa y guarda los datos de DREAMER en un archivo CSV.
    
    Parameters:
        dreamer_matfile (str): Ruta del archivo .MAT de DREAMER.
        output_csv_path (str): Ruta donde se guardará el archivo CSV.
    
    Returns:
        pd.DataFrame: Datos procesados en formato de DataFrame.
    """
    try:
        # Cargar datos de DREAMER
        logging.info("Cargando datos desde el archivo .MAT...")
        dreamer_data = load_dreamer_data(dreamer_matfile)
        logging.info("Datos cargados exitosamente.")
        
        # Extraer datos procesados
        logging.info("Procesando los datos...")
        df = extract_data(dreamer_data)
        logging.info("Datos procesados exitosamente.")
        
        # Guardar en CSV
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        logging.info(f"DataFrame guardado en: {output_csv_path}")
        
        return df
    except Exception as e:
        logging.error(f"Error en el procesamiento: {e}")
        raise

# Rutas de entrada y salida
ruta_mat = r"C:\Thesis\Dataset\DREAMER\DREAMER.mat"
output_csv = r"C:\Thesis\Code\data\dreamer_data_extremos.csv"

# Ejecutar procesamiento
if __name__ == "__main__":
    data_df = main(ruta_mat, output_csv)
