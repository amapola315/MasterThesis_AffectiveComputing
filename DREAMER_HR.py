import scipy.io as sio
import pandas as pd
import numpy as np
import os
import logging
import neurokit2 as nk

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_hrv_features(ecg_signal, sampling_rate=256):
    """
    Calcula características básicas de HRV (RMSSD, SDNN, LF, HF, LF/HF ratio) a partir de la señal de ECG.

    Parameters:
        ecg_signal (np.array): Señal de ECG.
        sampling_rate (int): Frecuencia de muestreo en Hz.

    Returns:
        dict: Diccionario con características de HRV.
    """
    try:
        # Seleccionar el primer canal si la señal es bidimensional
        if ecg_signal.ndim > 1:
            ecg_signal = ecg_signal[:, 1]
        
        # Procesar señal ECG para obtener R-peaks
        processed_ecg = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
        r_peaks = processed_ecg[1]["ECG_R_Peaks"]

        # Calcular características de HRV en el dominio del tiempo
        hrv_time = nk.hrv_time(r_peaks, sampling_rate=sampling_rate, show=False)[["HRV_RMSSD", "HRV_SDNN"]]
        
        # Calcular características de HRV en el dominio de la frecuencia
        hrv_freq = nk.hrv_frequency(r_peaks, sampling_rate=sampling_rate, show=False)[["HRV_LF", "HRV_HF", "HRV_LFHF"]]
        
        # Combinar características
        hrv_features = {**hrv_time.iloc[0].to_dict(), **hrv_freq.iloc[0].to_dict()}
        return hrv_features
    except Exception as e:
        logging.error(f"Error al calcular características de HRV: {e}")
        return {}

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

def extract_data_with_baseline_normalization(dreamer_data):
    """
    Extrae las puntuaciones de emoción y características de ECG de la base de datos DREAMER,
    aplicando normalización basada en el baseline.

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
            logging.info(f"Procesando participante {participant + 1} de {no_of_participants}...")
        
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
                    logging.info(f"  Procesando video {video + 1} de 18 para participante {participant + 1}...")
                
                    # Verificar que existan grabaciones de ECG para el video
                    if not (len(participant_data['ECG'][0][0]['baseline'][0][0]) > video and 
                            len(participant_data['ECG'][0][0]['stimuli'][0][0]) > video):
                        logging.warning(f"Datos de ECG faltantes para video {video + 1} del participante {participant + 1}.")
                        continue
                    
                    # Obtener grabaciones de ECG
                    ecg_baseline = participant_data['ECG'][0][0]['baseline'][0][0][video][0]
                    ecg_stimulus = participant_data['ECG'][0][0]['stimuli'][0][0][video][0]

                    # Calcular características de HRV para baseline y estímulo
                    baseline_hrv = calculate_hrv_features(ecg_baseline)
                    stimulus_hrv = calculate_hrv_features(ecg_stimulus)

                    # Agregar datos procesados a la lista
                    emotion_data.append({
                        "Age": age,
                        "Gender": gender,
                        "Participant": participant + 1,
                        "Video": video + 1,
                        "Valence": valence_scores[video][0],
                        "Arousal": arousal_scores[video][0],
                        "Dominance": dominance_scores[video][0],
                        **{f"Baseline_{key}": value for key, value in baseline_hrv.items()},
                        **{f"Stimulus_{key}": value for key, value in stimulus_hrv.items()},
                    })
                except (IndexError, ValueError) as e:
                    logging.warning(f"Error al procesar el video {video + 1} del participante {participant + 1}: {e}")
        except Exception as e:
            logging.error(f"Error al procesar los datos del participante {participant + 1}: {e}")

    return pd.DataFrame(emotion_data)

def main(dreamer_matfile, output_csv_path):
    """
    Carga, procesa y guarda los datos de DREAMER en un archivo CSV con normalización basada en baseline.
    
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
        
        # Extraer datos procesados con normalización basada en baseline
        logging.info("Procesando los datos con normalización basada en baseline...")
        df = extract_data_with_baseline_normalization(dreamer_data)
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
output_csv = r"C:\Thesis\Code\data\dreamer_data_hrv_limited.csv"

# Ejecutar procesamiento
if __name__ == "__main__":
    data_df = main(ruta_mat, output_csv)
