import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

# Ruta del archivo BVP
base_path = r"C:\Participants data"
bvp_file = f"{base_path}\\BVP.csv"

# Cargar la señal BVP
bvp_data = pd.read_csv(bvp_file, header=None, skiprows=2, names=['BVP'])

# Tasa de muestreo (sample rate)
with open(bvp_file, 'r') as file:
    file.readline()  # Ignorar la primera línea (timestamp inicial)
    sample_rate_bvp = float(file.readline().strip())  # Leer la segunda línea (sample rate)

# Calcular el número de muestras para los primeros 40 segundos (baseline)
baseline_samples = int(40 * sample_rate_bvp)

# Dividir en baseline y estímulo
baseline_bvp = bvp_data.iloc[:baseline_samples]  # Primeros 30 segundos de BVP
stimulus_bvp = bvp_data.iloc[baseline_samples:]  # Resto de BVP

# Calcular estadísticas para BVP
baseline_mean_bvp = np.mean(baseline_bvp['BVP'])
baseline_std_bvp = np.std(baseline_bvp['BVP'])
baseline_median_bvp = np.median(baseline_bvp['BVP'])

stimulus_mean_bvp = np.mean(stimulus_bvp['BVP'])
stimulus_std_bvp = np.std(stimulus_bvp['BVP'])
stimulus_median_bvp = np.median(stimulus_bvp['BVP'])

# Calcular estadísticas normalizadas basadas en baseline
norm_mean_bvp = stimulus_mean_bvp / baseline_mean_bvp
norm_std_bvp = stimulus_std_bvp / baseline_std_bvp
norm_median_bvp = stimulus_median_bvp / baseline_median_bvp

# Calcular características de HRV a partir de BVP
def calculate_hrv_from_bvp(bvp_signal, sampling_rate):
    try:
        # Procesar la señal BVP para extraer los picos de pulso
        signals, info = nk.ppg_process(bvp_signal, sampling_rate=sampling_rate)
        r_peaks = info["PPG_Peaks"]

        # Calcular características de HRV
        hrv_time = nk.hrv_time(r_peaks, sampling_rate=sampling_rate, show=False)[["HRV_RMSSD", "HRV_SDNN"]]
        hrv_freq = nk.hrv_frequency(r_peaks, sampling_rate=sampling_rate, show=False)[["HRV_LF", "HRV_HF", "HRV_LFHF"]]

        # Combinar resultados
        return {**hrv_time.iloc[0].to_dict(), **hrv_freq.iloc[0].to_dict()}
    except Exception as e:
        print(f"Error al calcular HRV: {e}")
        return {}

# HRV para baseline
baseline_hrv = calculate_hrv_from_bvp(baseline_bvp['BVP'].values, sample_rate_bvp)

# HRV para estímulo
stimulus_hrv = calculate_hrv_from_bvp(stimulus_bvp['BVP'].values, sample_rate_bvp)


# Mostrar resultados
#print("=== Raw Data BVP  ===")
#print(f"Baseline (Mean): {baseline_mean_bvp}")
#print(f"Baseline (Std): {baseline_std_bvp}")
#print(f"Baseline (Median): {baseline_median_bvp}\n")

#print(f"Stimulus (Mean): {stimulus_mean_bvp}")
#print(f"Stimulus (Std): {stimulus_std_bvp}")
#print(f"Stimulus (Median): {stimulus_median_bvp}\n")


print("=== Norm Data BVP  ===")
print(base_path)
print(f"Norm (Mean): {norm_mean_bvp}")
print(f"Norm (Std): {norm_std_bvp}")
print(f"Norm (Median): {norm_median_bvp}")

#print("=== HRV Features (Baseline) ===")
#print(baseline_hrv)

#print("\n=== HRV Features (Stimulus) ===")
#print(stimulus_hrv)
