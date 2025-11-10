"""
config.py - Configuración centralizada del sistema
"""

# -----------------------------
# CONFIGURACIÓN (ajustable)
# -----------------------------
TICKER = "META"                      # ticker a analizar
START_DATE = "2021-01-01"            # fecha inicio (puedes acortarla para menos procesamiento)
END_DATE = None                      # None -> hasta hoy
HORIZON_DAYS = 7                     # horizonte de predicción (días futuros que evaluamos)
HORIZON_HOURS = 10                    # horizonte de predicción en horas
MAX_ROWS = 1000                      # límite de días descargados (reduce carga)
RANDOM_FOREST_N_ESTIMATORS = 100     # reduce si quieres menor CPU (ej: 50)
RANDOM_STATE = 42
USE_PROPHET = False
MODEL_PATH = "rf_meta_model.joblib"  # archivo para guardar modelo
PLOT_PATH = "plots"                  # carpeta donde guardar png
DATA_CACHE_PATH = "data_cache.joblib" # cache de datos procesados