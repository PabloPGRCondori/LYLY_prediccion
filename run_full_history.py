"""
run_full_history.py - Script para descarga masiva de datos GDELT (1 Año)
"""
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm  # Barra de progreso
import big_data  # Reutilizamos lógica existente
import duckdb

# Configuración para la "Gran Descarga"
DAYS_TO_DOWNLOAD = 365
HISTORY_FILE = "gdelt_history_1year.csv"
DATA_DIR = "gdelt_data"

def download_file_with_progress(url, local_path):
    """Descarga con barra de progreso y manejo de errores"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024 # 1MB
            
            with open(local_path, 'wb') as f, tqdm(
                desc=os.path.basename(local_path),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                leave=False
            ) as bar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    bar.update(size)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error descargando {url}: {e}")
        return False

def run_full_history_download():
    print("==================================================")
    print(f"🚀 INICIANDO 'GRAN DESCARGA' - HISTORIAL {DAYS_TO_DOWNLOAD} DÍAS")
    print("==================================================\n")
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 1. Definir rango de fechas (Desde hace 2 días hacia atrás)
    end_date = datetime.now() - timedelta(days=2)
    start_date = end_date - timedelta(days=DAYS_TO_DOWNLOAD)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    print(f"📅 Rango: {start_date.date()} al {end_date.date()}")
    print(f"📦 Total archivos a procesar: {len(dates)}\n")
    
    downloaded_count = 0
    
    # 2. Bucle de Descarga
    print("--- PASO 1: Descarga Masiva ---")
    for date in tqdm(dates, desc="Progreso Total"):
        date_str = date.strftime("%Y%m%d")
        filename = f"{date_str}.export.CSV.zip"
        url = f"{big_data.GDELT_BASE_URL}{filename}"
        local_path = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(local_path):
            success = download_file_with_progress(url, local_path)
            if success:
                downloaded_count += 1
        else:
            # Si ya existe, asumimos que está bien
            pass
            
    print(f"\n✅ Descarga finalizada. Nuevos archivos: {downloaded_count}")
    
    # 3. Procesamiento (Reutilizando lógica de DuckDB)
    print("\n--- PASO 2: Procesamiento Masivo con DuckDB ---")
    # Nota: big_data.extract_and_process_duckdb() procesa TODO lo que hay en la carpeta
    # así que automáticamente tomará el año completo.
    
    try:
        df_history = big_data.extract_and_process_duckdb()
        
        if not df_history.empty:
            # Guardar CSV histórico
            print(f"\n💾 Guardando historial consolidado en: {HISTORY_FILE}")
            df_history.to_csv(HISTORY_FILE, index=False)
            print("✅ ¡Misión Cumplida! Historial generado exitosamente.")
        else:
            print("⚠️ El procesamiento terminó sin datos (revisar logs).")
            
    except Exception as e:
        print(f"❌ Error crítico en procesamiento: {e}")

if __name__ == "__main__":
    # Verificar si tqdm está instalado, si no, instalarlo
    try:
        import tqdm
    except ImportError:
        print("Instalando dependencia tqdm para barra de progreso...")
        os.system("pip install tqdm")
        import tqdm
        
    run_full_history_download()
