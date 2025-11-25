"""
big_data.py - Módulo para procesamiento masivo de datos (GDELT + DuckDB)
"""

import os
import requests
import zipfile
import duckdb
import pandas as pd
from datetime import datetime, timedelta
import io

# Configuración
GDELT_BASE_URL = "http://data.gdeltproject.org/events/"
DATA_DIR = "gdelt_data"
DB_FILE = "gdelt_meta.duckdb"

def download_gdelt_slice(start_date, days=7):
    """
    Descarga un 'slice' (fragmento) de datos de GDELT para los últimos 'days' días.
    GDELT publica archivos CSV diarios comprimidos en ZIP.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    print(f"\n=== BIG DATA: Descargando GDELT (Slice de {days} días) ===")
    
    downloaded_files = []
    
    # GDELT usa formato YYYYMMDD.export.CSV.zip
    current_date = pd.to_datetime(start_date)
    
    for _ in range(days):
        date_str = current_date.strftime("%Y%m%d")
        filename = f"{date_str}.export.CSV.zip"
        url = f"{GDELT_BASE_URL}{filename}"
        local_path = os.path.join(DATA_DIR, filename)
        
        # Solo descargar si no existe
        if not os.path.exists(local_path):
            print(f"⬇️ Descargando: {filename} ...")
            try:
                r = requests.get(url, stream=True)
                if r.status_code == 200:
                    with open(local_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            f.write(chunk)
                    downloaded_files.append(local_path)
                else:
                    print(f"⚠️ Archivo no encontrado en GDELT: {filename} (puede que aún no esté publicado)")
            except Exception as e:
                print(f"❌ Error descargando {filename}: {e}")
        else:
            print(f"✅ Ya existe: {filename}")
            downloaded_files.append(local_path)
            
        current_date -= timedelta(days=1)
        
    print(f"Total archivos listos para procesar: {len(downloaded_files)}")
    return downloaded_files

def unzip_files():
    """Descomprime los archivos ZIP descargados para facilitar la lectura por DuckDB"""
    print("\n=== BIG DATA: Descomprimiendo archivos... ===")
    zip_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".zip")]
    
    extracted_files = []
    for zip_file in zip_files:
        try:
            zip_path = os.path.join(DATA_DIR, zip_file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
                # Asumimos que hay un archivo CSV dentro
                for name in zip_ref.namelist():
                    extracted_path = os.path.join(DATA_DIR, name)
                    if os.path.exists(extracted_path):
                        extracted_files.append(extracted_path)
                        print(f"📦 Extraído: {name}")
        except Exception as e:
            print(f"❌ Error descomprimiendo {zip_file}: {e}")
            
    return extracted_files

def extract_and_process_duckdb():
    """
    Usa DuckDB para leer los CSVs extraídos.
    Filtra eventos relacionados con 'META', 'FACEBOOK', 'SOCIAL MEDIA', etc.
    """
    print("\n=== BIG DATA: Procesando con DuckDB (In-Memory/On-Disk) ===")
    
    # Primero descomprimimos
    unzip_files()
    
    # Patrón de archivos para DuckDB (CSVs extraídos)
    # GDELT export files usually end in .export.CSV
    files_pattern = os.path.join(DATA_DIR, "*.export.CSV")
    
    # Conexión a DuckDB
    con = duckdb.connect(database=DB_FILE, read_only=False)
    
    # ... (resto del comentario igual)
    
    # Definición de columnas explícita para evitar errores de sniffing
    # GDELT 1.0 tiene 58 columnas. Las definimos todas como VARCHAR para seguridad.
    # col1 = SQLDATE, col6 = Actor1Name, col34 = AvgTone
    cols_def = ", ".join([f"'col{i}': 'VARCHAR'" for i in range(58)])
    columns_struct = "{" + cols_def + "}"

    print("🦆 Ejecutando Query SQL sobre archivos CSV...")
    
    # Esta query filtra filas donde el Actor1Name contenga 'FACEBOOK' o 'META'
    # Usamos read_csv con parámetros robustos: sin quoting, tab separator, sin header.
    query = f"""
        SELECT 
            col1 as DateInt,
            AVG(TRY_CAST(col34 as FLOAT)) as Avg_Sentiment,
            COUNT(*) as News_Volume
        FROM read_csv(
            '{files_pattern}', 
            delim='\\t', 
            header=False, 
            quote='', 
            null_padding=true, 
            ignore_errors=true,
            auto_detect=false,
            strict_mode=false,
            columns={columns_struct}
        )
        WHERE 
            UPPER(col6) LIKE '%FACEBOOK%' 
            OR UPPER(col6) LIKE '%META PLATFORMS%'
            OR UPPER(col6) LIKE '%ZUCKERBERG%'
            OR UPPER(col16) LIKE '%FACEBOOK%' 
            OR UPPER(col16) LIKE '%META PLATFORMS%'
            OR UPPER(col16) LIKE '%ZUCKERBERG%'
            OR UPPER(col57) LIKE '%FACEBOOK%'
            OR UPPER(col57) LIKE '%META PLATFORMS%'
            OR UPPER(col57) LIKE '%ZUCKERBERG%'
        GROUP BY col1
        ORDER BY col1
    """
    
    try:
        # Ejecutar y convertir a Pandas (el resultado ya es pequeño)
        df_result = con.execute(query).df()
        
        # Post-procesamiento
        if not df_result.empty:
            df_result['Date'] = pd.to_datetime(df_result['DateInt'].astype(str), format='%Y%m%d')
            df_result = df_result[['Date', 'Avg_Sentiment', 'News_Volume']]
            print("\n✅ Procesamiento completado. Muestra de resultados:")
            print(df_result.head())
            print(f"Filas extraídas: {len(df_result)}")
        else:
            print("⚠️ La consulta no devolvió resultados (quizás no hubo noticias de META en estos días).")
            
        return df_result
        
    except Exception as e:
        print(f"❌ Error en DuckDB: {e}")
        return pd.DataFrame()
    finally:
        con.close()
        # Limpieza opcional: borrar CSVs extraídos para ahorrar espacio
        # for f in os.listdir(DATA_DIR):
        #     if f.endswith(".CSV") and not f.endswith(".zip"):
        #         os.remove(os.path.join(DATA_DIR, f))


def run_big_data_pipeline():
    """Función orquestadora para llamar desde el menú"""
    # Usar fecha de ayer hacia atrás para asegurar que los archivos existan
    yesterday = datetime.now() - timedelta(days=2)
    
    # 1. Descargar (Fase de Adquisición)
    # Bajamos 30 días para tener una muestra decente (aprox 1.5 GB comprimido)
    # Para la prueba rápida, bajamos solo 5 días.
    print("Para la prueba rápida descargaremos 5 días. En prod serían 30+.")
    download_gdelt_slice(yesterday, days=5)
    
    # 2. Procesar (Fase de ETL con DuckDB)
    df_big_data = extract_and_process_duckdb()
    
    return df_big_data
