"""
data_loader.py - Manejo de descarga y procesamiento de datos
"""

import os
import pandas as pd
import yfinance as yf
import joblib
from config import *

def download_data(ticker=TICKER, start=START_DATE, end=END_DATE, max_rows=MAX_ROWS, use_cache=True):
    """Descarga datos históricos desde Yahoo Finance y aplica limite de filas."""
    
    # Verificar si hay datos en cache
    if use_cache and os.path.exists(DATA_CACHE_PATH):
        print("Cargando datos desde cache...")
        return joblib.load(DATA_CACHE_PATH)
    
    print("Descargando datos desde Yahoo Finance...")
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        
        # Verificar si la descarga falló (DataFrame vacío o con error)
        if df.empty:
            print("❌ ERROR: No se pudieron descargar datos desde Yahoo Finance.")
            print("   Verifica:")
            print("   1. Tu conexión a internet")
            print("   2. El ticker 'META' es correcto")
            print("   3. Las fechas de inicio y fin son válidas")
            raise Exception("No se pudieron descargar datos reales. Verifica tu conexión y configuración.")
        
        df = df.reset_index()
        if len(df) > max_rows:
            df = df.tail(max_rows).reset_index(drop=True)
        
        # Verificar que las columnas existan antes de seleccionarlas
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        available_columns = [col for col in required_columns if col in df.columns]
        
        if len(available_columns) < len(required_columns):
            print(f"⚠️  Faltan columnas: {set(required_columns) - set(available_columns)}")
            print("💡 Ajustando estructura de datos...")
            
            # Asegurar que tengamos todas las columnas necesarias
            for col in required_columns:
                if col not in df.columns:
                    if col == 'Adj Close':
                        df['Adj Close'] = df.get('Close', 0)  # Usar Close si no hay Adj Close
                    elif col == 'Volume':
                        df['Volume'] = 1000000  # Valor por defecto
        
        # Seleccionar y renombrar columnas
        df = df[required_columns]
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Guardar en cache
        if use_cache:
            joblib.dump(df, DATA_CACHE_PATH)
            print(f"✅ Datos guardados en cache: {DATA_CACHE_PATH}")
        
        return df
    
    except Exception as e:
        print(f"❌ ERROR CRÍTICO al descargar datos: {e}")
        print("   No se pueden generar datos de ejemplo. Solo se aceptan datos reales.")
        raise Exception(f"Error al descargar datos reales: {e}")

def compute_technical_features(df):
    """Crea features simples y ligeros para el modelo."""
    df = df.copy()
    
    # returns
    df['Return'] = df['Close'].pct_change()
    
    # moving averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    
    # volatility (std of returns)
    df['Volatility_5'] = df['Return'].rolling(window=5).std()
    
    # price range
    df['Range'] = (df['High'] - df['Low']) / df['Open']
    
    # lag features (1..3 days)
    for lag in range(1, 4):
        df[f'lag_close_{lag}'] = df['Close'].shift(lag)
        df[f'lag_vol_{lag}'] = df['Volume'].shift(lag)
    
    # RSI (simple implementation)
    delta = df['Close'].diff()
    up = delta.where(delta > 0, 0.0)
    down = -delta.where(delta < 0, 0.0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['RSI_14'] = 100.0 - (100.0 / (1.0 + rs))
    
    # dropna
    df = df.dropna().reset_index(drop=True)
    
    return df

def clear_data_cache():
    """Elimina el cache de datos"""
    if os.path.exists(DATA_CACHE_PATH):
        os.remove(DATA_CACHE_PATH)
        print("Cache de datos eliminado")
    else:
        print("No hay cache de datos para eliminar")