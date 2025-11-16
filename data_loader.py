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
    
    if use_cache and os.path.exists(DATA_CACHE_PATH):
        print("Cargando datos desde cache...")
        return joblib.load(DATA_CACHE_PATH)
    
    print("Descargando datos desde Yahoo Finance...")
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        
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
    """Crea features simples y avanzados para el modelo."""
    df = df.copy()
    
    # returns
    df['Return'] = df['Close'].pct_change()
    
    # moving averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # volatility (std of returns)
    df['Volatility_5'] = df['Return'].rolling(window=5).std()
    df['Volatility_20'] = df['Close'].rolling(window=20).std()
    
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

    # Bollinger Bands
    sma20 = df['SMA_20']
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = sma20 + 2 * bb_std
    df['BB_lower'] = sma20 - 2 * bb_std
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / (sma20 + 1e-9)

    # Momentum
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)

    # OBV
    obv_step = (df['Close'].diff().fillna(0).gt(0).astype(int) - df['Close'].diff().fillna(0).lt(0).astype(int)) * df['Volume']
    df['OBV'] = obv_step.fillna(0).cumsum()
    
    # dropna
    df = df.dropna().reset_index(drop=True)
    
    return df

def augment_with_external_factors(df):
    start = df['Date'].iloc[0]
    end = df['Date'].iloc[-1] + pd.Timedelta(days=1)
    vix = yf.download('^VIX', start=start, end=end, progress=False)
    tnx = yf.download('^TNX', start=start, end=end, progress=False)
    spx = yf.download('^GSPC', start=start, end=end, progress=False)
    for name, series in [('VIX', vix), ('TNX', tnx), ('SPX', spx)]:
        if not series.empty:
            s = series.reset_index()
            date_col = 'Date' if 'Date' in s.columns else ('Datetime' if 'Datetime' in s.columns else s.columns[0])
            price_col = 'Close' if 'Close' in s.columns else ('Adj Close' if 'Adj Close' in s.columns else None)
            if price_col is None:
                other_cols = [c for c in s.columns if c != date_col]
                if not other_cols:
                    continue
                price_col = other_cols[-1]
            s['Date'] = pd.to_datetime(s[date_col])
            vals = s[price_col]
            if isinstance(vals, pd.DataFrame):
                vals = vals.iloc[:, 0]
            s_simple = pd.DataFrame({'Date': s['Date'], name: vals})
            df = pd.merge(df, s_simple, on='Date', how='left')
            df[f'{name}_Return'] = df[name].pct_change()
    df = df.dropna().reset_index(drop=True)
    return df

def attach_news_placeholder(df):
    return df

def clear_data_cache():
    """Elimina el cache de datos"""
    if os.path.exists(DATA_CACHE_PATH):
        os.remove(DATA_CACHE_PATH)
        print("Cache de datos eliminado")
    else:
        print("No hay cache de datos para eliminar")