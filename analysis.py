"""
analysis.py - Funciones de análisis y estadísticas
"""

import pandas as pd

def analyze_basic_stats(df):
    """Devuelve estadísticas básicas y gráficos simples."""
    stats = {
        'last_close': float(df['Close'].iloc[-1]),
        'mean_close': float(df['Close'].mean()),
        'median_volume': float(df['Volume'].median()),
        'volatility_20': float(df['Close'].pct_change().rolling(20).std().dropna().iloc[-1])
    }
    return stats

def compare_recent_vs_historical(df, window=30):
    """Compara últimas n días vs histórico (promedios)."""
    recent = df.tail(window)
    comparison = {
        'recent_mean_close': float(recent['Close'].mean()),
        'overall_mean_close': float(df['Close'].mean()),
        'recent_volatility': float(recent['Close'].pct_change().std()),
        'overall_volatility': float(df['Close'].pct_change().std())
    }
    return comparison

def get_data_summary(df):
    """Resumen completo de los datos"""
    summary = {
        'start_date': df['Date'].iloc[0].strftime('%Y-%m-%d'),
        'end_date': df['Date'].iloc[-1].strftime('%Y-%m-%d'),
        'total_rows': len(df),
        'columns': list(df.columns),
        'data_types': df.dtypes.to_dict()
    }
    return summary