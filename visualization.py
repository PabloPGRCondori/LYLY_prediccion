"""
visualization.py - Funciones de visualización y gráficos
"""

import matplotlib.pyplot as plt
import os
from config import *

def plot_price_and_forecast(df, y_test_indexed, y_test, y_pred, save_prefix=os.path.join(PLOT_PATH, "price_forecast")):
    """Grafica histórico close y predicciones sobre el conjunto de test."""
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Close'], label='Histórico Close', linewidth=1)
    
    # plot test true vs pred (align by Date index)
    test_dates = df['Date'].iloc[y_test_indexed.index]
    plt.plot(test_dates, y_test.values, label='Test - Real', linestyle='--')
    plt.plot(test_dates, y_pred, label='Test - Pred (RF)', linestyle=':')
    
    plt.title(f'{TICKER} - Histórico y Predicción (RandomForest)')
    plt.xlabel('Fecha')
    plt.ylabel('Precio (USD)')
    plt.legend()
    out_png = f"{save_prefix}.png"
    plt.tight_layout()
    plt.savefig(out_png)
    plt.show()

def plot_indicators(df, save_prefix=os.path.join(PLOT_PATH, "indicators")):
    """Grafica indicadores simples: SMA y RSI."""
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Close'], label='Close')
    
    if 'SMA_5' in df.columns and 'SMA_10' in df.columns:
        plt.plot(df['Date'], df['SMA_5'], label='SMA 5')
        plt.plot(df['Date'], df['SMA_10'], label='SMA 10')
    
    plt.title(f'{TICKER} - Precio y SMAs')
    plt.legend()
    plt.xlabel('Fecha')
    plt.ylabel('USD')
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_sma.png")
    plt.show()

    if 'RSI_14' in df.columns:
        plt.figure(figsize=(12,3))
        plt.plot(df['Date'], df['RSI_14'], label='RSI 14')
        plt.axhline(70, color='red', linestyle='--', linewidth=0.7)
        plt.axhline(30, color='green', linestyle='--', linewidth=0.7)
        plt.title(f'{TICKER} - RSI (14)')
        plt.xlabel('Fecha')
        plt.ylabel('RSI')
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_rsi.png")
        plt.show()

def plot_basic_stats(df):
    """Gráficas básicas de estadísticas"""
    
    # Precio y volumen
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(df['Date'], df['Close'])
    ax1.set_title(f'{TICKER} - Precio de Cierre')
    ax1.set_ylabel('USD')
    
    ax2.bar(df['Date'], df['Volume'], alpha=0.7)
    ax2.set_title('Volumen')
    ax2.set_ylabel('Volumen')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "basic_stats.png"))
    plt.show()