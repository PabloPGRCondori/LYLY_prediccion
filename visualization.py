import os
import matplotlib.pyplot as plt
import pandas as pd  # Importar pandas
from config import *

def plot_price_and_forecast(df, y_test_indexed, y_test, y_pred, save_prefix=os.path.join(PLOT_PATH, "price_forecast")):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Histórico Close', linewidth=1)
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
    plt.close()

def plot_indicators(df, save_prefix=os.path.join(PLOT_PATH, "indicators")):
    plt.figure(figsize=(12, 6))
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
    plt.close()

    if 'RSI_14' in df.columns:
        plt.figure(figsize=(12, 3))
        plt.plot(df['Date'], df['RSI_14'], label='RSI 14')
        plt.axhline(70, color='red', linestyle='--', linewidth=0.7)
        plt.axhline(30, color='green', linestyle='--', linewidth=0.7)
        plt.title(f'{TICKER} - RSI (14)')
        plt.xlabel('Fecha')
        plt.ylabel('RSI')
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_rsi.png")
        plt.close()

def plot_basic_stats(df, save_prefix=os.path.join(PLOT_PATH, "basic_stats")):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df['Date'], df['Close'], label='Close', color='tab:blue')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Precio (USD)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title(f'{TICKER} - Precio de Cierre y Volumen')

    ax2 = ax1.twinx()
    ax2.bar(df['Date'], df['Volume'], alpha=0.3, color='tab:orange', label='Volumen')
    ax2.set_ylabel('Volumen', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    fig.savefig(f"{save_prefix}.png")
    plt.close(fig)

def plot_sentiment_vs_price(big_data_df, price_df, save_path=os.path.join(PLOT_PATH, "impacto_noticias_meta.png")):
    """
    Genera un gráfico de doble eje:
    - Eje Y Izquierdo: Precio de Cierre (Close) de META (Línea)
    - Eje Y Derecho: Sentimiento Promedio (Avg_Sentiment) (Barras)
    """
    # Asegurar que ambos DataFrames tengan 'Date' como datetime y ordenado
    # Hacer copia para no modificar los originales
    bd_df = big_data_df.copy()
    pr_df = price_df.copy()
    
    bd_df['Date'] = pd.to_datetime(bd_df['Date'])
    pr_df['Date'] = pd.to_datetime(pr_df['Date'])
    
    # Filtrar price_df para cubrir solo el rango de fechas de big_data_df (más un margen si se desea)
    min_date = bd_df['Date'].min()
    max_date = bd_df['Date'].max()
    
    pr_df = pr_df[(pr_df['Date'] >= min_date) & (pr_df['Date'] <= max_date)]
    
    # Merge para alinear las fechas perfectamente
    merged_df = pd.merge(pr_df[['Date', 'Close']], bd_df[['Date', 'Avg_Sentiment']], on='Date', how='inner')
    
    if merged_df.empty:
        print("⚠️ No hay intersección de fechas entre precios y noticias para graficar.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Eje Y Izquierdo: Precio (Línea)
    color_price = 'tab:blue'
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Precio META (USD)', color=color_price)
    ax1.plot(merged_df['Date'], merged_df['Close'], color=color_price, marker='o', label='Precio Cierre')
    ax1.tick_params(axis='y', labelcolor=color_price)
    ax1.grid(True, alpha=0.3)

    # Eje Y Derecho: Sentimiento (Barras)
    ax2 = ax1.twinx()  # instanciar un segundo eje que comparte el mismo eje x
    color_sent = 'tab:red'
    ax2.set_ylabel('Sentimiento Promedio (GDELT)', color=color_sent)
    # Usamos barras con cierta transparencia
    ax2.bar(merged_df['Date'], merged_df['Avg_Sentiment'], color=color_sent, alpha=0.5, width=0.5, label='Sentimiento')
    ax2.tick_params(axis='y', labelcolor=color_sent)
    
    # Línea de referencia en 0 para sentimiento
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    plt.title(f'Impacto de Noticias en Precio: {TICKER} vs Sentimiento GDELT')
    fig.tight_layout()  # para que no se corten las etiquetas
    
    print(f"Generando gráfico de impacto: {save_path}")
    plt.savefig(save_path)
    plt.close(fig)