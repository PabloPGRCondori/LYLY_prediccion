import os
import matplotlib.pyplot as plt
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