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

    # --- PEGAR ESTO AL FINAL DE visualization.py ---

def plot_big_data_dashboard(sentiment_score, headlines, keywords, external_factors, save_path=os.path.join(PLOT_PATH, "big_data_dashboard.png")):
    """
    Genera un Dashboard 360°: Sentimiento, Tópicos, Riesgo y Macro.
    """
    import matplotlib.pyplot as plt

    # Configuración del Canvas (Lienzo)
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'INTELIGENCIA DE MERCADO 360°: {TICKER}', fontsize=20, fontweight='bold', color='#003366')
    
    # Diseño de cuadrícula: 2 filas, 2 columnas
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    
    # --- PANEL 1: NARRATIVA (SENTIMIENTO) ---
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Color semáforo
    if sentiment_score > 0.05: col, txt = 'green', 'OPTIMISTA'
    elif sentiment_score < -0.05: col, txt = 'red', 'PESIMISTA'
    else: col, txt = 'gray', 'NEUTRAL'
    
    ax1.barh([0], [sentiment_score], color=col, height=0.4)
    ax1.set_xlim(-1, 1)
    ax1.set_title(f'Narrativa de Medios\nEstado: {txt} ({sentiment_score:.2f})', fontweight='bold')
    ax1.axvline(0, color='black', linewidth=0.8)
    ax1.set_yticks([])
    ax1.set_xlabel('Negativo <---> Positivo')
    
    # Poner los titulares debajo
    y_start = -0.6
    ax1.text(0, y_start, "TITULARES ANALIZADOS:", fontsize=9, fontweight='bold', ha='center')
    for i, h in enumerate(headlines[:4]):
        clean = h.split(') ')[-1][:50] + "..."
        score = h.split(') ')[0] + ')'
        ax1.text(0, y_start - 0.15 - (i*0.12), f"{score} {clean}", ha='center', fontsize=8, 
                 bbox=dict(facecolor='white', edgecolor='#ddd', boxstyle='round'))
    ax1.set_ylim(-1.5, 0.5)

    # --- PANEL 2: TEMÁTICA (DE QUÉ HABLAN) ---
    ax2 = fig.add_subplot(gs[0, 1])
    if keywords:
        words, counts = zip(*keywords)
        bars = ax2.barh(words, counts, color='#3498db')
        ax2.set_title('Top Tópicos en Noticias\n(¿Qué mueve el mercado?)', fontweight='bold')
        ax2.set_xlabel('Frecuencia de mención')
        ax2.invert_yaxis() # Para que el más frecuente salga arriba
    else:
        ax2.text(0.5, 0.5, "Sin datos de tópicos", ha='center')

    # --- PANEL 3: PSICOLOGÍA (MIEDO/AVARICIA - VIX) ---
    ax3 = fig.add_subplot(gs[1, 0])
    vix = external_factors.get('VIX', 0)
    
    # Zonas de color
    ax3.axhspan(0, 15, color='green', alpha=0.1) # Complacencia
    ax3.axhspan(15, 25, color='yellow', alpha=0.1) # Normal
    ax3.axhspan(25, 60, color='red', alpha=0.1) # Pánico
    
    ax3.bar(['VIX (Miedo)'], [vix], color='orange' if vix < 25 else 'red', width=0.3)
    ax3.text(0, vix+1, f"{vix:.2f}", ha='center', fontweight='bold')
    ax3.set_ylim(0, 40)
    ax3.set_title('Psicología de Mercado (VIX)', fontweight='bold')
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    
    # --- PANEL 4: FUNDAMENTAL MACRO (COBRE) ---
    ax4 = fig.add_subplot(gs[1, 1])
    cobre = external_factors.get('Cobre', 0)
    
    ax4.plot(['Ref.', 'Actual'], [4.0, cobre], marker='o', color='#b87333', linewidth=3, markersize=12)
    ax4.axhline(4.5, color='green', linestyle='--', label='Boom Económico ($4.5)')
    ax4.axhline(3.5, color='red', linestyle='--', label='Recesión ($3.5)')
    
    ax4.text(1, cobre+0.1, f"${cobre:.2f}", ha='center', fontweight='bold', color='#b87333')
    ax4.set_ylim(3, 6)
    ax4.legend()
    ax4.set_title('Salud Económica Global (Cobre)', fontweight='bold')
    ax4.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"📊 Dashboard 360 generado: {save_path}")
    return save_path